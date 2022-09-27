from gc import freeze

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from monai.inferers import sliding_window_inference

from ..constants import CLASSES
from ..models.point_rend.point_rend import point_sample
from ..utils.other import get_mode_model


class SMPHubMapForward(nn.Module):
    """
    Order of AUX tasks:
    1. Contour (if defined)
    2. Igor Overlap (if defined)
    3. Centroids (if defined)
    """

    def __init__(
        self,
        loss_functions,
        clf_loss_coef=None,
        sliding_inference_config=None,
        # Required for segformers
        mask_resize=None,
        selected_multichannel=False,
        freeze_encoder_for_epochs=None,
        contour_mask_coef=None,
        contour_mask_loss_functions=None,
        igor_masks_overlap_coef=None,
        igor_masks_overlap_loss_functions=None,
        centroids_coef=None,
        centroids_loss_functions=None,
        label_smoothing_coef=None,
    ):
        super().__init__()
        if clf_loss_coef is not None and sliding_inference_config is not None:
            raise ValueError("sliding window does not support additional inputs or outputs")
        assert label_smoothing_coef is None or 0 < label_smoothing_coef < 0.5
        self.sliding_inference_config = sliding_inference_config
        self.loss_functions = loss_functions
        self.clf_loss_coef = clf_loss_coef
        self.selected_multichannel = selected_multichannel
        self.freeze_encoder_for_epochs = freeze_encoder_for_epochs
        self.contour_mask_coef = contour_mask_coef
        self.contour_mask_loss_functions = contour_mask_loss_functions
        self.igor_masks_overlap_coef = igor_masks_overlap_coef
        self.igor_masks_overlap_loss_functions = igor_masks_overlap_loss_functions
        self.centroids_coef = centroids_coef
        self.centroids_loss_functions = centroids_loss_functions
        self.label_smoothing_coef = label_smoothing_coef
        if self.clf_loss_coef is not None:
            self.clf_loss_func = nn.CrossEntropyLoss()
        if mask_resize is not None:
            self.mask_resize = torchvision.transforms.Resize(
                mask_resize, interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )
            self.is_first_call = True
        else:
            self.mask_resize = None

    def on_epoch_start(self, runner):
        if self.freeze_encoder_for_epochs is not None:
            if runner.epoch_step == 0:
                runner.model.freeze_unfreeze_encoder(freeze=True)
                print("Encoder freezed!")
            # Because first epoch is 0, so if we want freeze for 1 epoch we need
            # to unfreeze on epoch number 1, if for 2 - we need to unfreeze on epoch
            # number 2 and so on
            elif runner.epoch_step == self.freeze_encoder_for_epochs:
                runner.model.freeze_unfreeze_encoder(freeze=False)
                print("Encoder UNfreezed!")

    def _compute_all_losses(self, y_true, y_pred, prefix="", loss_functions=None):
        sum_loss = 0.0
        losses = dict()
        for loss_name, loss_func in self.loss_functions.items() if loss_functions is None else loss_functions.items():
            if isinstance(loss_func, dict):
                losses[prefix + loss_name] = loss_func["func"](y_pred, y_true) * loss_func["coef"]
            else:
                losses[prefix + loss_name] = loss_func(y_pred, y_true)
            sum_loss += losses[prefix + loss_name]
        losses[prefix + "loss"] = sum_loss
        return losses

    def forward(self, runner, batch):

        image, mask, organ, pixel_size = batch
        if self.mask_resize is not None and runner.is_train_loader:
            if self.is_first_call:
                self.mask_resize.to(mask.device)
                self.is_first_call = False
            mask = self.mask_resize(mask)
        if runner.is_train_loader and self.label_smoothing_coef is not None:
            mask = torch.abs(mask - self.label_smoothing_coef)
        # Creating np.array at first is recommended by torch
        organ_ids = torch.LongTensor(np.array([CLASSES.index(o) for o in organ])).to(image.device)
        mode = "train" if runner.loader_key == "train" else "val"
        model = get_mode_model(runner.model, mode=mode)
        if self.clf_loss_coef is not None:
            pred_mask, pred_class = model(image, case=organ_ids, scan=pixel_size)
        else:
            # We need sliding window here in order to handle inference with large models
            use_sliding = (
                self.sliding_inference_config is not None
                and not runner.is_train_loader
                and (
                    image.shape[2] > self.sliding_inference_config["roi_size"][0]
                    or image.shape[3] > self.sliding_inference_config["roi_size"][1]
                )
            )
            if use_sliding:
                pred_mask = sliding_window_inference(inputs=image, predictor=model, **self.sliding_inference_config)
            else:
                pred_mask = model(image, case=organ_ids, scan=pixel_size)

        if self.selected_multichannel:
            if (
                self.contour_mask_coef is not None
                or self.igor_masks_overlap_coef is not None
                or self.centroids_coef is not None
            ):
                n_aux_tasks = (
                    int(self.contour_mask_coef is not None)
                    + int(self.igor_masks_overlap_coef is not None)
                    + int(self.centroids_coef is not None)
                )
                # All AUX masks + main mask
                class_step_size = n_aux_tasks + 1
                pred_mask = torch.stack(
                    [
                        pred_mask[
                            el_id,
                            CLASSES.index(o) * class_step_size : CLASSES.index(o) * class_step_size + class_step_size,
                        ]
                        for el_id, o in enumerate(organ)
                    ]
                )
            else:
                pred_mask = torch.stack(
                    [pred_mask[el_id, CLASSES.index(o)] for el_id, o in enumerate(organ)]
                ).unsqueeze(dim=1)
        if (
            self.contour_mask_coef is not None
            or self.igor_masks_overlap_coef is not None
            or self.centroids_coef is not None
        ):
            # .contiguous() required for focal loss
            losses = self._compute_all_losses(mask[:, :1].contiguous(), pred_mask[:, :1].contiguous())
            ch_pointer = 1
            if self.contour_mask_coef is not None:
                contour_losses = self._compute_all_losses(
                    mask[:, ch_pointer : ch_pointer + 1].contiguous(),
                    pred_mask[:, ch_pointer : ch_pointer + 1].contiguous(),
                    prefix="contour_",
                    loss_functions=self.contour_mask_loss_functions
                    if self.contour_mask_loss_functions is not None
                    else None,
                )
                losses["loss"] += contour_losses.pop("contour_loss") * self.contour_mask_coef
                losses.update(contour_losses)
                ch_pointer += 1
            if self.igor_masks_overlap_coef is not None:
                imo_losses = self._compute_all_losses(
                    mask[:, ch_pointer : ch_pointer + 1].contiguous(),
                    pred_mask[:, ch_pointer : ch_pointer + 1].contiguous(),
                    prefix="IMO_",
                    loss_functions=self.igor_masks_overlap_loss_functions
                    if self.igor_masks_overlap_loss_functions is not None
                    else None,
                )
                losses["loss"] += imo_losses.pop("IMO_loss") * self.igor_masks_overlap_coef
                losses.update(imo_losses)
                ch_pointer += 1
            if self.centroids_coef is not None:
                centroid_losses = self._compute_all_losses(
                    mask[:, ch_pointer : ch_pointer + 1].contiguous(),
                    pred_mask[:, ch_pointer : ch_pointer + 1].contiguous(),
                    prefix="centroid_",
                    loss_functions=self.centroids_loss_functions if self.centroids_loss_functions is not None else None,
                )
                losses["loss"] += centroid_losses.pop("centroid_loss") * self.centroids_coef
                losses.update(centroid_losses)
            # Extract only main mask for metric computation
            mask = mask[:, :1].contiguous()
            pred_mask = pred_mask[:, :1].contiguous()
        else:
            losses = self._compute_all_losses(mask, pred_mask)

        if self.clf_loss_coef is not None:
            clf_loss = self.clf_loss_func(pred_class, organ_ids)
            losses["clf_loss"] = clf_loss
            losses["loss"] += clf_loss * self.clf_loss_coef

        inputs = {"mask": mask, "organ": organ}
        output = {"mask": pred_mask}

        return losses, inputs, output


class PointRandSMPHubMapForward(SMPHubMapForward):
    def __init__(self, loss_functions, freeze_encoder_for_epochs=None, point_rand_loss_coef=1.0):
        super().__init__(
            loss_functions=loss_functions,
            clf_loss_coef=None,
            sliding_inference_config=None,
            # Required for segformers
            mask_resize=None,
            selected_multichannel=False,
            freeze_encoder_for_epochs=freeze_encoder_for_epochs,
            contour_mask_coef=None,
            contour_mask_loss_functions=None,
            igor_masks_overlap_coef=None,
            igor_masks_overlap_loss_functions=None,
            centroids_coef=None,
            centroids_loss_functions=None,
            label_smoothing_coef=None,
        )
        self.point_rand_loss_coef = point_rand_loss_coef

    def forward(self, runner, batch):
        image, mask, organ, _ = batch

        mode = "train" if runner.loader_key == "train" else "val"
        model = get_mode_model(runner.model, mode=mode)
        if runner.is_train_loader:
            result = model(image)
            pred_mask = result.pop("coarse")
            gt_points = point_sample(mask, result["points"], mode="nearest", align_corners=False)
            points_loss = F.binary_cross_entropy_with_logits(result["rend"], gt_points)
        else:
            pred_mask = model(image)

        losses = self._compute_all_losses(mask, pred_mask)
        if runner.is_train_loader:
            losses["points_loss"] = points_loss
            losses["loss"] += points_loss * self.point_rand_loss_coef

        inputs = {"mask": mask, "organ": organ}
        output = {"mask": pred_mask}

        return losses, inputs, output
