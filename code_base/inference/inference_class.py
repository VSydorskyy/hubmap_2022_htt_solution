import os
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from scipy.ndimage import binary_fill_holes
from torch.cuda.amp import autocast
from tqdm import tqdm

from ..constants import CLASSES
from ..utils.mask import delete_small_segments_from_mask, get_pads, rle_encode_less_memory


def is_05(x):
    return abs(x - 0.5) < 1e-4


def torch_cat_with_none(tensor_1, tensor_2):
    if tensor_1 is None:
        return tensor_2
    elif tensor_2 is None:
        return tensor_1
    else:
        return np.concatenate([tensor_1, tensor_2])


def is_equal_with_type(el_1, el_2):
    if type(el_1) == type(el_2):
        return el_1 == el_2
    else:
        return False


class HubMapInference:
    def __init__(self, device, verbose=True, verbose_tqdm=True, use_sigmoid=True, use_amp=True):
        self.device = device
        self.verbose = verbose
        self.verbose_tqdm = verbose_tqdm
        self.use_sigmoid = use_sigmoid
        self.use_amp = use_amp

    def _apply_pred_act(self, input):
        # Handle aux out
        if isinstance(input, tuple):
            input, label = input
            if self.use_sigmoid:
                label = torch.sigmoid(label)

        if self.use_sigmoid:
            input = torch.sigmoid(input)

        return input

    def _tqdm_v(self, generator):
        if self.verbose_tqdm:
            return tqdm(generator)
        else:
            return generator

    def _print_v(self, msg):
        if self.verbose:
            print(msg)

    @torch.no_grad()
    def _predict(self, model, input, organ_ids, pixel_size, sliding_window_config):
        if self.use_amp:
            with autocast():
                if sliding_window_config is None:
                    pred = model(input.to(self.device), case=organ_ids.to(self.device), scan=pixel_size.to(self.device))
                else:
                    pred = sliding_window_inference(
                        inputs=input.to(self.device), predictor=model, **sliding_window_config
                    )
        else:
            if sliding_window_config is None:
                pred = model(input.to(self.device), case=organ_ids.to(self.device), scan=pixel_size.to(self.device))
            else:
                pred = sliding_window_inference(inputs=input.to(self.device), predictor=model, **sliding_window_config)
        # Cast back to float32 before sigmoid
        pred = self._apply_pred_act(pred.detach().float())
        pred = pred.cpu().numpy()
        torch.cuda.empty_cache()
        return pred

    @staticmethod
    def _postprocess_and_extract_rle(
        pred_mask,
        cls_id,
        original_height,
        original_width,
        scale_height,
        scale_width,
        pad_config,
        scale_back,
        tresh,
        min_area,
        fill_binary_holes,
        print_input_shape,
        is_relative_min_area,
        save_mask_path,
        im_id,
    ):
        pads = get_pads(original_height=original_height, original_width=original_width, **pad_config)
        if pads["pad_top"] > 0:
            pred_mask = pred_mask[pads["pad_top"] :]
        if pads["pad_bottom"] > 0:
            pred_mask = pred_mask[: -pads["pad_bottom"]]
        if pads["pad_left"] > 0:
            pred_mask = pred_mask[:, pads["pad_left"] :]
        if pads["pad_right"] > 0:
            pred_mask = pred_mask[:, : -pads["pad_right"]]

        if scale_back:
            pred_mask = cv2.resize(pred_mask, (scale_width, scale_height), interpolation=cv2.INTER_AREA)
        if save_mask_path is not None:
            cv2.imwrite(os.path.join(save_mask_path, str(im_id)) + ".png", (pred_mask * 255).astype(np.uint8))
        cur_tresh = tresh if isinstance(tresh, (int, float)) else tresh[cls_id]
        if print_input_shape and not isinstance(tresh, (int, float)):
            print(f"Using tresh {cur_tresh}")
        pred_mask = pred_mask > cur_tresh
        if fill_binary_holes:
            pred_mask = binary_fill_holes(pred_mask)
        if min_area is not None and pred_mask.sum() > 0:
            cls_min_area = min_area if isinstance(min_area, int) else min_area[cls_id]
            if is_relative_min_area:
                cls_min_area = int(np.prod(pred_mask.shape) * cls_min_area)
            if print_input_shape:
                print(f"Removing masks with area less then : {cls_min_area}")
            if cls_min_area > 0:
                pred_mask = delete_small_segments_from_mask(bool_mask=pred_mask, min_size=cls_min_area)
        return rle_encode_less_memory(pred_mask)

    @torch.no_grad()
    def predict_test_loader(
        self,
        nn_models,
        test_loader,
        tresh,
        pad_config,
        min_area=None,
        is_relative_min_area=False,
        model_coefs=None,
        use_rescaled=True,
        scale_back=False,
        fill_binary_holes=False,
        print_input_shape=False,
        sliding_window_config=None,
        save_mask_path=None,
        mean_type="mean",
    ):
        assert mean_type in ["mean", "gmean", "tsharpen"]
        if model_coefs is not None and mean_type != "mean":
            raise ValueError("model_coefs are supported only for `mean` mean_type")
        self._print_v(f"Using tresh : {tresh}")
        self._print_v(f"Using min_area : {min_area}. relative - {is_relative_min_area}")
        self._print_v(f"Using model coefs: {model_coefs}")
        self._print_v(f"Using save_mask_path: {save_mask_path}")
        self._print_v(f"Using mean_type: {mean_type}")
        if save_mask_path is not None:
            os.makedirs(save_mask_path)
        if model_coefs is not None:
            model_coefs = np.array(model_coefs)[:, None, None]
        predicted_df = {"id": [], "rle": []}

        # Patients prints make progress bar ugly :(
        for batch in tqdm(test_loader):
            # Unwrap batch
            image, _, organ, h, w, rescaled_h, rescaled_w, im_id, pixel_size = batch
            # Creating np.array at first is recommended by torch
            organ_ids = torch.LongTensor(np.array([CLASSES.index(o) for o in organ]))
            if print_input_shape:
                self._print_v(f"Input Image shape = {image.shape}")
            assert image.shape[0] == 1, "for now supports only bs 1"
            cls_id = CLASSES.index(organ[0])
            # Average over models
            pred_masks = []
            for model_idx, nn_model in enumerate(nn_models):
                if sliding_window_config is not None:
                    if isinstance(sliding_window_config, list):
                        current_sliding_window_config = sliding_window_config[model_idx]
                    elif isinstance(sliding_window_config, dict):
                        current_sliding_window_config = sliding_window_config
                    else:
                        raise ValueError(f"{type(sliding_window_config)} is invalid type for sliding_window_config")
                    # For small images we should not use slicing (if it is enabled)
                    if current_sliding_window_config is not None and (
                        image.shape[2] > current_sliding_window_config["roi_size"][0]
                        or image.shape[3] > current_sliding_window_config["roi_size"][1]
                    ):
                        if print_input_shape:
                            self._print_v("Using sliding window")
                    else:
                        current_sliding_window_config = None
                else:
                    current_sliding_window_config = None
                if print_input_shape:
                    # SMPWrapper case
                    if hasattr(nn_model.model, "encoder"):
                        print(
                            f"Model idx : {model_idx}. "
                            f"Encoder: {type(nn_model.model.encoder).__name__}. "
                            f"Decoder: {type(nn_model.model.decoder).__name__}. "
                        )
                    # PointRand(SMPWrapper) case
                    elif hasattr(nn_model.model, "backbone"):
                        print(
                            f"Model idx : {model_idx}. "
                            f"Encoder: {type(nn_model.model.backbone.encoder).__name__}. "
                            f"Decoder: {type(nn_model.model.backbone.decoder).__name__}. "
                        )
                    # TTAWrapper(SMPWrapper) case
                    elif hasattr(nn_model.model, "model"):
                        # TTAWrapper(PointRand(SMPWrapper)) case
                        if hasattr(nn_model.model.model, "backbone"):
                            print(
                                f"Model idx : {model_idx}. "
                                f"Encoder: {type(nn_model.model.model.backbone.encoder).__name__}. "
                                f"Decoder: {type(nn_model.model.model.backbone.decoder).__name__}. "
                            )
                        # TTAWrapper(SMPWrapper) case
                        elif hasattr(nn_model.model.model, "encoder"):
                            print(
                                f"Model idx : {model_idx}. "
                                f"Encoder: {type(nn_model.model.model.encoder).__name__}. "
                                f"Decoder: {type(nn_model.model.model.decoder).__name__}. "
                            )
                    else:
                        raise ValueError("Unsupported model type")
                pred_mask = self._predict(
                    model=nn_model,
                    input=image,
                    organ_ids=organ_ids,
                    pixel_size=pixel_size,
                    sliding_window_config=current_sliding_window_config,
                )
                if pred_mask.shape[1] < len(CLASSES):
                    mask_cls_id = 0
                elif pred_mask.shape[1] == len(CLASSES):
                    mask_cls_id = cls_id
                else:
                    raise ValueError("Unsupported number of channels in predicted mask")
                pred_mask = pred_mask[0, mask_cls_id]
                pred_masks.append(pred_mask)
            if model_coefs is None:
                pred_masks = np.stack(pred_masks, axis=0)
                if mean_type == "mean":
                    pred_masks = pred_masks.mean(axis=0)
                elif mean_type == "gmean":
                    pred_masks = np.prod(pred_masks, axis=0) ** (1 / pred_masks.shape[0])
                elif mean_type == "tsharpen":
                    pred_masks = (pred_masks**0.5).mean(axis=0)
            else:
                pred_masks = (np.stack(pred_masks, axis=0) * model_coefs).sum(axis=0) / model_coefs.sum()
            # Unwrap bs 1
            h, w, rescaled_h, rescaled_w, im_id = (
                h[0].item(),
                w[0].item(),
                rescaled_h[0].item(),
                rescaled_w[0].item(),
                im_id[0] if isinstance(im_id[0], str) else im_id[0].item(),
            )

            predicted_df["id"].append(im_id)
            predicted_df["rle"].append(
                self._postprocess_and_extract_rle(
                    pred_mask=pred_masks,
                    cls_id=cls_id,
                    original_height=rescaled_h if use_rescaled else h,
                    original_width=rescaled_w if use_rescaled else w,
                    scale_height=h,
                    scale_width=w,
                    pad_config=pad_config,
                    scale_back=scale_back,
                    tresh=tresh,
                    min_area=min_area,
                    fill_binary_holes=fill_binary_holes,
                    print_input_shape=print_input_shape,
                    is_relative_min_area=is_relative_min_area,
                    save_mask_path=save_mask_path,
                    im_id=im_id,
                )
            )

        predicted_df = pd.DataFrame(predicted_df)

        return predicted_df
