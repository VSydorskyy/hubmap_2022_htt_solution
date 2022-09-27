from typing import Tuple

import numpy as np
import pandas as pd
import torch
from catalyst.dl import Callback, CallbackOrder

from ..constants import CLASSES
from ..utils.metrics import dice_coeff


class DiceMetric(Callback):
    def __init__(
        self,
        output_label_key: str = "mask",
        input_label_key: str = "mask",
        input_organ_key: str = "organ",
        loader_names: Tuple = ("valid"),
        use_sigmoid: bool = True,
        tresh: float = 0.5,
    ):
        super().__init__(CallbackOrder.Metric)

        self.output_label_key = output_label_key
        self.input_label_key = input_label_key
        self.input_organ_key = input_organ_key
        self.loader_names = loader_names
        self.use_sigmoid = use_sigmoid
        self.tresh = tresh

        self.running_pred_labels = []
        self.running_real_labels = []
        self.running_organs = []

    @staticmethod
    def _get_mask_channel_id(mask, cls_name):
        if mask.shape[0] == 1:
            return 0
        else:
            return CLASSES.index(cls_name)

    def on_batch_end(self, runner):
        if runner.loader_key in self.loader_names:
            y_hat_label = runner.output[self.output_label_key].detach()
            if self.use_sigmoid:
                y_hat_label = torch.sigmoid(y_hat_label)
            y_hat_label = y_hat_label.cpu().numpy() > self.tresh

            y_label = runner.input[self.input_label_key].detach()
            y_label = y_label.cpu().numpy() > 0.5

            organs = runner.input[self.input_organ_key]

            self.running_pred_labels.extend(
                [mask[self._get_mask_channel_id(mask, cls_name), :, :] for mask, cls_name in zip(y_hat_label, organs)]
            )
            self.running_real_labels.extend(
                [mask[self._get_mask_channel_id(mask, cls_name), :, :] for mask, cls_name in zip(y_label, organs)]
            )
            self.running_organs += organs

    def on_loader_end(self, runner):
        if runner.loader_key in self.loader_names:
            dices = [dice_coeff(gt, pr) for gt, pr in zip(self.running_real_labels, self.running_pred_labels)]
            score_df = pd.DataFrame({"score": dices, "organ": self.running_organs})
            organ_scores = score_df.groupby("organ")["score"].mean()

            runner.loader_metrics["dice_score"] = score_df["score"].mean()
            for organ, score in organ_scores.to_dict().items():
                runner.loader_metrics[f"{organ}_dice"] = score

            self.running_pred_labels = []
            self.running_real_labels = []
            self.running_organs = []
