from typing import List

import timm
import torch
from pytorch_toolbelt.modules import DecoderModule, EncoderModule, make_n_channel_input


def _take(elements: List[torch.Tensor], indexes):
    return list([elements[i] for i in indexes])


class TimmEncoder(EncoderModule):
    def __init__(self, timm_model_name, in_channels=1, pretrained=True, layers=None):
        if pretrained:
            print("Pretrained NN")
        timm_model = timm.create_model(
            timm_model_name, pretrained=pretrained, features_only=True, in_chans=in_channels, out_indices=layers
        )
        filters = timm_model.feature_info.channels()
        strides = timm_model.feature_info.reduction()
        # if layers is None:
        layers = range(len(filters))
        super().__init__(filters, strides, layers)
        self.timm_model = timm_model
        self.layers = layers

    @property
    def encoder_layers(self):
        return self._layers

    def forward(self, x):
        output_features = self.timm_model(x)
        return _take(output_features, self._layers)
