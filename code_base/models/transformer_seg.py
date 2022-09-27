import torch
import torch.nn as nn
import torchvision
from transformers import PretrainedConfig, SegformerForSemanticSegmentation

from .transformer_configs import SEGFORMERS


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_classes: int,
        pretrained: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        else:
            config = PretrainedConfig(**SEGFORMERS[model_name])
            self.model = SegformerForSemanticSegmentation(config)

        if model_name in [
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "nvidia/segformer-b5-finetuned-ade-640-640",
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        ]:
            self.model.decode_head.classifier = nn.Conv2d(
                in_channels=self.model.decode_head.classifier.in_channels,
                out_channels=n_classes,
                kernel_size=self.model.decode_head.classifier.kernel_size,
                stride=self.model.decode_head.classifier.stride,
            )
        else:
            raise ValueError(f"{model_name} is not supported")
        resize_size = model_name.split("-")[-2:]
        resize_size = tuple([int(el) for el in resize_size])
        self.resize_step = torchvision.transforms.Resize(resize_size)
        self.to(device)
        self.device = device

    def forward(self, input, **kwargs):
        output = self.model(pixel_values=input).logits
        if self.training:
            return output
        else:
            return self.resize_step(output)
