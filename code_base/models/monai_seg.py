from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR, UNet
except:
    print("monai package was not imported")


class MonaiWrapper(nn.Module):
    def __init__(
        self,
        monai_kwargs: Dict,
        arch_name: str = "SwinUNETR",
        pretrained: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()
        if arch_name == "SwinUNETR":
            arch = SwinUNETR
        elif arch_name == "UNet":
            arch = UNet
        else:
            raise ValueError(f"Unknown architecture {arch_name}")

        self.model = arch(**monai_kwargs)
        if pretrained is not None:
            print(f"Loading checkpoint from {pretrained}")
            self.model.load_from(torch.load(pretrained, map_location="cpu"))
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.model(x)
        return x
