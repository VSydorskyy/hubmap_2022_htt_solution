from pprint import pprint
from typing import Dict, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from ..constants import CLASSES
from .point_rend.point_rend import PointRend

try:
    from .deeplabv3_gc import DeepLabV3GC, DeepLabV3PlusGC
    from .fpn_gc import FPNGC
    from .unet_asym import UnetAsymmetric
    from .unet_gc import UnetGC
    from .unetplusplus_gc import UnetPlusPlusGC
    from .unetx3 import UnetMultiHead
except:
    print("Custom SMP modifications were not imported. Newer version of SMP")


class SMPWrapper(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        arch_name: str = "FPN",
        pretrained: bool = True,
        in_channels: int = 3,
        activation: Optional[str] = None,
        aux_params: Optional[Dict] = None,
        return_only_mask: bool = False,
        device: str = "cpu",
        add_smp_kwargs: Optional[Dict] = None,
        case_embedding_dim: Optional[int] = None,
        use_slice_idx: bool = False,
        point_rand_config: Optional[dict] = None,
    ):
        super().__init__()
        self.return_only_mask = return_only_mask
        add_smp_kwargs = {} if add_smp_kwargs is None else add_smp_kwargs
        self.use_pure_gc_id = False
        self.use_point_rand = point_rand_config is not None

        if arch_name == "FPN":
            arch = smp.FPN
        elif arch_name == "Unet":
            arch = smp.Unet
        elif arch_name == "UnetMultiHead":
            arch = UnetMultiHead
            self.use_pure_gc_id = True
        elif arch_name == "UnetAsymmetric":
            arch = UnetAsymmetric
        elif arch_name == "UnetPP":
            arch = smp.UnetPlusPlus
        elif arch_name == "DeepLabV3":
            arch = smp.DeepLabV3
        elif arch_name == "DeepLabV3P":
            arch = smp.DeepLabV3Plus
        elif arch_name == "UnetGC":
            arch = UnetGC
        elif arch_name == "FPNGC":
            arch = FPNGC
        elif arch_name == "UnetPPGC":
            arch = UnetPlusPlusGC
        elif arch_name == "DeepLabV3GC":
            arch = DeepLabV3GC
        elif arch_name == "DeepLabV3PGC":
            arch = DeepLabV3PlusGC
        else:
            raise ValueError(f"Unknown architecture {arch_name}")

        if "GC" in arch_name:
            add_smp_kwargs["gc_dim"] = 0
            if case_embedding_dim is not None:
                add_smp_kwargs["gc_dim"] += case_embedding_dim
            if use_slice_idx:
                add_smp_kwargs["gc_dim"] += 1

        if isinstance(pretrained, str):
            encoder_weights = pretrained
        else:
            encoder_weights = "imagenet" if pretrained else None
        print(f"Using {encoder_weights} encoder weights")
        print(f"Additional SMP kwargs :")
        pprint(add_smp_kwargs)

        if self.use_point_rand:
            print("Wrapping model into Point Rand")
            backbone = arch(
                encoder_name=backbone_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=activation,
                aux_params=aux_params,
                **add_smp_kwargs,
            )
            self.model = PointRend(backbone=backbone, **point_rand_config)
        else:
            self.model = arch(
                encoder_name=backbone_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=activation,
                aux_params=aux_params,
                **add_smp_kwargs,
            )
        if case_embedding_dim is not None:
            self.case_embedding = nn.Embedding(len(CLASSES), case_embedding_dim)
        else:
            self.case_embedding = None
        self.use_slice_idx = use_slice_idx
        self.device = device
        self.to(self.device)

    def freeze_unfreeze_encoder(self, freeze=True):
        if self.use_point_rand:
            for param in self.model.backbone.encoder.parameters():
                param.requires_grad = not freeze
        else:
            for param in self.model.encoder.parameters():
                param.requires_grad = not freeze

    def forward(self, x, case=None, scan=None):
        gc = []
        if self.use_slice_idx:
            gc.append(scan.unsqueeze(dim=-1))
        if self.case_embedding is not None:
            gc.append(self.case_embedding(case))
        if len(gc) > 0:
            gc = torch.cat(gc, dim=-1)
            x = self.model(x, gc)
        elif self.use_pure_gc_id:
            x = self.model(x, case)
        else:
            x = self.model(x)
        if self.return_only_mask:
            # Handle AUX case
            if isinstance(x, tuple):
                x = x[0]
        return x
