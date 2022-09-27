# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter

from code_base.models.point_rend.point_rend_utils import (
    inference_sampling_points,
    point_sample,
    training_sampling_points,
)


class PointHead(nn.Module):
    def __init__(self, in_c=163, num_classes=3, k=3, beta=0.75, binary=True, refine_inference=False, verbose=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_c, 128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(128, 128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(128, num_classes, 1),
        )
        self.k = k
        self.beta = beta
        self.stride = 16
        self.binary = binary
        self.refine_inference = refine_inference
        self.verbose = verbose

    def forward(self, x, res2, out):
        """
        1. Fine-grained features are interpolated from the second stage of the network
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        if not self.training:
            return self.inference(x, res2, out)

        num_points = x.shape[-1] // self.stride
        points = training_sampling_points(out, num_points * num_points, self.k, self.beta, binary=self.binary)
        res2 = res2.to(out, non_blocking=True)

        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(res2, points, align_corners=False)

        feature_representation = torch.cat([fine, coarse], dim=1)

        rend = self.mlp(feature_representation)

        return {"coarse": out, "rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, x, res2, out):
        """
        During inference, subdivision uses N=1024
        (i.e., the number of points in the stride 16 map of a 512×512 image)
        """
        num_points = 1024

        if self.refine_inference:
            if self.verbose:
                print("Refine Inference")
            points_idx, points = inference_sampling_points(out, num_points, binary=self.binary)

            coarse = point_sample(out, points, align_corners=False)
            fine = point_sample(res2, points, align_corners=False)

            feature_representation = torch.cat([fine, coarse], dim=1)

            rend = self.mlp(feature_representation)
            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            out = out.reshape(B, C, -1)
            out = out.scatter(2, points_idx, rend)
            out = out.view(B, C, H, W)
        return out


class PointRend(nn.Module):
    def __init__(
        self, backbone, in_ch, num_classes, binary=True, backbone_type="resnet", refine_inference=False, verbose=False
    ):
        super().__init__()
        self.backbone = backbone
        self.layer2 = {}
        assert backbone_type in ["resnet", "effnet"]
        self.backbone_type = backbone_type
        if self.backbone_type == "resnet":
            self.backbone.encoder.layer2.register_forward_hook(self.get_features("layer2"))
            self.layername = "layer2"
        if self.backbone_type == "effnet":
            self.backbone.encoder.blocks[3].register_forward_hook(self.get_features("blocks3"))
            self.layername = "blocks3"
        self.head = PointHead(in_ch, num_classes, binary=binary, refine_inference=refine_inference, verbose=verbose)

    def get_features(self, name):
        def hook(model, input, output):
            self.layer2[name] = output.detach()

        return hook

    def forward(self, x):
        output = self.backbone(x)

        res = self.head(x, self.layer2[self.layername], output)
        return res


# if __name__ == "__main__":
#     import segmentation_models_pytorch as smp
#     unet = smp.create_model("unet")
#     x = torch.rand(1, 3, 224, 224)
#     model = PointRend(backbone=unet, in_ch=129, num_classes=1)
#     print(model(x))
