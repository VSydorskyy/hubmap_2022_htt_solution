import re
from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class Clasifier(nn.Module):
    def __init__(
        self,
        nn_embed_size,
        classes_num,
        classifier_type,
        second_dropout_rate=None,
        hidden_dims=None,
        first_dropout_rate=None,
    ):
        super().__init__()

        if classifier_type == "relu":
            self.classifier = nn.Sequential(
                nn.Linear(nn_embed_size, hidden_dims),
                nn.ReLU(),
                nn.Dropout(p=first_dropout_rate),
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Dropout(p=second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "elu":
            self.classifier = nn.Sequential(
                nn.Dropout(first_dropout_rate),
                nn.Linear(nn_embed_size, hidden_dims),
                nn.ELU(),
                nn.Dropout(second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "swish":
            self.classifier = nn.Sequential(
                nn.Dropout(first_dropout_rate),
                nn.Linear(nn_embed_size, hidden_dims),
                Swish_module(),
                nn.Dropout(second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "dima":
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(nn_embed_size),
                nn.Linear(nn_embed_size, hidden_dims),
                nn.BatchNorm1d(hidden_dims),
                nn.PReLU(hidden_dims),
                nn.Dropout(p=second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "prelu":
            self.classifier = nn.Sequential(
                nn.Dropout(first_dropout_rate),
                nn.Linear(nn_embed_size, hidden_dims),
                nn.PReLU(hidden_dims),
                nn.Dropout(p=second_dropout_rate),
                nn.Linear(hidden_dims, classes_num),
            )
        elif classifier_type == "drop_linear":
            self.classifier = nn.Sequential(
                nn.Dropout(p=second_dropout_rate),
                nn.Linear(nn_embed_size, classes_num),
            )
        elif classifier_type == "identity":
            self.classifier = nn.Identity()
        elif classifier_type == "arcface_paper":
            self.classifier = nn.Sequential(
                nn.Dropout(second_dropout_rate),
                nn.Linear(nn_embed_size, classes_num),
                nn.BatchNorm1d(classes_num),
            )
        elif classifier_type == "option-D":
            self.classifier = nn.Sequential(
                nn.Linear(nn_embed_size, classes_num, bias=True),
                nn.BatchNorm1d(classes_num),
                torch.nn.PReLU(),
            )
        elif classifier_type == "drop-option-D":
            self.classifier = nn.Sequential(
                nn.Dropout(p=second_dropout_rate),
                nn.Linear(nn_embed_size, classes_num, bias=True),
                nn.BatchNorm1d(classes_num),
                torch.nn.PReLU(),
            )
        elif classifier_type == "option-S":
            self.classifier = nn.Sequential(nn.Linear(nn_embed_size, classes_num), Swish_module())
        else:
            raise ValueError("Invalid classifier_type")

    def forward(self, input):
        return self.classifier(input)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            print("GeM `p` is not trainable")
            self.p = p
        self.eps = eps
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.gem(x, p=self.p, eps=self.eps))

    def gem(self, x, p=3, eps=1e-5):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0] if isinstance(self.p, nn.Parameter) else self.p)
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class PoolingLayer(nn.Module):
    def __init__(self, pool_type: str, pool_type_kwargs: Dict[str, Any] = {}):
        super().__init__()

        self.pool_type = pool_type

        if self.pool_type == "AdaptiveAvgPool2d":
            self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pool_type == "GeM":
            self.pool_layer = GeM(**pool_type_kwargs)
        else:
            raise RuntimeError(f"{self.pool_type} is invalid pool_type")

    def forward(self, x):
        bs, ch, h, w = x.shape
        if self.pool_type == "AdaptiveAvgPool2d":
            x = self.pool_layer(x)
            x = x.view(bs, ch)
        elif self.pool_type == "GeM":
            x = self.pool_layer(x)
        return x


class TimmWrapper(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        in_channels: int = 5,
        device: str = "cpu",
        additional_timm_kwargs: Dict = {},
        pool_type: str = "AdaptiveAvgPool2d",
        pool_type_kwargs: Dict[str, Any] = {},
        classifier_kwargs: Dict[str, Any] = {
            "classifier_type": "elu",
            "first_dropout_rate": 0.4,
            "hidden_dims": 512,
            "second_dropout_rate": 0.2,
        },
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, in_chans=in_channels, features_only=True, **additional_timm_kwargs
        )
        self.pool = PoolingLayer(pool_type=pool_type, pool_type_kwargs=pool_type_kwargs)
        self.classifier = Clasifier(
            nn_embed_size=self.backbone.feature_info.channels()[-1], classes_num=num_classes, **classifier_kwargs
        )

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.pool(x)
        return self.classifier(x)


class TimmWrapperV2(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        in_channels: int = 5,
        device: str = "cpu",
        additional_timm_kwargs: Dict = {},
        pool_type: str = "AdaptiveAvgPool2d",
        pool_type_kwargs: Dict[str, Any] = {},
        classifier_kwargs: Dict[str, Any] = {
            "classifier_type": "elu",
            "first_dropout_rate": 0.4,
            "hidden_dims": 512,
            "second_dropout_rate": 0.2,
        },
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, in_chans=in_channels, **additional_timm_kwargs
        )
        self.pool = PoolingLayer(pool_type=pool_type, pool_type_kwargs=pool_type_kwargs)

        if re.match(r"tf_efficientnet_b.*_ns", backbone_name) is not None:
            nn_embed_size = self.backbone.classifier.in_features
        else:
            raise RuntimeError(f"For {backbone_name} backbone nn_embed_size is not defined")
        self.classifier = Clasifier(nn_embed_size=nn_embed_size, classes_num=num_classes, **classifier_kwargs)

        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        return self.classifier(x)
