import functools

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch.nn.utils.parametrizations import spectral_norm as spectral_normalization

from .attention import AttentionBlock, CBAMModule, ChannelLinearAttention, ECAModule, PositionLinearAttention, SEModule
from .upsample import BilinearUpsample2x, UpsampleBilinearAdditiveUpsample2x, UpsampleResidualDeconvolutionUpsample2x


class LNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class ConvNextBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, norm=LNorm, drop_path=0.0, layer_scale_init_value=1e-6, *args, **kwargs
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, padding_mode="reflect"
        )  # depthwise conv
        self.norm = norm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        if out_channels != in_channels:
            self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.project = nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = self.project(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x

        x = input + self.drop_path(x)
        return x


def Conv2dSN(*args, spectral_norm=False, bias=False, **kwargs):
    if spectral_norm:
        sn = lambda x: spectral_normalization(x)
    else:
        sn = lambda x: x
    return sn(nn.Conv2d(*args, bias=bias, **kwargs))


def Conv2dSN3x3(*args, in_channels, out_channels, stride=1, bias=False, spectral_norm=False, **kwargs):
    # print(f'Conv2dSN3x3 {bias}')
    return Conv2dSN(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        padding_mode="reflect",
        bias=bias,
        spectral_norm=spectral_norm,
        **kwargs
    )


def add_norm(norm, in_channels):
    if norm is None:
        return torch.nn.Identity()
    if type(norm) == functools.partial:
        if norm.func is nn.GroupNorm:
            return norm(num_channels=in_channels)
    return norm(in_channels)


class PreActBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        act=nn.Mish,
        spectral_norm=True,
        norm=None,
        bias=False,
        downsample="upsample",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = add_norm(norm, in_channels)
        self.act1 = act()
        self.downsample = nn.Identity()
        if stride == 2:
            self.shortcut = nn.Sequential(
                torch.nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True),
                # nn.AvgPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias),
            )

            if downsample == "upsample":
                # print('upsample')
                self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
                stride = 1
            if downsample == "avg":
                # print('avg')
                self.downsample = nn.AvgPool2d(2)
                stride = 1
            if downsample == "strided":
                # print('strided')
                pass

        elif in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias))

        self.conv1 = Conv2dSN3x3(
            in_channels=in_channels, out_channels=out_channels, stride=stride, spectral_norm=spectral_norm, bias=bias
        )
        self.norm2 = add_norm(norm, out_channels)
        self.act2 = act()
        self.conv2 = Conv2dSN3x3(
            in_channels=out_channels, out_channels=out_channels, spectral_norm=spectral_norm, bias=bias
        )

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, "shortcut") else x

        out = self.act1(self.norm1(self.downsample(x)))
        out = self.conv1(out)
        out = self.conv2(self.act2(self.norm2(out)))
        out += shortcut
        return out


class PreActBlock_Enc(nn.Module):  # depthwise inverted bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        act=nn.Mish,
        spectral_norm=True,
        norm=None,
        bias=False,
        downsample="upsample",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = add_norm(norm, in_channels)
        self.act1 = act()
        self.downsample = nn.Identity()
        if stride == 2:
            self.shortcut = nn.Sequential(
                torch.nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias),
            )

            if downsample == "upsample":
                # print('upsample')
                self.downsample = torch.nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
                stride = 1

            if downsample == "avg":
                # print('avg')
                self.downsample = nn.AvgPool2d(2)
                stride = 1

            if downsample == "strided":
                # print('strided')
                pass

            self.downsample = DownsampleLayernorm(in_channels=in_channels, out_channels=out_channels)

        elif in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias))

        # print('In-out', in_channels, out_channels)
        self.block = Block(dim=out_channels)

    def forward(self, x):
        out = self.downsample(x)
        out = self.block(out)
        return out


class DownsampleLayernorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.ln = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=2,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode="reflect"
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x

        x = input + self.drop_path(x)
        return x


class PostActBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, act=nn.Mish, spectral_norm=True, norm=None, bias=False, se=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.se = None
        self.downsample = None
        self.project = None

        # print(f'in_channels {in_channels} out_channels {out_channels}')
        self.conv1 = Conv2dSN3x3(
            in_channels=in_channels, out_channels=out_channels, stride=stride, spectral_norm=spectral_norm, bias=bias
        )
        self.bn1 = add_norm(norm, out_channels)
        self.act1 = act()

        self.conv2 = Conv2dSN3x3(
            in_channels=out_channels, out_channels=out_channels, stride=1, spectral_norm=spectral_norm, bias=bias
        )
        self.bn2 = add_norm(norm, out_channels)
        if se:
            self.se = SEModule(out_channels)
            # self.se = eca_layer(out_channels)

        self.act2 = act()

        if stride == 2:
            self.downsample = nn.AvgPool2d(2)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                add_norm(norm, in_channels=out_channels),
            )

    def zero_init_last_bn(self):
        if isinstance(self.bn2, nn.BatchNorm2d):
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        if self.project is not None:
            residual = self.project(residual)

        x += residual
        x = self.act2(x)

        return x


class NewBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        act=nn.Mish,
        spectral_norm=True,
        norm=None,
        bias=False,
        downsample="upsample",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = Conv2dSN3x3(
            in_channels=in_channels, out_channels=out_channels, stride=stride, spectral_norm=spectral_norm, bias=bias
        )
        self.norm1 = add_norm(norm, out_channels)
        self.act1 = act()

        self.conv2 = Conv2dSN3x3(
            in_channels=out_channels, out_channels=out_channels, spectral_norm=spectral_norm, bias=bias
        )
        self.norm2 = add_norm(norm, out_channels)
        self.act2 = act()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        return out


class NestedInception(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        mid_channels=None,
        use_shortcut=False,
        act=nn.ReLU,
        norm=False,
        bias=True,
        stride=1,
        spectral_norm=False,
        downsample="upsample",
        kernel_size=3,
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        mid_channels = mid_channels if mid_channels is not None else in_channels // 4
        if use_shortcut:
            self.shortcut = (
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )
        else:
            self.shortcut = None

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )
        self.bn1 = nn.BatchNorm2d(mid_channels) if norm else nn.Identity()
        self.act1 = act() if act is not None else nn.Identity()

        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )
        self.bn2 = nn.BatchNorm2d(mid_channels) if norm else nn.Identity()
        self.act2 = act() if act is not None else nn.Identity()

        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )

        self.bn3 = nn.BatchNorm2d(3 * mid_channels) if norm else nn.Identity()
        self.act3 = act() if act is not None else nn.Identity()

        self.last_conv = nn.Conv2d(in_channels=3 * mid_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = 0

        out_10 = self.conv1(x)
        out_11 = self.act1(self.bn1(out_10))

        out_20 = self.conv2(out_11)
        out_21 = self.act2(self.bn2(out_20))

        out_30 = self.conv3(out_21)

        out = torch.cat([out_10, out_20, out_30], dim=1)
        out = self.act3(self.bn3(out))
        out = self.last_conv(out)

        return shortcut + out


class SimpleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        use_shortcut=False,
        act=nn.ReLU,
        norm=False,
        bias=True,
        stride=1,
        spectral_norm=False,
        downsample="upsample",
        kernel_size=3,
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        if use_shortcut:
            self.shortcut = (
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )
        else:
            self.shortcut = None

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding=kernel_size // 2,
            padding_mode="reflect",
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act1 = act() if act is not None else nn.Identity()

    def forward(self, x):
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = 0

        out = self.conv1(x)
        out = self.act1(self.bn1(out))

        return shortcut + out
