#!/usr/bin/env python
# coding: utf-8


from math import hypot

import torch
import torch.nn as nn
from pytorch_toolbelt.modules import (
    BilinearAdditiveUpsample2d,
    DeconvolutionUpsample2d,
    ResidualDeconvolutionUpsample2d,
)
from torch import Tensor


def bilinear_upsample_initializer(x):
    cc = x.size(2) // 2
    cr = x.size(3) // 2

    for i in range(x.size(2)):
        for j in range(x.size(3)):
            x[..., i, j] = hypot(cc - i, cr - j)

    y = 1 - x / x.sum(dim=(2, 3), keepdim=True)
    y = y / y.sum(dim=(2, 3), keepdim=True)
    return y


def icnr_init(tensor: torch.Tensor, upscale_factor=2, initializer=nn.init.kaiming_normal):
    """Fill the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution"
    - Andrew Aitken et al. (2017), this inizialization should be used in the
    last convolutional layer before a PixelShuffle operation
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        upscale_factor: factor to increase spatial resolution by
        initializer: inizializer to be used for sub_kernel inizialization
    Examples:
        >>> upscale = 8
        >>> num_classes = 10
        >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
        >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
        >>> ps = PixelShuffle(upscale)
        >>> kernel = ICNR(conv_shuffle.weight, scale_factor=upscale)
        >>> conv_shuffle.weight.data.copy_(kernel)
        >>> output = ps(conv_shuffle(previous_layer_features))
        >>> print(output.shape)
        torch.Size([8, 10, 256, 256])
    .. _Checkerboard artifact free sub-pixel convolution:
        https://arxiv.org/abs/1707.02937
    """
    new_shape = [int(tensor.shape[0] / (upscale_factor**2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = initializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor**2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)
    return kernel


class DepthToSpaceUpsample2d(nn.Module):
    """
    NOTE: This block is not fully ready yet. Need to figure out how to correctly initialize
    default weights to have bilinear upsample identical to OpenCV results
    https://github.com/pytorch/pytorch/pull/5429
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        n = 2**scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

        with torch.no_grad():
            self.conv.weight.data = icnr_init(
                self.conv.weight, upscale_factor=scale_factor, initializer=bilinear_upsample_initializer
            )

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        x = self.shuffle(self.conv(x))
        return x


class UpsampleClass(nn.Module):
    def __init__(self, upsample, in_channels, channel_div=None, out_channels=None):
        super().__init__()
        self.upsample = upsample
        self.channel_div = channel_div
        self.in_channels = in_channels
        assert (channel_div is not None) or (out_channels is not None)
        self.out_ch = out_channels
        if self.out_ch is None:
            self.out_ch = self.in_channels // self.channel_div

    @property
    def out_channels(self):
        return self.out_ch

    def forward(self, x):
        return self.upsample(x)


class AdditiveUpsample2d(nn.Module):
    """
    https://arxiv.org/abs/1707.05847
    """

    def __init__(self, in_channels: int, scale_factor: int = 2, n: int = 4, bilinear=True):
        super().__init__()
        if in_channels % n != 0:
            raise ValueError(f"Number of input channels ({in_channels})must be divisable by n ({n})")

        self.in_channels = in_channels
        self.out_channels = in_channels // n
        if bilinear:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        else:
            self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor)
        self.n = n

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        x = self.upsample(x)
        n, c, h, w = x.size()
        x = x.reshape(n, c // self.n, self.n, h, w).mean(2)
        return x


class UpsampleBilinearAdditiveUpsample2x(UpsampleClass):
    def __init__(self, in_channels):
        upsample = AdditiveUpsample2d(in_channels=in_channels, scale_factor=2, n=4, bilinear=True)
        super().__init__(upsample=upsample, in_channels=in_channels, channel_div=4)


class UpsampleNearestAdditiveUpsample2x(UpsampleClass):
    def __init__(self, in_channels):
        upsample = AdditiveUpsample2d(in_channels=in_channels, scale_factor=2, n=4, bilinear=False)
        super().__init__(upsample=upsample, in_channels=in_channels, channel_div=4)


class UpsampleNearestAdditiveUpsample(UpsampleClass):
    def __init__(self, in_channels, scale_factor=2):
        upsample = AdditiveUpsample2d(
            in_channels=in_channels, scale_factor=scale_factor, n=2**scale_factor, bilinear=False
        )
        super().__init__(upsample=upsample, in_channels=in_channels, channel_div=2**scale_factor)


class UpsampleResidualDeconvolutionUpsample2x(UpsampleClass):
    def __init__(self, in_channels):
        upsample = ResidualDeconvolutionUpsample2d(in_channels=in_channels, scale_factor=2, n=4)
        super().__init__(upsample=upsample, in_channels=in_channels, channel_div=4)


class UpsampleDepthToSpaceUpsample2x(UpsampleClass):
    def __init__(self, in_channels):
        upsample = DepthToSpaceUpsample2d(in_channels=in_channels, scale_factor=2)
        super().__init__(upsample=upsample, in_channels=in_channels, channel_div=4)


class BilinearUpsample2x(UpsampleClass):
    def __init__(self, in_channels):
        upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        super().__init__(upsample=upsample, in_channels=in_channels, channel_div=1)


class UpsampleBilinearAdditiveUpsample(UpsampleClass):
    def __init__(self, in_channels, scale_factor=2):
        upsample = BilinearAdditiveUpsample2d(in_channels=in_channels, scale_factor=scale_factor, n=2**scale_factor)
        super().__init__(upsample=upsample, in_channels=in_channels, channel_div=2**scale_factor)


class UpsampleResidualDeconvolutionUpsample(UpsampleClass):
    def __init__(self, in_channels, scale_factor=2):
        upsample = ResidualDeconvolutionUpsample2d(
            in_channels=in_channels, scale_factor=scale_factor, n=2**scale_factor
        )
        super().__init__(upsample=upsample, in_channels=in_channels, channel_div=2**scale_factor)


class UpsampleDepthToSpaceUpsample(UpsampleClass):
    def __init__(self, in_channels, scale_factor=2):
        upsample = DepthToSpaceUpsample2d(in_channels=in_channels, scale_factor=scale_factor, n=2**scale_factor)
        super().__init__(upsample=upsample, in_channels=in_channels, channel_div=2**scale_factor)


class BilinearUpsample(UpsampleClass):
    def __init__(self, in_channels, scale_factor=2, out_channels=None):
        if out_channels is None:
            out_channels = in_channels // (scale_factor**2)

        upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )
        super().__init__(upsample=upsample, in_channels=in_channels, out_channels=out_channels)


class BilinearUpsample4x(UpsampleClass):
    def __init__(self, in_channels, out_channels=None, scale_factor=4):
        if out_channels is None:
            out_channels = in_channels // 16

        upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        super().__init__(upsample=upsample, in_channels=in_channels, out_channels=out_channels)


class BilinearUpsample2x(UpsampleClass):
    def __init__(self, in_channels, out_channels=None, scale_factor=4):
        if out_channels is None:
            out_channels = in_channels // 4

        upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        super().__init__(upsample=upsample, in_channels=in_channels, out_channels=out_channels)


class NearestUpsample2x(UpsampleClass):
    def __init__(self, in_channels, out_channels=None, scale_factor=4):
        if out_channels is None:
            out_channels = in_channels // 4

        upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )

        super().__init__(upsample=upsample, in_channels=in_channels, out_channels=out_channels)


from .pixel_shuffle_fastai import PixelShuffle_ICNR


class PixelShuffle4x(UpsampleClass):
    def __init__(self, in_channels, out_channels=None, scale_factor=4):
        if out_channels is None:
            out_channels = in_channels // 16

        upsample = PixelShuffle_ICNR(ni=in_channels, nf=out_channels, scale=scale_factor, blur=False, act_cls=nn.GELU)

        super().__init__(upsample=upsample, in_channels=in_channels, out_channels=out_channels)


assert UpsampleBilinearAdditiveUpsample2x(in_channels=16)(torch.zeros((1, 16, 256, 256))).size() == (1, 4, 512, 512)
assert UpsampleResidualDeconvolutionUpsample2x(in_channels=16)(torch.zeros((1, 16, 256, 256))).size() == (
    1,
    4,
    512,
    512,
)
assert UpsampleDepthToSpaceUpsample2x(in_channels=16)(torch.zeros((1, 16, 256, 256))).size() == (1, 4, 512, 512)
