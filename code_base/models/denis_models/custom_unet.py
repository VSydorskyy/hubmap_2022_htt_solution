#!/usr/bin/env python
# coding: utf-8
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Mish, init

# UpsampleBilinearAdditiveUpsample,UpsampleBilinearAdditiveUpsample2x, UpsampleResidualDeconvolutionUpsample2x,BilinearUpsample2x
from .attention import (
    AttentionBlock,
    CBAMModule,
    ChannelLinearAttention,
    ECAModule,
    PositionLinearAttention,
    SEModule,
    TripletAttention,
)
from .unet_blocks import ConvNextBlock, NestedInception, PostActBlock, PreActBlock, PreActBlock_Enc
from .upsample import *


def _take(elements, indexes):
    return list([elements[i] for i in indexes])


class EncoderModule(nn.Module):
    def __init__(self, channels: List[int], strides: List[int], layers: List[int]):
        super().__init__()
        assert len(channels) == len(strides)

        self._layers = layers
        self._output_strides = _take(strides, layers)
        self._output_filters = _take(channels, layers)

    def forward(self, x):
        input = x
        output_features = []
        for layer in self.encoder_layers:
            output = layer(input)
            output_features.append(output)
            input = output
        # Return only features that were requested
        return _take(output_features, self._layers)

    @property
    def output_strides(self) -> List[int]:
        return self._output_strides

    @property
    def output_filters(self) -> List[int]:
        return self._output_filters

    @property
    def encoder_layers(self):
        raise NotImplementedError

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)


class UNetEncoder(EncoderModule):
    def __init__(
        self,
        start_channels,
        depth=4,
        encoder_channels=2.0,
        act=Mish,
        stride=2,
        spectral_norm=False,
        norm=None,
        bias=False,
        block=PreActBlock,
        n_blocks=2,
        stem=None,
        DO=0.25,
        se="SE",
    ):
        # current_stride=1
        # current_ch=start_channels

        if (type(encoder_channels) == int) or (type(encoder_channels) == float):
            channels = [int(start_channels * (2**i)) for i in range(depth)]

        elif type(encoder_channels) == list:
            channels = encoder_channels
        else:
            raise ValueError(f"Unsupported type for encoder_channels {type(encoder_channels)}. Should be int or list")

        strides = [2 * i for i in range(depth)]
        layers = range(depth)
        super().__init__(channels=channels, strides=strides, layers=layers)

        self.all_layers = nn.ModuleList()
        self.all_layers.append(stem)

        # current_stride=1
        # current_channels=out_channels=start_channels
        if type(n_blocks) == int:
            n_blocks = [n_blocks for i in range(depth - 1)]  # stem is the first layer
        assert len(n_blocks) == depth - 1

        for i in range(depth - 1):
            print(f"Encoder level {i} channels {channels[i]} our_channels {channels[i+1]}")
            self.all_layers.append(
                UNetEncoderBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    act=act,
                    pre_dropout_rate=0.0,
                    post_dropout_rate=DO,
                    spectral_norm=spectral_norm,
                    norm=norm,
                    bias=bias,
                    block=block,
                    n_blocks=n_blocks[i],
                    se=se,
                )
            )

    @property
    def encoder_layers(self):
        return self.all_layers


class UNetEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample=None,
        act=Mish,
        pre_dropout_rate=0.0,
        post_dropout_rate=0.0,
        spectral_norm=None,
        norm=None,
        bias=False,
        block=PreActBlock,
        n_blocks=2,
        se="SE",
    ):
        super().__init__()

        # self.downsample=torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        # print(f'EncoderBlock in_channels {in_channels} out_channels {out_channels}')
        self.blocks = [
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                act=act,
                spectral_norm=spectral_norm,
                norm=norm,
                bias=bias,
            )
        ]
        for i in range(n_blocks - 1):
            self.blocks.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    act=act,
                    spectral_norm=spectral_norm,
                    norm=norm,
                    bias=bias,
                )
            )
        self.blocks = nn.Sequential(*self.blocks)

        # self.pre_drop = nn.Dropout(pre_dropout_rate, inplace=True)
        self.post_drop = nn.Dropout(post_dropout_rate, inplace=True)
        if se is None:
            self.se = nn.Identity()
        else:
            if se == "SE":
                self.se = SEModule(in_channels)
            elif se == "ECA":
                self.se = ECAModule(in_channels)
            elif se == "CBAM":
                self.se = CBAMModule(in_channels)
            else:
                raise (f"Unsupported attention {se}")

    def forward(self, x):
        out = self.se(x)
        out = self.blocks(out)
        out = self.post_drop(out)
        return out


class DecoderModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        raise NotImplementedError

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)


class UNetDecoder(DecoderModule):
    def __init__(
        self,
        features,
        start_features: int,
        act=Mish,
        spectral_norm=False,
        norm=None,
        bias=False,
        block=PostActBlock,
        upsample=UpsampleNearestAdditiveUpsample2x,
        n_blocks=2,
        DO=0.0,
        se="SE",
        attnUnet=False,
        channel_attention=False,
        positional_attention=False,
    ):
        super().__init__()
        decoder_features = start_features
        reversed_features = list(reversed(features))

        # print('Reversed features ', reversed_features)

        output_filters = [decoder_features]

        if type(n_blocks) == int:
            n_blocks = [n_blocks for i in range(len(reversed_features))]
        assert len(n_blocks) == len(reversed_features)

        self.center = UNetCentralBlock(
            reversed_features[0],
            decoder_features,
            spectral_norm=spectral_norm,
            norm=norm,
            bias=bias,
            se=se,
            DO=DO,
            block=block,
            act=act,
        )

        blocks = []
        for block_index, encoder_features in enumerate(reversed_features):
            blocks.append(
                UNetDecoderBlock(
                    in_dec_channels=output_filters[-1],
                    in_enc_channels=encoder_features,
                    out_channels=decoder_features,
                    act=act,
                    spectral_norm=spectral_norm,
                    norm=norm,
                    bias=bias,
                    block=block,
                    se=se,
                    scale_factor=2 if block_index > 0 else 1,
                    n_blocks=n_blocks[len(reversed_features) - block_index - 1],
                    attnUnet=attnUnet,
                    channel_attention=channel_attention,
                    positional_attention=positional_attention,
                    DO=DO,
                    upsample=upsample,
                )
            )
            output_filters.append(decoder_features)
            decoder_features = decoder_features // 2

        self.blocks = nn.ModuleList(blocks)
        self.output_filters = output_filters

    def forward(self, features):
        reversed_features = list(reversed(features))
        decoder_outputs = [self.center(reversed_features[0])]

        for block_index, (decoder_block, encoder_output) in enumerate(zip(self.blocks, reversed_features)):
            decoder_outputs.append(decoder_block(decoder_outputs[-1], encoder_output))

        return decoder_outputs


class UNetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_dec_channels,
        in_enc_channels,
        out_channels,
        act=Mish,
        pre_dropout_rate=0.0,
        post_dropout_rate=0.0,
        spectral_norm=False,
        norm=None,
        bias=False,
        block=PreActBlock,
        scale_factor=1.0,
        n_blocks=3,
        se="SE",
        upsample=BilinearUpsample2x,
        DO=0.0,
        attnUnet=True,
        channel_attention=False,
        positional_attention=False,
    ):
        super().__init__()

        self.scale_factor = scale_factor
        if scale_factor == 2:
            self.upsample = upsample(in_channels=in_dec_channels)
            in_dec_channels = self.upsample.out_channels
        else:
            self.upsample = nn.Identity()

        if channel_attention:
            self.CLA = ChannelLinearAttention()
        else:
            self.CLA = nn.Identity()

        if positional_attention:
            self.PLA = PositionLinearAttention(in_enc_channels)
        else:
            self.PLA = nn.Identity()

        if attnUnet:
            self.attn = AttentionBlock(F_g=in_dec_channels, F_l=in_enc_channels, n_coefficients=in_dec_channels // 2)
        else:
            self.attn = lambda gate, skip_connection: skip_connection

        self.blocks = [
            block(
                in_channels=in_dec_channels + in_enc_channels,
                out_channels=out_channels,
                act=act,
                spectral_norm=spectral_norm,
                norm=norm,
                bias=bias,
            )
        ]
        for i in range(n_blocks - 1):
            self.blocks.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    act=act,
                    spectral_norm=spectral_norm,
                    norm=norm,
                    bias=bias,
                )
            )
        self.blocks = nn.Sequential(*self.blocks)

        if se is None:
            self.se = nn.Identity()
        else:
            if se == "SE":
                self.se = SEModule(in_dec_channels + in_enc_channels)
            elif se == "ECA":
                self.se = ECAModule(in_dec_channels + in_enc_channels)
            elif se == "CBAM":
                self.se = CBAMModule(in_dec_channels + in_enc_channels)
            elif se == "TRIPLET":
                self.se = TripletAttention()
            else:
                raise (f"Unsupported attention {se}")

        self.post_dropout = nn.Dropout2d(DO)

    def forward(self, x, enc):
        x = self.upsample(x)
        enc = self.PLA(enc)
        enc = self.CLA(enc)
        a = self.attn(gate=x, skip_connection=enc)
        out = torch.cat([x, a], 1)
        out = self.post_dropout(out)
        out = self.se(out)
        out = self.blocks(out)
        return out


class UNetCentralBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        act=nn.GELU,
        spectral_norm=False,
        norm=None,
        bias=False,
        block=NestedInception,
        n_blocks=1,
        se="SE",
        DO=0,
    ):
        super().__init__()

        self.blocks = [
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                act=act,
                spectral_norm=spectral_norm,
                norm=norm,
                bias=bias,
            )
        ]
        for i in range(n_blocks - 1):
            self.blocks.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    act=act,
                    spectral_norm=spectral_norm,
                    norm=norm,
                    bias=bias,
                )
            )
        self.blocks = nn.Sequential(*self.blocks)

        self.post_drop = nn.Dropout(DO, inplace=True)
        if se is None:
            self.se = nn.Identity()
        else:
            if se == "SE":
                self.se = SEModule(in_channels)
            elif se == "ECA":
                self.se = ECAModule(in_channels)
            elif se == "CBAM":
                self.se = CBAMModule(in_channels)
            elif se == "TRIPLET":
                self.se = TripletAttention()
            else:
                raise (f"Unsupported attention {se}")

    def forward(self, x):
        out = self.se(x)
        out = self.blocks(out)
        out = self.post_drop(out)
        return out


class UNetPostresize(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        out_channels=1,
        act=nn.SiLU,
        norm=nn.BatchNorm2d,
        bias=False,
        last_upsample=UpsampleBilinearAdditiveUpsample,
        multistage_upsample=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if encoder.strides[0] == 2:
            self.upsample = last_upsample(self.decoder.output_filters[-1])
        elif encoder.strides[0] == 4 and (not multistage_upsample):
            print("single stage")
            self.upsample = last_upsample(self.decoder.output_filters[-1], scale_factor=4)
        elif encoder.strides[0] == 4 and multistage_upsample:
            print("multi stage")
            self.upsample1 = last_upsample(self.decoder.output_filters[-1], scale_factor=2)
            self.conv = NestedInception(
                in_channels=self.upsample1.out_channels,
                mid_channels=self.upsample1.out_channels,
                out_channels=self.upsample1.out_channels,
                act=act,
                norm=norm,
                bias=bias,
                kernel_size=3,
            )
            self.upsample2 = last_upsample(self.upsample1.out_channels, scale_factor=2)
            self.upsample = nn.Sequential(self.upsample1, self.conv, self.upsample2)
            # last_upsample(self.decoder.output_filters[-1]//4, scale_factor=2))
            self.upsample.out_channels = (self.decoder.output_filters[-1] // 4) // 4
        else:
            self.upsample = nn.Identity()
            self.upsample.out_channels = self.decoder.output_filters[-1]

        self.last_conv = nn.Sequential(
            NestedInception(
                in_channels=self.upsample.out_channels,
                mid_channels=self.upsample.out_channels,
                out_channels=out_channels,
                act=act,
                norm=norm,
                bias=bias,
                kernel_size=3,
            ),
            # nn.Conv2d(self.upsample.out_channels, out_channels,1,bias=True))
        )
        self._init_weight(self.decoder)
        self._init_weight(self.last_conv)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def normalize_batch(self, x):
        with torch.no_grad():
            b = x.clone().to(torch.float32)
            self.mean = self.mean.to(b)
            self.std = self.std.to(b)
            return (b - self.mean) / self.std

    def forward(
        self,
        x,
        # Added in order to handle input with GC
        **kwargs,
    ):
        out = self.normalize_batch(x)
        features = self.encoder(out)
        features = self.decoder(features)

        out = self.upsample(features[-1])
        output = self.last_conv(out)
        return output

    def _init_weight(self, module):
        for n, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                # print(n,m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                pass
                # print(n,m)
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm2d") != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print("initialize network with %s" % init_type)
        self.decoder.apply(init_func)
        self.last_conv.apply(init_func)


from .timm_encoder import TimmEncoder
from .upsample import UpsampleBilinearAdditiveUpsample


def set_dropout(model, drop_rate=0.1, specific_name="drop2"):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            if specific_name is not None:
                if name == specific_name:
                    child.p = drop_rate
            else:
                child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def get_model(
    model_name="hrnet_w18_small_v2",
    in_channels=5,
    se=None,
    attn_unet=False,
    bias=False,
    channel_attention=False,
    positional_attention=False,
    norm=nn.BatchNorm2d,
    act=nn.SiLU,
    DO=0.0,
    spectral_norm=False,
    layers=[0, 1, 2, 3],
    upsample=UpsampleNearestAdditiveUpsample2x,
    last_upsample=UpsampleBilinearAdditiveUpsample,
    multistage_upsample=False,
    out_channels=3,
    n_blocks=1,
    block=NestedInception,
    pretrained=True,
    drop_rate=0.0,
    device="cpu",
):

    if type(model_name) == str:
        print("Using Timm Encoder")
        encoder = TimmEncoder(model_name, in_channels=in_channels, layers=layers, pretrained=pretrained)
    else:
        encoder = model_name
    decoder = UNetDecoder(
        features=encoder.channels,
        start_features=encoder.channels[-1],
        act=act,
        spectral_norm=spectral_norm,
        norm=norm,
        bias=bias,
        block=block,
        DO=DO,
        n_blocks=n_blocks,
        se=se,
        attnUnet=attn_unet,
        channel_attention=channel_attention,
        positional_attention=positional_attention,
        upsample=upsample,
    )

    model = UNetPostresize(
        encoder=encoder,
        decoder=decoder,
        out_channels=out_channels,
        last_upsample=last_upsample,
        multistage_upsample=multistage_upsample,
    )
    set_dropout(model.encoder, drop_rate=drop_rate)
    model.to(device)
    return model
