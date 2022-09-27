import inspect
from enum import Enum

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, normal_, uniform_, xavier_uniform_
from torch.nn.utils import spectral_norm, weight_norm

NormType = Enum("NormType", "Batch BatchZero Weight Spectral Instance InstanceZero")


def delegates(to=None, keep=False, but=None):
    "Decorator: replace `**kwargs` in signature with params from `to`"
    if but is None:
        but = []

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to.__init__ if isinstance(to, type) else to, f
        from_f = getattr(from_f, "__func__", from_f)
        to_f = getattr(to_f, "__func__", to_f)
        if hasattr(from_f, "__delwrap__"):
            return f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd and k not in but
        }
        sigd.update(s2)
        if keep:
            sigd["kwargs"] = k
        else:
            from_f.__delwrap__ = to_f
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def noop(x=None, *args, **kwargs):
    "Do nothing"
    return x


def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0.0 if zero else 1.0)
    return bn


@delegates(nn.BatchNorm2d)
def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm("BatchNorm", nf, ndim, zero=norm_type == NormType.BatchZero, **kwargs)


# Cell
@delegates(nn.InstanceNorm2d)
def InstanceNorm(nf, ndim=2, norm_type=NormType.Instance, affine=True, **kwargs):
    "InstanceNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm("InstanceNorm", nf, ndim, zero=norm_type == NormType.InstanceZero, affine=affine, **kwargs)


def nested_attr(o, attr, default=None):
    "Same as `getattr`, but if `attr` includes a `.`, then looks inside nested objects"
    try:
        for a in attr.split("."):
            o = getattr(o, a)
    except AttributeError:
        return default
    return o


# Cell
def nested_callable(o, attr):
    "Same as `nested_attr` but if not found will return `noop`"
    return nested_attr(o, attr, noop)


def getcallable(o, attr):
    "Calls `getattr` with a default of `noop`"
    return getattr(o, attr, noop)


def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a


def init_linear(m, act_func=None, init="auto", bias_std=0.01):
    if getattr(m, "bias", None) is not None and bias_std is not None:
        if bias_std != 0:
            normal_(m.bias, 0, bias_std)
        else:
            m.bias.data.zero_()
    if init == "auto":
        if act_func in (F.relu_, F.leaky_relu_):
            init = kaiming_uniform_
        else:
            init = nested_callable(act_func, "__class__.__default_init__")
        if init == noop:
            init = getcallable(act_func, "__default_init__")
    if callable(init):
        init(m.weight)


# Cell
def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <= 3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')


class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."

    @delegates(nn.Conv2d)
    def __init__(
        self,
        ni,
        nf,
        ks=3,
        stride=1,
        padding=None,
        bias=None,
        ndim=2,
        norm_type=NormType.Batch,
        bn_1st=True,
        act_cls=nn.ReLU,
        transpose=False,
        init="auto",
        xtra=None,
        bias_std=0.01,
        **kwargs,
    ):
        if padding is None:
            padding = (ks - 1) // 2 if not transpose else 0
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None:
            bias = not (bn or inn)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        init_linear(conv, act, init=init, bias_std=bias_std)
        if norm_type == NormType.Weight:
            conv = weight_norm(conv)
        elif norm_type == NormType.Spectral:
            conv = spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act is not None:
            act_bn.append(act)
        if bn:
            act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn:
            act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st:
            act_bn.reverse()
        layers += act_bn
        if xtra:
            layers.append(xtra)
        super().__init__(*layers)


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


# Cell
class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."

    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type=NormType.Weight, act_cls=nn.ReLU):
        super().__init__()
        nf = ifnone(nf, ni)
        layers = [
            ConvLayer(ni, nf * (scale**2), ks=1, norm_type=norm_type, act_cls=act_cls, bias_std=0),
            nn.PixelShuffle(scale),
        ]
        if norm_type == NormType.Weight:
            layers[0][0].weight_v.data.copy_(icnr_init(layers[0][0].weight_v.data))
            layers[0][0].weight_g.data.copy_(
                ((layers[0][0].weight_v.data ** 2).sum(dim=[1, 2, 3]) ** 0.5)[:, None, None, None]
            )
        else:
            layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))
        if blur:
            layers += [nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)
