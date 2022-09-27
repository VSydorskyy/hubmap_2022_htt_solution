from .gc_tta import SegmentationTTAWrapperKwargs
from .monai_seg import MonaiWrapper
from .smp_chlast import FPNChLast, UnetChLast, UnetPlusPlusChLast
from .smp_seg import SMPWrapper
from .timm_clf import TimmWrapper, TimmWrapperV2
from .transformer_seg import TransformerWrapper

try:
    from .unet_gc import UnetGC
    from .unetx3 import UnetMultiHead
except:
    print("Custom SMP modifications were not imported. Newer version of SMP")
