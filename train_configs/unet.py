from glob import glob

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
from albumentations import HistogramMatching
from albumentations.pytorch import ToTensorV2
from catalyst import callbacks

from code_base.callbacks import DiceMetric, SWACallback
from code_base.constants import CLASSES
from code_base.datasets import HubMapDataset, HubMapPseudoDataset
from code_base.forwards import SMPHubMapForward, PointRandSMPHubMapForward
from code_base.models import SMPWrapper
from code_base.train_functions import catalyst_training
from code_base.utils.other import get_domain_adaptation_images
from code_base.transforms.crop import CachedCropNonEmptyMaskIfExists

B_S = 32
ROOT_PATH = "data/train_images/"
DEBUG = False
IMG_SIZE = None
DYNAMIC_SIZE_MODE = "scale_or"
USE_ONE_CHANNEL = True
TO_RGB = True
TGT_METRIC = "dice_score"
MAXIMIZE_METRIC = True
# AUG constants
p = 0.5
crop_size = 512

CONFIG = {
    "seed": 42,
    "df_path": "data/train.csv",
    "split_path": "data/cv_split5_v2.npy",
    "exp_name": "unet_pointrandX2loss_timmefficientnetb7NS_strongaugs_512_lr1e3_BceDiceFocalIoU_bs32_testscale_HMGTEXAndOldTrainP1_datax8_1ch_masksamplingP05_ConcatDataset_DenisPseudoScaledV3X16_denisx2scales_datav2_HpaPseudoV3FX1",
    "files_to_save": (glob("code_base/**/*.py") + [__file__] + ["scripts/main_train.py"]),
    "folds": [0, 1, 2, 3, 4],
    "train_function": catalyst_training,
    "train_function_args": {
        "train_dataset_class": HubMapPseudoDataset,
        "train_dataset_config": {
            "root": ROOT_PATH,
            "pseudo_df_path": "data/hpa_add/v3_full.csv",
            "pseudo_root": "data/hpa_add/v2",
            "debug": DEBUG,
            "img_size": IMG_SIZE,
            "dynamic_resize_mode": DYNAMIC_SIZE_MODE,
            "use_one_channel_mask": USE_ONE_CHANNEL,
            "to_rgb": TO_RGB,
            "additional_scalers": {
                'prostate': 0.15 * 2,
                'spleen': 1 * 2,
                'lung': 0.5 * 2,
                'kidney': 1 * 2,
                'largeintestine': 1 * 2
            },
            "do_cutmix": True,
            "cutmix_transform": lambda : [
                A.PadIfNeeded(
                    min_height=crop_size,
                    min_width=crop_size,
                    # pad_height_divisor=32,
                    # pad_width_divisor=32,
                    border_mode=4,
                    value=None,
                    mask_value=None,
                    always_apply=True,
                ),
                # Sample Non-Empty mask with prob 0.5
                # Otherwise empty OR Non-Empty mask will be sampled
                A.OneOrOther(first=A.CropNonEmptyMaskIfExists(512, 512), second=A.RandomCrop(512, 512), p=0.5),
            ],
            "cutmix_params": {"prob": 0.5, "alpha": 1.0},
            "transform": [
                # dihedral_aug
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                # brightness_aug
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1.0, contrast_limit=(-0.2, 0.2), brightness_limit=(-0.1, 0.1)),
                        A.HueSaturationValue(
                            p=1.0,
                            always_apply=False,
                            hue_shift_limit=(-30, 30),
                            sat_shift_limit=(-90, 20),
                            val_shift_limit=(-20, 20),
                        ),
                        A.RandomGamma(p=0.5, gamma_limit=(50, 200)),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.ChannelShuffle(p=0.1),
                        A.ColorJitter(p=0.1),
                        A.FancyPCA(p=0.1),
                        A.augmentations.transforms.CLAHE(p=0.1),
                    ],
                    p=0.2,
                ),
                # distortion_aug
                A.OneOf([A.OpticalDistortion(p=0.3), A.GridDistortion(p=0.3), A.ElasticTransform(p=0.1)], p=0.5),
                # noise_aug
                A.OneOf(
                    [
                        A.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
                        A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0)),
                        A.MultiplicativeNoise(
                            always_apply=False, p=1.0, multiplier=(0.9, 1.1), per_channel=True, elementwise=True
                        ),
                    ],
                    p=0.1,
                ),
                # resize_crop
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.3, rotate_limit=20, p=0.5),
                A.Normalize(),
                ToTensorV2(transpose_mask=True),
            ],
            "hist_matching_params_init": lambda : dict(
                p=1.0,
                reference_images=get_domain_adaptation_images(
                    [
                        "data/gtex/*",
                    ], 
                    rgb=TO_RGB
                ),
                read_fn=lambda x: x,
                blend_ratio=(0.5, 1.0),
            ),
            "dataset_repeat": 8,
        },
        "add_train_dataset_class": [HubMapPseudoDataset, HubMapDataset],
        "add_train_dataset_df_path": [
            None,
            "data/gtex/v3/train.csv",
        ],
        "add_train_dataset_config": [
            {
                "root": ROOT_PATH,
                "pseudo_df_path": "data/hpa_add/v3_full.csv",
                "pseudo_root": "data/hpa_add/v2",
                "debug": DEBUG,
                "img_size": None,
                "dynamic_resize_mode": None,
                "use_one_channel_mask": USE_ONE_CHANNEL,
                "to_rgb": TO_RGB,
                "additional_scalers": {
                    'prostate': 0.15 * 2,
                    'spleen': 1 * 2,
                    'lung': 0.5 * 2,
                    'kidney': 1 * 2,
                    'largeintestine': 1 * 2
                },
                "do_cutmix": True,
                "cutmix_transform": lambda: [
                    A.PadIfNeeded(
                        min_height=crop_size,
                        min_width=crop_size,
                        # pad_height_divisor=32,
                        # pad_width_divisor=32,
                        border_mode=4,
                        value=None,
                        mask_value=None,
                        always_apply=True,
                    ),
                    # Sample Non-Empty mask with prob 0.5
                    # Otherwise empty OR Non-Empty mask will be sampled
                    A.OneOrOther(first=A.CropNonEmptyMaskIfExists(512, 512), second=A.RandomCrop(512, 512), p=0.5),
                ],
                "cutmix_params": {"prob": 0.5, "alpha": 1.0},
                "transform": [
                    # dihedral_aug
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Transpose(p=0.5),
                    # brightness_aug
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=1.0, contrast_limit=(-0.2, 0.2), brightness_limit=(-0.1, 0.1)),
                            A.HueSaturationValue(
                                p=1.0,
                                always_apply=False,
                                hue_shift_limit=(-30, 30),
                                sat_shift_limit=(-90, 20),
                                val_shift_limit=(-20, 20),
                            ),
                            A.RandomGamma(p=0.5, gamma_limit=(50, 200)),
                        ],
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.ChannelShuffle(p=0.1),
                            A.ColorJitter(p=0.1),
                            A.FancyPCA(p=0.1),
                            A.augmentations.transforms.CLAHE(p=0.1),
                        ],
                        p=0.2,
                    ),
                    # distortion_aug
                    A.OneOf([A.OpticalDistortion(p=0.3), A.GridDistortion(p=0.3), A.ElasticTransform(p=0.1)], p=0.5),
                    # noise_aug
                    A.OneOf(
                        [
                            A.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
                            A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0)),
                            A.MultiplicativeNoise(
                                always_apply=False, p=1.0, multiplier=(0.9, 1.1), per_channel=True, elementwise=True
                            ),
                        ],
                        p=0.1,
                    ),
                    # resize_crop
                    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.3, rotate_limit=20, p=0.5),
                    A.Normalize(),
                    ToTensorV2(transpose_mask=True),
                ],
                "dataset_repeat": 4,
            },
            {
                "root": "data/gtex/v3/images",
                "debug": DEBUG,
                "img_size": None,
                "dynamic_resize_mode": None,
                "use_one_channel_mask": USE_ONE_CHANNEL,
                "to_rgb": TO_RGB,
                "imread_backend": "tifi",
                # Scales taken from gtxportal
                "additional_scalers": {
                    'prostate': 12.666936462970458 * 0.15 * 2,
                    'spleen': 1.0006070416835289 * 1 * 2,
                    'lung': 1.530149736948604 * 0.5 * 2,
                    'kidney': 1.011736139214893 * 1 * 2,
                    'largeintestine': 0.46337515176042093 * 1 * 2
                },
                "do_cutmix": True,
                "cutmix_transform": lambda : [
                    A.PadIfNeeded(
                        min_height=crop_size,
                        min_width=crop_size,
                        # pad_height_divisor=32,
                        # pad_width_divisor=32,
                        border_mode=4,
                        value=None,
                        mask_value=None,
                        always_apply=True,
                    ),
                    # Sample Non-Empty mask with prob 0.5
                    # Otherwise empty OR Non-Empty mask will be sampled
                    A.OneOrOther(first=A.CropNonEmptyMaskIfExists(512, 512), second=A.RandomCrop(512, 512), p=0.5),
                ],
                "cutmix_params": {"prob": 0.5, "alpha": 1.0},
                "transform": [
                    # dihedral_aug
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Transpose(p=0.5),
                    # brightness_aug
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=1.0, contrast_limit=(-0.2, 0.2), brightness_limit=(-0.1, 0.1)),
                            A.HueSaturationValue(
                                p=1.0,
                                always_apply=False,
                                hue_shift_limit=(-30, 30),
                                sat_shift_limit=(-90, 20),
                                val_shift_limit=(-20, 20),
                            ),
                            A.RandomGamma(p=0.5, gamma_limit=(50, 200)),
                        ],
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.ChannelShuffle(p=0.1),
                            A.ColorJitter(p=0.1),
                            A.FancyPCA(p=0.1),
                            A.augmentations.transforms.CLAHE(p=0.1),
                        ],
                        p=0.2,
                    ),
                    # distortion_aug
                    A.OneOf([A.OpticalDistortion(p=0.3), A.GridDistortion(p=0.3), A.ElasticTransform(p=0.1)], p=0.5),
                    # noise_aug
                    A.OneOf(
                        [
                            A.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
                            A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0)),
                            A.MultiplicativeNoise(
                                always_apply=False, p=1.0, multiplier=(0.9, 1.1), per_channel=True, elementwise=True
                            ),
                        ],
                        p=0.1,
                    ),
                    # resize_crop
                    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.3, rotate_limit=20, p=0.5),
                    A.Normalize(),
                    ToTensorV2(transpose_mask=True),
                ],
                "dataset_repeat": 16,
            },
        ],
        "val_dataset_class": HubMapDataset,
        "val_dataset_config": {
            "root": ROOT_PATH,
            "debug": DEBUG,
            "img_size": IMG_SIZE,
            "dynamic_resize_mode": DYNAMIC_SIZE_MODE,
            "use_one_channel_mask": USE_ONE_CHANNEL,
            "to_rgb": TO_RGB,
            "additional_scalers": {
                'prostate': 0.15 * 2,
                'spleen': 1 * 2,
                'lung': 0.5 * 2,
                'kidney': 1 * 2,
                'largeintestine': 1 * 2
            },
            "transform": [
                A.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_height_divisor=32,
                    pad_width_divisor=32,
                    border_mode=4,
                    value=None,
                    mask_value=None,
                    always_apply=True,
                ),
                A.Normalize(),
                ToTensorV2(transpose_mask=True),
            ],
        },
        "train_dataloader_config": {
            "batch_size": B_S,
            "shuffle": True,
            "drop_last": True,
            "num_workers": 8,
            "pin_memory": True,
        },
        "val_dataloader_config": {
            "batch_size": 1,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 8,
            "pin_memory": True,
        },
        "nn_model_class": SMPWrapper,
        "nn_model_config": {
            "backbone_name": "timm-efficientnet-b7",
            "num_classes": 1,  # len(CLASSES),
            "arch_name": "Unet",
            "pretrained": "noisy-student",
            "point_rand_config": {"in_ch": 161, "num_classes": 1, "backbone_type": "effnet"},
        },
        "optimizer_init": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
        "scheduler_init": lambda optimizer, len_train: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.5,
            min_lr=1e-7,
            mode="max" if MAXIMIZE_METRIC else "min",
        ),
        # "scheduler_init": lambda optimizer, len_train: torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=len_train*5 + 50,
        #     eta_min=1e-6,
        # ),
        "forward": lambda: PointRandSMPHubMapForward(
            {
                "softbce": smp.losses.SoftBCEWithLogitsLoss(),
                # Tversky here is DiceLoss :)
                "tversky": smp.losses.TverskyLoss(mode="multilabel", log_loss=False),
                "focal": smp.losses.FocalLoss(mode="multilabel"),
                "jaccard": smp.losses.JaccardLoss(mode="multilabel"),
            },
            point_rand_loss_coef=2.0,
        ),
        "n_epochs": 200,
        "catalyst_callbacks": lambda: [
            callbacks.BackwardCallback(metric_key="loss"),
            callbacks.OptimizerCallback(
                metric_key="loss",
                accumulation_steps=1,
            ),
            DiceMetric(),
            SWACallback(
                num_of_swa_models=3,
                maximize=True,
                loader_key="valid",
                logging_metric="dice_score",
                verbose=True,
            ),
            callbacks.EarlyStoppingCallback(
                patience=11,
                loader_key="valid",
                metric_key=TGT_METRIC,
                minimize=not MAXIMIZE_METRIC,
            ),
            callbacks.SchedulerCallback(loader_key="valid", metric_key=TGT_METRIC, mode="epoch"),
        ],
        "main_metric": TGT_METRIC,
        "minimize_metric": not MAXIMIZE_METRIC,
        "fp16": True,
    },
}
