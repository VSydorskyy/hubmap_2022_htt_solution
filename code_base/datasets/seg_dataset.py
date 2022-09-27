import math
import os
import random
from copy import deepcopy

import albumentations.augmentations.crops.functional as albu_f
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from albumentations import Compose, HistogramMatching, Normalize
from albumentations.pytorch import ToTensorV2
from joblib import delayed
from tqdm import tqdm

try:
    import torchstain
except:
    print("torchstain was not imported")

from ..constants import CLASSES, PIXEL_SCALE, PIXEL_SIZE
from ..transforms import HistogramMatchingPerClass, Mosaic
from ..utils.mask import (
    get_centroid,
    get_contour_mask,
    get_igor_masks_overlap,
    get_mask_instances,
    rle_decode,
    rle_decode_multichannel_mask,
)
from ..utils.other import imread_rgb
from ..utils.parallel import ProgressParallel
from .cutmix import rand_bbox


class HubMapDataset(data.Dataset):
    """
    Item selection logic:
    1. Load image+mask+meta from cache or from disk THEN perform cutmix_transform (if needed)
    2. Take image+mask+meta from step 1 and perform CutMix (if needed) with another image+mask from step 1
    THEN perform transform (if needed)
    3. Take image+mask+meta from step 2 and perform Mosaic (if needed) with other 3 images+masks from step 2
    THEN perform last_transform (if needed)

    Order of AUX masks:
    1. Contour (if defined)
    2. Igor Overlap (if defined)
    3. Centroids (if defined)
    """

    def __init__(
        self,
        df,
        root,
        name_col="id",
        organ_col="organ",
        height_col="img_height",
        width_col="img_width",
        cls_weight_col="sampler_col",
        tgt_col="rle",
        transform=None,
        last_transform=None,
        precompute=True,
        test_mode=False,
        n_cores=64,
        img_size=None,
        dynamic_resize_mode=None,
        additional_scalers=None,
        debug=False,
        do_cutmix=False,
        cutmix_params={"prob": 0.5, "alpha": 1.0},
        do_mosaic=False,
        mosaic_params={"height": 512, "width": 512, "p": 0.5},
        cutmix_transform=None,
        ext=".tiff",
        dataset_repeat=1,
        hist_matching_params_init=None,
        scale_down=True,
        use_one_channel_mask=False,
        to_rgb=False,
        imread_backend="cv2",
        crop_size=None,
        notempty_crop_prob=0.5,
        image_interpolation=cv2.INTER_NEAREST,
        contour_config=None,
        igor_overlap_config=None,
        centroid_config=None,
        organs_include=None,
        apply_torchstain_norm=False,
    ):
        if dynamic_resize_mode is not None:
            assert dynamic_resize_mode in [
                "scale_32",
                "noscale_32",
                "scale_or",
            ]
        if contour_config is not None and not use_one_channel_mask:
            raise ValueError("contour config is supported only with one channel")
        if igor_overlap_config is not None and not use_one_channel_mask:
            raise ValueError("igor_overlap config is supported only with one channel")
        if centroid_config is not None and not use_one_channel_mask:
            raise ValueError("centroid config is supported only with one channel")
        if dynamic_resize_mode is not None and img_size is not None:
            raise ValueError("img size is dynamic for dynamic resize")
        if do_cutmix and test_mode:
            raise ValueError("Cutmix can be used only in training")
        if do_mosaic and test_mode:
            raise ValueError("Mosaic can be used only in training")
        if crop_size is not None:
            if not precompute:
                raise ValueError("Internal dataset crop supported only with precompute")
            if cutmix_transform is not None:
                raise ValueError("Internal dataset crop is not supported only with defined cutmix_transform")
        self.root = root
        if debug:
            self.df = df.iloc[:32].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
        if organs_include is not None:
            print(f"Excluding organs : {set(self.df[organ_col]) - set(organs_include)}")
            self.df = self.df[self.df[organ_col].isin(organs_include)].reset_index(drop=True)
            print(f"Remain df rows : {len(self.df)}")
        self.image_size = img_size
        self.dynamic_resize_mode = dynamic_resize_mode
        self.additional_scalers = additional_scalers
        self.scale_down = scale_down
        self.organ_col = organ_col
        self.tgt_col = tgt_col
        self.width_col = width_col
        self.height_col = height_col
        # Not pretty straightforward but ok
        self.id_col = name_col
        self.apply_torchstain_norm = apply_torchstain_norm
        if self.apply_torchstain_norm:
            self.torchstain_normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
        # Dirty hack, because
        # NotImplementedError: HistogramMatching can not be serialized.
        if hist_matching_params_init is not None:
            hist_matching_params = hist_matching_params_init()
            hist_matching_idx = hist_matching_params.pop("idx", 0)
            hist_matching_class = (
                HistogramMatchingPerClass if hist_matching_params.pop("per_class", False) else HistogramMatching
            )
            if hist_matching_idx == 0:
                transform = [hist_matching_class(**hist_matching_params)] + transform
            else:
                transform = (
                    transform[:hist_matching_idx]
                    + [hist_matching_class(**hist_matching_params)]
                    + transform[hist_matching_idx:]
                )
            print(f"Dataset Transforms: \n{transform}")
        self.transform = Compose(transform) if transform is not None else transform
        self.last_transform = Compose(last_transform) if last_transform is not None else last_transform
        self.precompute = precompute
        self.test_mode = test_mode
        # We have to initialize it each time separately because it may use CachedCropNonEmptyMaskIfExists
        # In such a case re-init == empty cache
        self.cutmix_transform = Compose(cutmix_transform()) if cutmix_transform is not None else cutmix_transform
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        self.do_mosaic = do_mosaic
        if self.do_mosaic:
            self.mosaic = Mosaic(**mosaic_params)
        self.ext = ext
        self.dataset_repeat = dataset_repeat
        self.use_one_channel_mask = use_one_channel_mask
        self.imread = imread_rgb if to_rgb else cv2.imread
        self.imread_backend = imread_backend
        self.crop_size = crop_size
        self.notempty_crop_prob = notempty_crop_prob
        self.image_interpolation = image_interpolation
        self.contour_config = contour_config
        self.igor_overlap_config = igor_overlap_config
        self.centroid_config = centroid_config

        self.df[f"{name_col}_with_root"] = self.df[name_col].apply(lambda x: os.path.join(root, str(x)) + ext)
        self.name_col = f"{name_col}_with_root"

        if self.precompute:
            if n_cores is not None:
                print("Reading images ...")
                self.image_cache = ProgressParallel(n_jobs=n_cores, total=len(self.df))(
                    delayed(self.imread)(im_path, backend=self.imread_backend) for im_path in self.df[self.name_col]
                )
                print("Images in RAM")
                if (
                    self.image_size is not None
                    or dynamic_resize_mode is not None
                    or self.additional_scalers is not None
                ):
                    print("Resizing images ...")
                    if self.image_size is not None:
                        self.image_cache = ProgressParallel(n_jobs=n_cores, total=len(self.image_cache))(
                            delayed(cv2.resize)(img, self.image_size, interpolation=cv2.INTER_AREA)
                            for img in self.image_cache
                        )
                    # Dynamic resize
                    else:
                        self.image_cache = ProgressParallel(n_jobs=n_cores, total=len(self.image_cache))(
                            delayed(cv2.resize)(
                                img, self._get_dynamic_size(h=h, w=w, organ=organ), interpolation=cv2.INTER_AREA
                            )
                            for img, organ, w, h in zip(
                                self.image_cache,
                                self.df[self.organ_col],
                                self.df[self.width_col],
                                self.df[self.height_col],
                            )
                        )
                    print("Image resized")
                if self.apply_torchstain_norm:
                    print("Stain normalizing images ...")
                    self.image_cache = ProgressParallel(n_jobs=n_cores, total=len(self.image_cache))(
                        delayed(self.torchstain_normalizer.normalize)(I=img, stains=False) for img in self.image_cache
                    )
                    # Take only normed image
                    self.image_cache = [el[0] for el in self.image_cache]
                    print("Stain normalized images")
                self.image_cache = {i: el for i, el in enumerate(self.image_cache)}
                if not self.test_mode:
                    print("Reading masks ...")
                    self.mask_cache = ProgressParallel(n_jobs=n_cores, total=len(self.df))(
                        delayed(rle_decode_multichannel_mask)(
                            class_name=organ, mask_rle=rle, shape=(w, h), color=1, one_channel=use_one_channel_mask
                        )
                        for rle, organ, w, h in zip(
                            self.df[self.tgt_col],
                            self.df[self.organ_col],
                            self.df[self.width_col],
                            self.df[self.height_col],
                        )
                    )
                    print("Masks in RAM")
                    if (
                        self.image_size is not None
                        or dynamic_resize_mode is not None
                        or self.additional_scalers is not None
                    ):
                        print("Resizing masks ...")
                        if self.image_size is not None:
                            self.mask_cache = ProgressParallel(n_jobs=n_cores, total=len(self.mask_cache))(
                                delayed(cv2.resize)(mask, self.image_size, interpolation=self.image_interpolation)
                                for mask in self.mask_cache
                            )
                        # Dynamic resize
                        else:
                            self.mask_cache = ProgressParallel(n_jobs=n_cores, total=len(self.mask_cache))(
                                delayed(cv2.resize)(
                                    mask,
                                    self._get_dynamic_size(h=h, w=w, organ=organ),
                                    interpolation=self.image_interpolation,
                                )
                                for mask, organ, w, h in zip(
                                    self.mask_cache,
                                    self.df[self.organ_col],
                                    self.df[self.width_col],
                                    self.df[self.height_col],
                                )
                            )
                        print("Masks resized")
                    self.mask_cache = {i: el for i, el in enumerate(self.mask_cache)}
                    if self.crop_size is not None:
                        print("Precomputing crop stats ...")
                        self.crop_stats = {
                            # Taken from albumentations.CropNonEmptyMaskIfExists
                            k: self._crop_stat(v)
                            for k, v in tqdm(self.mask_cache.items())
                        }
                        print("Crop stats precomputed")
            else:
                raise NotImplementedError()

        if cls_weight_col is not None and cls_weight_col in self.df.columns:
            print("Sampler col - initialized")
            self.sample_weights = self.df[cls_weight_col]
        else:
            print("Sampler col is not initialized!")
            self.sample_weights = None

    def _get_dynamic_size(self, h, w, organ):
        if self.dynamic_resize_mode is not None:
            do_scale, do_32 = self.dynamic_resize_mode.split("_")
            do_scale = do_scale == "scale"
            do_32 = do_32 == "32"
        else:
            do_scale, do_32 = False, False
        if do_scale:
            if self.scale_down:
                h = int(h / PIXEL_SCALE[organ])
                w = int(w / PIXEL_SCALE[organ])
            else:
                h = int(h * PIXEL_SCALE[organ])
                w = int(w * PIXEL_SCALE[organ])
        if self.additional_scalers is not None:
            if self.scale_down:
                h = int(h / self.additional_scalers[organ])
                w = int(w / self.additional_scalers[organ])
            else:
                h = int(h * self.additional_scalers[organ])
                w = int(w * self.additional_scalers[organ])
        if do_32:
            h = int(math.ceil(h / 32) * 32)
            w = int(math.ceil(w / 32) * 32)
        return (w, h)

    @staticmethod
    def _crop_stat(mask):
        if (mask > 0).any():
            if mask.ndim == 3:
                return np.argwhere(mask.sum(axis=-1))
            else:
                return np.argwhere(mask)
        else:
            return None

    def _crop_image_mask(self, index, image, mask):
        if self.crop_stats[index] is not None and np.random.binomial(n=1, p=self.notempty_crop_prob):
            non_zero_yx = self.crop_stats[index]
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, self.crop_size[1] - 1)
            y_min = y - random.randint(0, self.crop_size[0] - 1)
            x_min = np.clip(x_min, 0, mask.shape[1] - self.crop_size[1])
            y_min = np.clip(y_min, 0, mask.shape[0] - self.crop_size[0])
            image = albu_f.crop(
                img=image, x_min=x_min, y_min=y_min, x_max=x_min + self.crop_size[1], y_max=y_min + self.crop_size[0]
            )
            mask = albu_f.crop(
                img=mask, x_min=x_min, y_min=y_min, x_max=x_min + self.crop_size[1], y_max=y_min + self.crop_size[0]
            )
        else:
            h_start, w_start = random.random(), random.random()
            image = albu_f.random_crop(
                img=image, crop_height=self.crop_size[0], crop_width=self.crop_size[1], h_start=h_start, w_start=w_start
            )
            mask = albu_f.random_crop(
                img=mask, crop_height=self.crop_size[0], crop_width=self.crop_size[1], h_start=h_start, w_start=w_start
            )
        return image, mask

    def _prepare_img_and_friends_from_idx(self, index: int):
        index = index % len(self.df)
        organ = self.df[self.organ_col].iloc[index]
        h = self.df[self.height_col].iloc[index]
        w = self.df[self.width_col].iloc[index]
        if self.dynamic_resize_mode is not None or self.additional_scalers is not None:
            rescaled_w, rescaled_h = self._get_dynamic_size(h=h, w=w, organ=organ)
        else:
            rescaled_w, rescaled_h = -1, -1
        im_id = self.df[self.id_col].iloc[index]

        if self.precompute:
            if self.test_mode:
                mask = -1
            else:
                mask = self.mask_cache[index]
            image = self.image_cache[index]
        else:
            if self.test_mode:
                mask = -1
            else:
                mask = rle_decode_multichannel_mask(
                    class_name=self.df[self.organ_col].iloc[index],
                    mask_rle=self.df[self.tgt_col].iloc[index],
                    shape=(w, h),
                    color=1,
                    one_channel=self.use_one_channel_mask,
                )
                if self.image_size is not None:
                    mask = cv2.resize(mask, self.image_size, interpolation=self.image_interpolation)
                if self.dynamic_resize_mode is not None or self.additional_scalers is not None:
                    mask = cv2.resize(mask, (rescaled_w, rescaled_h), interpolation=self.image_interpolation)
            image = self.imread(self.df[self.name_col].iloc[index], backend=self.imread_backend)
            if self.image_size is not None:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
            if self.dynamic_resize_mode is not None or self.additional_scalers is not None:
                image = cv2.resize(image, (rescaled_w, rescaled_h), interpolation=cv2.INTER_AREA)
            if self.apply_torchstain_norm:
                image = self.torchstain_normalizer.normalize(I=image, stains=False)[0]

        # Handle one channel case
        if not self.test_mode and mask.ndim < 3:
            mask = mask[:, :, None]

        if self.cutmix_transform is not None:
            result = self.cutmix_transform(image=image, mask=mask, index=index)
            image, mask = result["image"], result["mask"]

        if self.crop_size is not None:
            image, mask = self._crop_image_mask(index=index, image=image, mask=mask)

        if self.dynamic_resize_mode is not None and self.dynamic_resize_mode.split("_")[0] == "scale":
            pixel_size = PIXEL_SIZE[organ]
        else:
            pixel_size = PIXEL_SIZE["unscaled"]
        if self.additional_scalers is not None:
            if self.scale_down:
                pixel_size /= self.additional_scalers[organ]
            else:
                pixel_size *= self.additional_scalers[organ]
        pixel_size = torch.scalar_tensor(pixel_size)

        return image, mask, organ, h, w, rescaled_h, rescaled_w, im_id, pixel_size

    def _prepare_img_and_friends_from_idx_lvl2(self, index: int):
        image, mask, organ, h, w, rescaled_h, rescaled_w, im_id, pixel_size = self._prepare_img_and_friends_from_idx(
            index
        )

        # Cutmix
        if self.do_cutmix and np.random.binomial(n=1, p=self.cutmix_params["prob"]):
            cutmix_ix = np.random.choice(np.where(self.df[self.organ_col] == organ)[0])
            (
                cutmix_image,
                cutmix_mask,
                cutmix_organ,
                *_,
            ) = self._prepare_img_and_friends_from_idx(cutmix_ix)
            assert cutmix_organ == organ
            lam_cutmix = np.random.beta(self.cutmix_params["alpha"], self.cutmix_params["alpha"])
            bbx1, bby1, bbx2, bby2, lam_cutmix = rand_bbox(image.shape, lam_cutmix)
            mixed_image, mixed_mask = image.copy(), mask.copy()
            mixed_image[bbx1:bbx2, bby1:bby2] = cutmix_image[bbx1:bbx2, bby1:bby2]
            mixed_mask[bbx1:bbx2, bby1:bby2] = cutmix_mask[bbx1:bbx2, bby1:bby2]
            mask = np.clip(mask, 0.0, 1.0)
        else:
            mixed_image = image
            mixed_mask = mask

        if self.test_mode:
            if self.transform is not None:
                mixed_image = self.transform(image=mixed_image, class_name=organ)["image"]
        else:
            if self.transform is not None:
                result = self.transform(image=mixed_image, mask=mixed_mask, class_name=organ)
                mixed_image, mixed_mask = result["image"], result["mask"]

        return mixed_image, mixed_mask, organ, h, w, rescaled_h, rescaled_w, im_id, pixel_size

    def __getitem__(self, index):

        (
            image,
            mask,
            organ,
            h,
            w,
            rescaled_h,
            rescaled_w,
            im_id,
            pixel_size,
        ) = self._prepare_img_and_friends_from_idx_lvl2(index)

        if self.do_mosaic:
            image_cache, mask_cache = [], []
            for _ in range(3):
                mosaic_ix = np.random.choice(np.where(self.df[self.organ_col] == organ)[0])
                (
                    mosaic_image,
                    mosaic_mask,
                    mosaic_organ,
                    *_,
                ) = self._prepare_img_and_friends_from_idx_lvl2(mosaic_ix)
                assert mosaic_organ == organ
                image_cache.append(mosaic_image)
                mask_cache.append(mosaic_mask)
            result = self.mosaic(image=image, mask=mask, image_cache=image_cache, mask_cache=mask_cache)
            image, mask = result["image"], result["mask"]

        if self.test_mode:
            if self.last_transform is not None:
                image = self.last_transform(image=image)["image"].float()
        else:
            if self.last_transform is not None:
                result = self.last_transform(image=image, mask=mask)
                image, mask = result["image"].float(), result["mask"].float()

        is_numpy = isinstance(mask, np.ndarray)
        use_add_masks = (
            self.contour_config is not None or self.igor_overlap_config is not None or self.centroid_config is not None
        )

        if use_add_masks:
            add_masks = []

        if not self.test_mode and self.contour_config is not None:
            contour_mask = get_contour_mask(
                numpy_float_mask=mask[:, :, 0] if is_numpy else mask[0].numpy(), **self.contour_config
            )
            if is_numpy:
                add_masks.append(contour_mask[:, :, None])
            else:
                add_masks.append(torch.from_numpy(contour_mask).unsqueeze(dim=0))

        # Mask instances are required for both centroids and igor_masks_overlap
        if not self.test_mode and (self.igor_overlap_config is not None or self.centroid_config is not None):
            mask_instances = get_mask_instances(numpy_float_mask=mask[:, :, 0] if is_numpy else mask[0].numpy())

        if not self.test_mode and self.igor_overlap_config is not None:
            if mask_instances is not None:
                igor_mask_overlap = get_igor_masks_overlap(masks=mask_instances, **self.igor_overlap_config)
            else:
                igor_mask_overlap = np.zeros(mask[0].shape, np.float32)
            if is_numpy:
                add_masks.append(igor_mask_overlap[:, :, None])
            else:
                add_masks.append(torch.from_numpy(igor_mask_overlap).unsqueeze(dim=0))

        if not self.test_mode and self.centroid_config is not None:
            if mask_instances is not None:
                centroid = get_centroid(masks=mask_instances, **self.centroid_config)
            else:
                centroid = np.zeros(mask[0].shape, np.float32)
            if is_numpy:
                add_masks.append(centroid[:, :, None])
            else:
                add_masks.append(torch.from_numpy(centroid).unsqueeze(dim=0))

        if use_add_masks:
            if is_numpy:
                mask = np.concatenate([mask] + add_masks, axis=-1)
            else:
                mask = torch.cat([mask] + add_masks, dim=0)

        if self.test_mode:
            return image, mask, organ, h, w, rescaled_h, rescaled_w, im_id, pixel_size
        else:
            return image, mask, organ, pixel_size

    def __len__(self):
        return int(len(self.df) * self.dataset_repeat)


class HubMapPseudoDataset(HubMapDataset):
    def __init__(
        self,
        df,
        root,
        pseudo_df_path,
        pseudo_root,
        pseudo_rate=1.0,
        pseudo_ext="",
        name_col="id",
        organ_col="organ",
        height_col="img_height",
        width_col="img_width",
        cls_weight_col="sampler_col",
        tgt_col="rle",
        transform=None,
        last_transform=None,
        precompute=True,
        test_mode=False,
        n_cores=64,
        img_size=None,
        dynamic_resize_mode=None,
        additional_scalers=None,
        debug=False,
        do_cutmix=False,
        cutmix_params={"prob": 0.5, "alpha": 1.0},
        do_mosaic=False,
        mosaic_params={"height": 512, "width": 512, "p": 0.5},
        cutmix_transform=None,
        ext=".tiff",
        dataset_repeat=1,
        hist_matching_params_init=None,
        scale_down=True,
        use_one_channel_mask=False,
        to_rgb=False,
        imread_backend="cv2",
        crop_size=None,
        notempty_crop_prob=0.5,
        image_interpolation=cv2.INTER_NEAREST,
        contour_config=None,
        igor_overlap_config=None,
        centroid_config=None,
        organs_include=None,
        apply_torchstain_norm=False,
    ):
        super().__init__(
            df,
            root=root,
            name_col=name_col,
            organ_col=organ_col,
            height_col=height_col,
            width_col=width_col,
            cls_weight_col=cls_weight_col,
            tgt_col=tgt_col,
            transform=transform,
            last_transform=last_transform,
            precompute=precompute,
            test_mode=test_mode,
            n_cores=n_cores,
            img_size=img_size,
            dynamic_resize_mode=dynamic_resize_mode,
            additional_scalers=additional_scalers,
            debug=debug,
            do_cutmix=do_cutmix,
            cutmix_params=cutmix_params,
            do_mosaic=do_mosaic,
            mosaic_params=mosaic_params,
            cutmix_transform=cutmix_transform,
            ext=ext,
            dataset_repeat=dataset_repeat,
            hist_matching_params_init=hist_matching_params_init,
            scale_down=scale_down,
            use_one_channel_mask=use_one_channel_mask,
            to_rgb=to_rgb,
            imread_backend=imread_backend,
            crop_size=crop_size,
            notempty_crop_prob=notempty_crop_prob,
            image_interpolation=image_interpolation,
            contour_config=contour_config,
            igor_overlap_config=igor_overlap_config,
            centroid_config=centroid_config,
            organs_include=organs_include,
            apply_torchstain_norm=apply_torchstain_norm,
        )
        self.pseudo_df = pd.read_csv(pseudo_df_path)
        if organs_include is not None:
            self.pseudo_df = self.pseudo_df[self.pseudo_df[organ_col].isin(organs_include)].reset_index(drop=True)
        self.pseudo_df[f"{name_col}_with_root"] = self.pseudo_df[name_col].apply(
            lambda x: os.path.join(pseudo_root, str(x)) + pseudo_ext
        )
        assert pseudo_rate > 0, "rate should be greater than 0"
        self.pseudo_samples = int(len(self.df) * self.dataset_repeat * pseudo_rate)
        self.p_pseudo = self.pseudo_samples / (self.pseudo_samples + len(self.df) * self.dataset_repeat)
        self.pseudo_organs = set(self.pseudo_df[self.organ_col])
        print(
            f"{self.pseudo_samples} pseudo images will be added each epoch. "
            f"{self.p_pseudo} - counted probability of pseudo sample. "
            f"Pseudo contains next organs: {self.pseudo_organs}"
        )

    def _prepare_img_and_friends_from_idx(self, index: int, organ_name: str = None):
        # If index less then len(self.df) * dataset_repeat
        # Then sample from original data
        # Otherwise select random pseudo sample
        if index < int(len(self.df) * self.dataset_repeat):
            assert organ_name is None, "organ_name can be specified only for pseudo case"
            index = index % len(self.df)
            organ = self.df[self.organ_col].iloc[index]
            h = self.df[self.height_col].iloc[index]
            w = self.df[self.width_col].iloc[index]
            im_id = self.df[self.id_col].iloc[index]
            mask_rle = self.df[self.tgt_col].iloc[index]
            im_path = self.df[self.name_col].iloc[index]
            is_pseudo_sample = False
        else:
            if organ_name is not None:
                index = np.random.choice(np.where(self.pseudo_df[self.organ_col] == organ_name)[0])
            else:
                index = np.random.randint(low=0, high=len(self.pseudo_df))
            organ = self.pseudo_df[self.organ_col].iloc[index]
            h = self.pseudo_df[self.height_col].iloc[index]
            w = self.pseudo_df[self.width_col].iloc[index]
            im_id = self.pseudo_df[self.id_col].iloc[index]
            mask_rle = self.pseudo_df[self.tgt_col].iloc[index]
            im_path = self.pseudo_df[self.name_col].iloc[index]
            is_pseudo_sample = True

        if self.dynamic_resize_mode is not None or self.additional_scalers is not None:
            rescaled_w, rescaled_h = self._get_dynamic_size(h=h, w=w, organ=organ)
        else:
            rescaled_w, rescaled_h = -1, -1

        # Pseudo dataset is pretty big, so we will upload from disk pseudo images
        if self.precompute and not is_pseudo_sample:
            if self.test_mode:
                mask = -1
            else:
                mask = self.mask_cache[index]
            image = self.image_cache[index]
        else:
            if self.test_mode:
                mask = -1
            else:
                mask = rle_decode_multichannel_mask(
                    class_name=organ,
                    mask_rle=mask_rle,
                    shape=(w, h),
                    color=1,
                    one_channel=self.use_one_channel_mask,
                )
                if self.image_size is not None:
                    mask = cv2.resize(mask, self.image_size, interpolation=self.image_interpolation)
                if self.dynamic_resize_mode is not None or self.additional_scalers is not None:
                    mask = cv2.resize(mask, (rescaled_w, rescaled_h), interpolation=self.image_interpolation)
            image = self.imread(im_path, backend=self.imread_backend)
            if self.image_size is not None:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
            if self.dynamic_resize_mode is not None or self.additional_scalers is not None:
                image = cv2.resize(image, (rescaled_w, rescaled_h), interpolation=cv2.INTER_AREA)
            if self.apply_torchstain_norm:
                image = self.torchstain_normalizer.normalize(I=image, stains=False)[0]

        # Handle one channel case
        if not self.test_mode and mask.ndim < 3:
            mask = mask[:, :, None]

        if self.cutmix_transform is not None:
            result = self.cutmix_transform(image=image, mask=mask, index=index)
            image, mask = result["image"], result["mask"]

        if self.crop_size is not None:
            image, mask = self._crop_image_mask(index=index, image=image, mask=mask)

        if self.dynamic_resize_mode is not None and self.dynamic_resize_mode.split("_")[0] == "scale":
            pixel_size = PIXEL_SIZE[organ]
        else:
            pixel_size = PIXEL_SIZE["unscaled"]
        if self.additional_scalers is not None:
            if self.scale_down:
                pixel_size /= self.additional_scalers[organ]
            else:
                pixel_size *= self.additional_scalers[organ]
        pixel_size = torch.scalar_tensor(pixel_size)

        return image, mask, organ, h, w, rescaled_h, rescaled_w, im_id, pixel_size

    def _prepare_img_and_friends_from_idx_lvl2(self, index: int, organ_name: str = None):
        image, mask, organ, h, w, rescaled_h, rescaled_w, im_id, pixel_size = self._prepare_img_and_friends_from_idx(
            index, organ_name=organ_name
        )

        # Cutmix
        if self.do_cutmix and np.random.binomial(n=1, p=self.cutmix_params["prob"]):
            if np.random.uniform() < self.p_pseudo and organ in self.pseudo_organs:
                cutmix_ix = self.__len__()
                specified_organ = organ
            else:
                cutmix_ix = np.random.choice(np.where(self.df[self.organ_col] == organ)[0])
                specified_organ = None
            (
                cutmix_image,
                cutmix_mask,
                cutmix_organ,
                *_,
            ) = self._prepare_img_and_friends_from_idx(cutmix_ix, organ_name=specified_organ)
            assert cutmix_organ == organ
            lam_cutmix = np.random.beta(self.cutmix_params["alpha"], self.cutmix_params["alpha"])
            bbx1, bby1, bbx2, bby2, lam_cutmix = rand_bbox(image.shape, lam_cutmix)
            mixed_image, mixed_mask = image.copy(), mask.copy()
            mixed_image[bbx1:bbx2, bby1:bby2] = cutmix_image[bbx1:bbx2, bby1:bby2]
            mixed_mask[bbx1:bbx2, bby1:bby2] = cutmix_mask[bbx1:bbx2, bby1:bby2]
            mask = np.clip(mask, 0.0, 1.0)
        else:
            mixed_image = image
            mixed_mask = mask

        if self.test_mode:
            if self.transform is not None:
                mixed_image = self.transform(image=mixed_image, class_name=organ)["image"]
        else:
            if self.transform is not None:
                result = self.transform(image=mixed_image, mask=mixed_mask, class_name=organ)
                mixed_image, mixed_mask = result["image"], result["mask"]

        return mixed_image, mixed_mask, organ, h, w, rescaled_h, rescaled_w, im_id, pixel_size

    def __getitem__(self, index):

        (
            image,
            mask,
            organ,
            h,
            w,
            rescaled_h,
            rescaled_w,
            im_id,
            pixel_size,
        ) = self._prepare_img_and_friends_from_idx_lvl2(index)

        if self.do_mosaic:
            image_cache, mask_cache = [], []
            for _ in range(3):
                if np.random.uniform() < self.p_pseudo and organ in self.pseudo_organs:
                    mosaic_ix = self.__len__()
                    specified_organ = organ
                else:
                    mosaic_ix = np.random.choice(np.where(self.df[self.organ_col] == organ)[0])
                    specified_organ = None
                (
                    mosaic_image,
                    mosaic_mask,
                    mosaic_organ,
                    *_,
                ) = self._prepare_img_and_friends_from_idx_lvl2(mosaic_ix, organ_name=specified_organ)
                assert mosaic_organ == organ
                image_cache.append(mosaic_image)
                mask_cache.append(mosaic_mask)
            result = self.mosaic(image=image, mask=mask, image_cache=image_cache, mask_cache=mask_cache)
            image, mask = result["image"], result["mask"]

        if self.test_mode:
            if self.last_transform is not None:
                image = self.last_transform(image=image)["image"].float()
        else:
            if self.last_transform is not None:
                result = self.last_transform(image=image, mask=mask)
                image, mask = result["image"].float(), result["mask"].float()

        is_numpy = isinstance(mask, np.ndarray)
        use_add_masks = (
            self.contour_config is not None or self.igor_overlap_config is not None or self.centroid_config is not None
        )

        if use_add_masks:
            add_masks = []

        if not self.test_mode and self.contour_config is not None:
            contour_mask = get_contour_mask(
                numpy_float_mask=mask[:, :, 0] if is_numpy else mask[0].numpy(), **self.contour_config
            )
            if is_numpy:
                add_masks.append(contour_mask[:, :, None])
            else:
                add_masks.append(torch.from_numpy(contour_mask).unsqueeze(dim=0))

        # Mask instances are required for both centroids and igor_masks_overlap
        if not self.test_mode and (self.igor_overlap_config is not None or self.centroid_config is not None):
            mask_instances = get_mask_instances(numpy_float_mask=mask[:, :, 0] if is_numpy else mask[0].numpy())

        if not self.test_mode and self.igor_overlap_config is not None:
            if mask_instances is not None:
                igor_mask_overlap = get_igor_masks_overlap(masks=mask_instances, **self.igor_overlap_config)
            else:
                igor_mask_overlap = np.zeros(mask[0].shape, np.float32)
            if is_numpy:
                add_masks.append(igor_mask_overlap[:, :, None])
            else:
                add_masks.append(torch.from_numpy(igor_mask_overlap).unsqueeze(dim=0))

        if not self.test_mode and self.centroid_config is not None:
            if mask_instances is not None:
                centroid = get_centroid(masks=mask_instances, **self.centroid_config)
            else:
                centroid = np.zeros(mask[0].shape, np.float32)
            if is_numpy:
                add_masks.append(centroid[:, :, None])
            else:
                add_masks.append(torch.from_numpy(centroid).unsqueeze(dim=0))

        if use_add_masks:
            if is_numpy:
                mask = np.concatenate([mask] + add_masks, axis=-1)
            else:
                mask = torch.cat([mask] + add_masks, dim=0)

        if self.test_mode:
            return image, mask, organ, h, w, rescaled_h, rescaled_w, im_id, pixel_size
        else:
            return image, mask, organ, pixel_size

    def __len__(self):
        return int(len(self.df) * self.dataset_repeat) + self.pseudo_samples
