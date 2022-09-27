import random
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from albumentations.augmentations.bbox_utils import (
        convert_bboxes_to_albumentations,
        denormalize_bbox,
        normalize_bbox,
    )
except:
    print("albumentations.augmentations.bbox_utils was not imported")
from albumentations.core.transforms_interface import DualTransform


def mosaic4(image_batch: List[np.ndarray], height: int, width: int, fill_value: int = 0) -> np.ndarray:
    """Arrange the images in a 2x2 grid. Images can have different shape.
    This implementation is based on YOLOv5 with some modification:
    https://github.com/ultralytics/yolov5/blob/932dc78496ca532a41780335468589ad7f0147f7/utils/datasets.py#L648
    Args:
        image_batch (List[np.ndarray]): image list. The length should be 4.
        height (int): Height of output mosaic image
        width (int): Width of output mosaic image
        fill_value (int): padding value
    """
    if len(image_batch) != 4:
        raise ValueError(f"Length of image_batch should be 4. Got {len(image_batch)}")

    if len(image_batch[0].shape) == 2:
        out_shape = [height, width]
    else:
        out_shape = [height, width, image_batch[0].shape[2]]

    center_x = width // 2
    center_y = height // 2
    img4 = np.full(out_shape, fill_value, dtype=np.uint8)  # base image with 4 tiles
    for i, img in enumerate(image_batch):
        (h, w) = img.shape[:2]

        # place img in img4
        # this based on the yolo5's implementation
        #
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = (
                max(center_x - w, 0),
                max(center_y - h, 0),
                center_x,
                center_y,
            )  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = (
                center_x,
                max(center_y - h, 0),
                min(center_x + w, width),
                center_y,
            )
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = (
                max(center_x - w, 0),
                center_y,
                center_x,
                min(height, center_y + h),
            )
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = (
                center_x,
                center_y,
                min(center_x + w, width),
                min(height, center_y + h),
            )
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

    return img4


def bbox_mosaic4(bbox: Tuple, rows: int, cols: int, position_index: int, height: int, width: int):
    """Put the given bbox in one of the cells of the 2x2 grid.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)` or `(x_min, y_min, x_max, y_max, label, ...)`.
        rows (int): Height of input image that corresponds to one of the mosaic cells
        cols (int): Width of input image that corresponds to one of the mosaic cells
        position_index (int): Index of the mosaic cell. 0: top left, 1: top right, 2: bottom left, 3: bottom right
        height (int): Height of output mosaic image
        width (int): Width of output mosaic image
    """
    bbox = denormalize_bbox(bbox, rows, cols)
    bbox, tail = bbox[:4], tuple(bbox[4:])
    center_x = width // 2
    center_y = height // 2
    if position_index == 0:  # top left
        shift_x = center_x - cols
        shift_y = center_y - rows
    elif position_index == 1:  # top right
        shift_x = center_x
        shift_y = center_y - rows
    elif position_index == 2:  # bottom left
        shift_x = center_x - cols
        shift_y = center_y
    elif position_index == 3:  # bottom right
        shift_x = center_x
        shift_y = center_y
    bbox = (
        bbox[0] + shift_x,
        bbox[1] + shift_y,
        bbox[2] + shift_x,
        bbox[3] + shift_y,
    )

    bbox = normalize_bbox(bbox, height, width)
    return tuple(bbox + tail)


class Mosaic(DualTransform):
    """Mosaic augmentation arranges randomly selected four images into single one like the 2x2 grid layout.
    Note:
        This augmentation requires additional helper targets as sources of additional
        image and bboxes.
        The targets are:
        - `image_cache`: list of images or 4 dimensional np.nadarray whose first dimension is batch size.
        - `bboxes_cache`: list of bounding boxes. The bounding box format is specified in `bboxes_format`.
        You should make sure that the bounding boxes of i-th image (image_cache[i]) are given by bboxes_cache[i].
        Here is a typical usage:
        ```
        data = transform(image=image, image_cache=image_cache)
        # or
        data = transform(image=image, image_cache=image_cache, bboxes=bboxes, bboxes_cache=bboxes_cache)
        ```
        You can set `image_cache` whose length is less than 3. In such a case, the same image will be selected
        multiple times.
        Note that the image specified by `image` argument is always included.
    Args:
        height (int)): height of the mosaiced image.
        width (int): width of the mosaiced image.
        fill_value (int): padding value.
        replace (bool): whether to allow replacement in sampling or not. When the value is `True`, the same image
            can be selected multiple times. When False, the length of `image_cache` (and `bboxes_cache`) should
            be at least 3.
            This replacement rule is applied only to `image_cache`. So, if the `image_cache` contains the same image as
            the one specified in `image` argument, it can make image that includes duplication for the `image` even if
            `replace=False` is set.
        bboxes_forma (str)t: format of bounding box. Should be on of "pascal_voc", "coco", "yolo".
    Targets:
        image, mask, bboxes, image_cache, mask_cache, bboxes_cache
    Image types:
        uint8, float32
    Reference:
    [Bochkovskiy] Bochkovskiy A, Wang CY, Liao HYM. （2020） "YOLOv 4 : Optimal speed and accuracy of object detection.",
    https://arxiv.org/pdf/2004.10934.pdf
    """

    def __init__(
        self,
        height,
        width,
        replace=True,
        fill_value=0,
        bboxes_format="pascal_voc",
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.replace = replace
        self.fill_value = fill_value
        self.bboxes_format = bboxes_format
        self.__target_dependence = {}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("height", "width", "replace", "fill_value", "bboxes_cache_format")

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        self.update_target_dependence(**kwargs)
        return super().__call__(force_apply=force_apply, **kwargs)

    @property
    def target_dependence(self) -> Dict:
        return self.__target_dependence

    @target_dependence.setter
    def target_dependence(self, value):
        self.__target_dependence = value

    def update_target_dependence(self, **kwargs):
        """Update target dependence dynamically."""
        self.target_dependence = {}
        if "image" in kwargs:
            self.target_dependence["image"] = {"image_cache": kwargs["image_cache"]}
        if "mask" in kwargs:
            self.target_dependence["mask"] = {"mask_cache": kwargs["mask_cache"]}
        if "bboxes" in kwargs:
            self.target_dependence["bboxes"] = {
                "image": kwargs["image"],
                "image_cache": kwargs["image_cache"],
                "bboxes_cache": kwargs["image_cache"],
            }

    def apply(self, image, image_cache, indices, height, width, fill_value, **params):
        image_batch = []
        for i in indices:
            if i == 0:
                image_batch.append(image)
            else:
                image_batch.append(image_cache[i - 1])
        return mosaic4(image_batch, height, width, fill_value)

    def apply_to_mask(self, mask, mask_cache, indices, height, width, fill_value, **params):
        mask_batch = []
        for i in indices:
            if i == 0:
                mask_batch.append(mask)
            else:
                mask_batch.append(mask_cache[i - 1])
        return mosaic4(mask_batch, height, width, fill_value)

    def apply_to_bbox(self, bbox, image_shape, position, height, width, **params):
        rows, cols = image_shape[:2]
        return bbox_mosaic4(bbox, rows, cols, position, height, width)

    def apply_to_bboxes(
        self, bboxes, bboxes_cache, image, image_cache, indices, height, width, bboxes_format, **params
    ):
        new_bboxes = []
        for i, index in enumerate(indices):
            if index == 0:
                image_shape = image.shape
                target_bboxes = bboxes
            else:
                image_shape = image_cache[index - 1].shape
                target_bboxes = bboxes_cache[index - 1]
                rows, cols = image_shape[:2]
                target_bboxes = convert_bboxes_to_albumentations(
                    target_bboxes, source_format=bboxes_format, rows=rows, cols=cols
                )
            for bbox in target_bboxes:
                new_bbox = self.apply_to_bbox(bbox, image_shape, i, height, width)
                new_bboxes.append(new_bbox)
        return new_bboxes

    def apply_to_keypoint(self, **params):
        pass  # TODO

    def get_params(self) -> Dict[str, Any]:
        image_cache = self.target_dependence["image"]["image_cache"]
        n = len(image_cache)
        indices = 1 + np.random.choice(
            range(n), size=3, replace=self.replace
        )  # 3 additional image indices. The 0-th index is reserved for the target image.
        indices = [0] + list(indices)
        random.shuffle(indices)  # target image + additional images
        return {
            "indices": indices,
            "height": self.height,
            "width": self.width,
            "fill_value": self.fill_value,
            "bboxes_format": self.bboxes_format,
        }
