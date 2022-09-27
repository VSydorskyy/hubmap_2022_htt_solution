import cv2
import numpy as np

from ..constants import CLASSES


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# modified from: https://www.kaggle.com/inversion/run-length-decoding-quick-start
def rle_decode(mask_rle, shape, color=1):
    """TBD
    Args:
        mask_rle (str): run-length as string formated (start length)
        shape (tuple of ints): (height,width) of array to return
    Returns:
        Mask (np.array)
            - 1 indicating mask
            - 0 indicating background
    """
    # Split the string by space, then convert it into a integer array
    s = np.array(mask_rle.split(), dtype=int)

    # Every even value is the start, every odd value is the "run" length
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    # The image image is actually flattened since RLE is a 1D "run"
    if len(shape) == 3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
    else:
        h, w = shape
        img = np.zeros((h * w,), dtype=np.float32)

    # The color here is actually just any integer you want!
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color

    # Don't forget to change the image back to the original shape
    return img.reshape(shape).T


# https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
# with transposed mask
def rle_encode_less_memory(img):
    # the image should be transposed
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def delete_small_segments_from_mask(bool_mask, min_size):
    assert len(bool_mask.shape) == 2

    num_component, component = cv2.connectedComponents(bool_mask.astype(np.uint8))

    pp_bool_mask = np.zeros(bool_mask.shape, bool)
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            pp_bool_mask[p] = True

    return pp_bool_mask


def get_mask_instances(numpy_float_mask):
    num_component, component = cv2.connectedComponents((numpy_float_mask.copy() > 0.5).astype(np.uint8))
    if num_component == 1:
        return None
    else:
        return [(component == c).astype(np.uint8) for c in range(1, num_component)]


def get_igor_masks_overlap(masks, min_dist=7):
    if len(masks):
        dilated_mask = np.sum(
            [cv2.dilate(src=m, kernel=get_igor_kernel_size(m, min_dist=min_dist), iterations=1) for m in masks], axis=0
        )
        return (dilated_mask > 1).astype(np.float32)
    else:
        return None


def get_igor_kernel_size(mask, min_dist=7):
    kernel = int(np.sqrt(np.sqrt(mask.sum())))
    return np.ones((kernel, kernel), np.uint8) if kernel > min_dist else np.ones((min_dist, min_dist), np.uint8)


def get_centroid(masks, min_radius=5):
    cntrs = np.zeros(masks[0].shape)
    for mask in masks:
        if mask.sum() > 0:
            m = cv2.moments(mask)
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            radius = int(np.sqrt(np.sqrt(mask.sum())))
            cntrs = cv2.circle(cntrs, (cx, cy), radius if radius > min_radius else min_radius, 1, -1)
    return cntrs.astype(np.float32)


def overlay_mask(img, mask, color_id=0):
    color = np.ones(mask.shape) * 255
    img_c = img.copy()
    img_c[:, :, color_id][mask > 0.5] = color[mask > 0.5]
    return img_c


def rle_decode_multichannel_mask(class_name, mask_rle, shape, color=1, one_channel=False):
    if one_channel:
        return rle_decode(mask_rle, shape, color=color)
    else:
        mask = np.zeros((*shape, len(CLASSES)))
        mask[:, :, CLASSES.index(class_name)] = rle_decode(mask_rle, shape, color=color)
        return mask


def get_pads(
    original_height, original_width, min_height=None, min_width=None, pad_height_divisor=None, pad_width_divisor=None
):
    rows = original_height
    cols = original_width
    if min_height is not None:
        if rows < min_height:
            h_pad_top = int((min_height - rows) / 2.0)
            h_pad_bottom = min_height - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0
    else:
        pad_remained = rows % pad_height_divisor
        pad_rows = pad_height_divisor - pad_remained if pad_remained > 0 else 0

        h_pad_top = pad_rows // 2
        h_pad_bottom = pad_rows - h_pad_top

    if min_width is not None:
        if cols < min_width:
            w_pad_left = int((min_width - cols) / 2.0)
            w_pad_right = min_width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0
    else:
        pad_remainder = cols % pad_width_divisor
        pad_cols = pad_width_divisor - pad_remainder if pad_remainder > 0 else 0

        w_pad_left = pad_cols // 2
        w_pad_right = pad_cols - w_pad_left

    return {
        "pad_top": h_pad_top,
        "pad_bottom": h_pad_bottom,
        "pad_left": w_pad_left,
        "pad_right": w_pad_right,
    }


def get_contour_mask(numpy_float_mask, thickness=1):
    if numpy_float_mask.sum() > 0.0:
        contours, _ = cv2.findContours(
            (numpy_float_mask.copy() > 0.5).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        mask_contours = np.zeros(numpy_float_mask.shape, np.uint8)
        mask_contours = cv2.drawContours(mask_contours, contours, -1, (1), thickness)
        return mask_contours.astype(np.float32)
    else:
        return np.zeros(numpy_float_mask.shape, np.float32)
