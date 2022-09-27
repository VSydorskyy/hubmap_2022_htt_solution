import numpy as np


def dice_coeff(gt, pred):
    assert len(gt.shape) == 2
    assert gt.shape == pred.shape

    denom = gt.sum().astype(np.float32) + pred.sum().astype(np.float32)
    if denom == 0:
        return 1.0

    intersect = np.sum(gt & pred).astype(np.float32)
    result = (2 * intersect) / denom
    return result
