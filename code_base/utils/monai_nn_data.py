import torch

from ..constants import CLASSES


def monai_x_2_to_model_input(x):
    return torch.permute(x, (0, 2, 3, 1)).unsqueeze(dim=1)


def monai_y_2_to_model_input(y):
    y = torch.stack(
        [y[:, i * len(CLASSES) : (i + 1) * len(CLASSES), :, :] for i in range(int(y.shape[1] // len(CLASSES)))], dim=2
    )
    return torch.permute(y, (0, 1, 3, 4, 2))


def monai_y_model_input_2_vanila(y):
    y_s = y.shape
    if len(y_s) == 5:
        return torch.permute(y, (0, 4, 1, 2, 3)).reshape(y_s[0] * y_s[4], y_s[1], y_s[2], y_s[3])
    elif len(y_s) == 4:
        return torch.permute(y, (3, 0, 1, 2))
    else:
        raise ValueError(f"{len(y_s)} - invalid number of dims")
