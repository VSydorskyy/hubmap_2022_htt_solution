import json
import os
import random
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
import tifffile as tifi
import torch
from joblib import delayed

try:
    import torchstain
except:
    print("torchstain was not imported")

from .parallel import ProgressParallel


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        result = json.dump(data, f, ensure_ascii=False, indent=4)
    return result


def get_mode_model(model, mode="train"):
    if isinstance(model, dict):
        return model[mode]
    return model


def get_domain_adaptation_images(
    glob_regex,
    rgb=False,
    additional_scalers=None,
    image_interpolation=cv2.INTER_NEAREST,
    apply_torchstain_norm=False,
    n_cores=64,
):
    if apply_torchstain_norm:
        torchstain_normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
    if isinstance(glob_regex, dict):
        print("Loading Domain Adaptation Images ...")
        images = {
            k: [imread_rgb(fname) if rgb else cv2.imread(fname) for fname in glob(v, recursive=True)]
            for k, v in glob_regex.items()
        }
        print("Domain Adaptation Images Loaded!")
        if additional_scalers is not None:
            print("Resizing Domain Adaptation Images ...")
            for k in images:
                images[k] = [
                    cv2.resize(
                        im,
                        (int(im.shape[1] / additional_scalers[k]), int(im.shape[0] / additional_scalers[k])),
                        interpolation=image_interpolation,
                    )
                    for im in images[k]
                ]
            print("Domain Adaptation Images resized!")
        if apply_torchstain_norm:
            print("Stain normalizing Domain Adaptation Images ...")
            for k in images:
                images[k] = ProgressParallel(n_jobs=n_cores, total=len(images[k]))(
                    delayed(torchstain_normalizer.normalize)(I=img, stains=False) for img in images[k]
                )
                # Take only normed image
                images[k] = [el[0] for el in images[k]]
            print("Stain normalized Domain Adaptation Images")
    else:
        print("Loading Domain Adaptation Images ...")
        if isinstance(glob_regex, list):
            im_names = []
            for one_regex in glob_regex:
                im_names.extend(glob(one_regex, recursive=True))
        else:
            im_names = glob(glob_regex, recursive=True)
        images = [imread_rgb(fname) if rgb else cv2.imread(fname) for fname in im_names]
        print(f"Domain Adaptation Images Loaded! Loaded {len(images)} images")
        if additional_scalers is not None:
            print("Resizing Domain Adaptation Images ...")
            images = [
                cv2.resize(
                    im,
                    (int(im.shape[1] / additional_scalers), int(im.shape[0] / additional_scalers)),
                    interpolation=image_interpolation,
                )
                for im in images
            ]
            print("Domain Adaptation Images resized!")
        if apply_torchstain_norm:
            print("Stain normalizing Domain Adaptation Images ...")
            images = ProgressParallel(n_jobs=n_cores, total=len(images))(
                delayed(torchstain_normalizer.normalize)(I=img, stains=False) for img in images
            )
            # Take only normed image
            images = [el[0] for el in images]
            print("Stain normalized Domain Adaptation Images")
    return images


def imread_rgb(path, backend="cv2", verbose=False):
    if backend == "cv2":
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    elif backend == "tifi":
        image = tifi.imread(path)
        if verbose:
            print(image.shape)
        if len(image.shape) > 3:
            image = image[0, 0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        return image
    else:
        raise ValueError(f"Unsupported backend ({backend})")


def load_state_dict_not_strict(model, chkp):
    model_chkp = model.state_dict()
    matched_keys = []
    new_chkp = OrderedDict()
    for k in model_chkp:
        if k in model_chkp and model_chkp[k].shape == chkp[k].shape:
            new_chkp[k] = chkp[k]
            matched_keys.append(k)
        else:
            new_chkp[k] = model_chkp[k]
    missed_keys = set(model_chkp.keys()) - set(matched_keys)
    if len(missed_keys) > 0:
        missed_keys = "\n".join(missed_keys)
        print(f"Missed keys:\n{missed_keys}")
    extra_keys = set(chkp.keys()) - set(matched_keys)
    if len(extra_keys) > 0:
        extra_keys = "\n".join(extra_keys)
        print(f"Extra keys:\n{extra_keys}")
    model.load_state_dict(new_chkp)
    return model


def seed_everything(seed: int, deterministic_torch_backends: bool = False):

    print(f"Fixing seed: {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic_torch_backends:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
