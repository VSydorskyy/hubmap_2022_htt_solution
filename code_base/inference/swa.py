from collections import OrderedDict
from typing import List, Optional

import torch


def apply_avarage_weights_on_swa_path(
    swa_path: str,
    use_distributed: bool = False,
    take_best: Optional[int] = None,
    maximize: bool = True,
    verbose: bool = True,
):
    chkp = torch.load(
        swa_path,
        map_location="cpu",
    )
    chkp = sorted([v for k, v in chkp.items()], key=lambda x: -x[0] if maximize else x[0])
    if verbose:
        print(" ; ".join([f"Itter {v[2]} Score {v[0]}" for v in chkp]))
    return avarage_weights(
        [v[1] for v in chkp],
        delete_module=use_distributed,
        take_best=take_best,
    )


def avarage_weights(
    nn_weights: List[OrderedDict],
    delete_module: bool = False,
    take_best: Optional[int] = None,
):
    if take_best is not None:
        print("solo model")
        avaraged_dict = OrderedDict()
        for k in nn_weights[take_best].keys():
            if delete_module:
                new_k = k[len("module.") :]
            else:
                new_k = k

            avaraged_dict[new_k] = nn_weights[take_best][k]
    else:
        n_nns = len(nn_weights)
        if n_nns < 2:
            raise RuntimeError("Please provide more then 2 checkpoints")

        avaraged_dict = OrderedDict()
        for k in nn_weights[0].keys():
            if delete_module:
                new_k = k[len("module.") :]
            else:
                new_k = k

            avaraged_dict[new_k] = sum(nn_weights[i][k] for i in range(n_nns)) / float(n_nns)

    return avaraged_dict
