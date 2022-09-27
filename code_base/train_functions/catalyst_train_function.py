import json
from pprint import pprint
from typing import Callable, List, Optional, OrderedDict, Union

import numpy as np
import pandas as pd
import torch
from catalyst import dl, metrics

from ..inference import apply_avarage_weights_on_swa_path
from ..utils.other import get_mode_model, load_state_dict_not_strict, seed_everything


class CustomRunner(dl.Runner):
    def _dynamic_meters_updated(self, batch_metrics_dict):
        if len(batch_metrics_dict) > len(self.meters.keys()):
            additional_loss_metric_names = list(set(batch_metrics_dict.keys()) - set(self.meters.keys()))
            for add_key in additional_loss_metric_names:
                self.meters[add_key] = metrics.AdditiveMetric(compute_on_call=False)
        for key in batch_metrics_dict.keys():
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

    def on_epoch_start(self, runner: "IRunner"):
        if hasattr(runner.criterion, "on_epoch_start"):
            runner.criterion.on_epoch_start(runner)
        return super().on_epoch_start(runner)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {}

    def on_loader_end(self, runner):
        for key in self.meters.keys():
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def handle_batch(self, batch):

        losses, inputs, outputs = self.criterion(self, batch)

        self.batch_metrics.update(losses)
        self._dynamic_meters_updated(losses)
        self.input = inputs
        self.output = outputs

    def on_loader_end(self, runner):
        for key in self.meters.keys():
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def filter_dict(input_dict, model_part_name):
    filtered_dict = OrderedDict()
    for k in input_dict.keys():
        if model_part_name in k:
            new_k = k[len(model_part_name) + 1 :]
            filtered_dict[new_k] = input_dict[k]
    return filtered_dict


def load_checkpoint(checkpoint_path, input_model, checkpoint_type: str = "catalyst", strict_load: bool = True):
    if checkpoint_type == "catalyst":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not strict_load:
            input_model = load_state_dict_not_strict(model=input_model, chkp=checkpoint)
        else:
            input_model.load_state_dict(checkpoint)
    elif checkpoint_type == "swa":
        checkpoint = apply_avarage_weights_on_swa_path(checkpoint_path)
        try:
            if not strict_load:
                input_model = load_state_dict_not_strict(model=input_model, chkp=checkpoint)
            else:
                input_model.load_state_dict(checkpoint)
        except:
            print("Switching `use_distributed` to True")
            checkpoint = apply_avarage_weights_on_swa_path(checkpoint_path, use_distributed=True)
            if not strict_load:
                input_model = load_state_dict_not_strict(model=input_model, chkp=checkpoint)
            else:
                input_model.load_state_dict(checkpoint)
    else:
        raise RuntimeError(f"{checkpoint_type} is invalid value for `checkpoint_type`")
    return input_model


def catalyst_training(
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame],
    exp_name: str,
    seed: int,
    train_dataset_class: torch.utils.data.Dataset,
    val_dataset_class: Optional[torch.utils.data.Dataset],
    train_dataset_config: dict,
    val_dataset_config: Optional[dict],
    train_dataloader_config: dict,
    val_dataloader_config: Optional[dict],
    nn_model_class: torch.nn.Module,
    nn_model_config: dict,
    optimizer_init: Callable,
    scheduler_init: Callable,
    forward: Union[torch.nn.Module, Callable],
    n_epochs: int,
    catalyst_callbacks: Callable,
    main_metric: str,
    minimize_metric: bool,
    fp16: bool = False,
    valid_loader: str = "valid",
    pretrained_chekpoints: Optional[Union[List[str], str]] = None,
    checkpoint_type: str = "catalyst",
    use_one_checkpoint: bool = False,
    strict_checkpoint: bool = True,
    class_weights_path: Optional[str] = None,
    create_ema_model: bool = False,
    add_train_dataset_class: Optional[Union[torch.utils.data.Dataset, List[torch.utils.data.Dataset]]] = None,
    add_train_dataset_config: Optional[Union[dict, List[dict]]] = None,
    add_train_dataset_df_path: Optional[Union[Optional[str], List[Optional[str]]]] = None,
):
    # Set reproducibility
    seed_everything(seed)
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training Device : {device}")

    train_dataset = train_dataset_class(
        df=train_df,
        **train_dataset_config,
    )
    if add_train_dataset_class is not None and add_train_dataset_config is not None:
        if isinstance(add_train_dataset_class, list) and isinstance(add_train_dataset_config, list):
            assert len(add_train_dataset_class) == len(add_train_dataset_config) == len(add_train_dataset_df_path)
            print("Using additional train datasetS")
            train_dataset = [train_dataset]
            for add_train_dataset_class_solo, add_train_dataset_config_solo, add_train_dataset_df_path_solo in zip(
                add_train_dataset_class, add_train_dataset_config, add_train_dataset_df_path
            ):
                add_tain_df_solo = (
                    train_df if add_train_dataset_df_path_solo is None else pd.read_csv(add_train_dataset_df_path_solo)
                )
                train_dataset.append(
                    add_train_dataset_class_solo(
                        df=add_tain_df_solo,
                        **add_train_dataset_config_solo,
                    )
                )
            train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        else:
            print("Using additional train dataset")
            add_tain_df = train_df if add_train_dataset_df_path is None else pd.read_csv(add_train_dataset_df_path)
            add_train_dataset = add_train_dataset_class(
                df=add_tain_df,
                **add_train_dataset_config,
            )
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, add_train_dataset])

    val_dataset = val_dataset_class(
        df=val_df,
        **val_dataset_config,
    )
    if class_weights_path is not None:
        with open(class_weights_path) as f:
            class_weights = json.load(f)
        print("Using Sample Weights:")
        pprint(class_weights)
        # Hotfix - json convert keys to str
        class_weights = {int(k): v for k, v in class_weights.items()}
        sample_weights = np.array([class_weights[el] for el in train_dataset.sample_weights])
        assert len(sample_weights) == len(train_dataset)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
    else:
        sampler = None

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            **train_dataloader_config,
        ),
        "valid": torch.utils.data.DataLoader(val_dataset, worker_init_fn=worker_init_fn, **val_dataloader_config),
    }

    model = nn_model_class(device=device, **nn_model_config)

    if pretrained_chekpoints is not None:
        if use_one_checkpoint:
            print(f"Loading model: {pretrained_chekpoints}")
            model = load_checkpoint(pretrained_chekpoints, model, checkpoint_type, strict_load=strict_checkpoint)
        else:
            fold_id = int(exp_name.split("/")[-1].split("_")[-1])
            print(f"Loading model: {pretrained_chekpoints[fold_id]}")
            model = load_checkpoint(
                pretrained_chekpoints[fold_id], model, checkpoint_type, strict_load=strict_checkpoint
            )

    if create_ema_model:
        model = {"train": model}
        model["val"] = nn_model_class(device=device, **nn_model_config)
        model["val"].load_state_dict(model["train"].state_dict())

    print(get_mode_model(model))
    for k in loaders.keys():
        print(f"{k} Loader Len = {len(loaders[k])}")

    optimizer = optimizer_init(get_mode_model(model))
    scheduler = scheduler_init(optimizer, len(loaders["train"]))

    runner = CustomRunner()

    if not isinstance(forward, torch.nn.Module):
        forward = forward()

    runner.train(
        model=model,
        optimizer=optimizer,
        criterion=forward,
        scheduler=scheduler,
        loaders=loaders,
        logdir=exp_name,
        num_epochs=n_epochs,
        seed=seed,
        verbose=True,
        load_best_on_end=False,
        valid_loader=valid_loader,
        valid_metric=main_metric,
        timeit=True,
        minimize_valid_metric=minimize_metric,
        fp16=fp16,
        callbacks=catalyst_callbacks(),  # We need to call this to make unique objects
    )  # for each fold

    return runner
