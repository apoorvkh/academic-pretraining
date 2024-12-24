import json
import os
from pathlib import Path
from typing import Any

import torch.optim
import torchrunx
import tyro
from accelerate.utils import check_cuda_p2p_ib_support
from src.models import ModelT, get_model_class
from torch.utils.data import Dataset
from transformers import PreTrainedModel, Trainer, TrainingArguments


def get_model(model_type: ModelT) -> PreTrainedModel:
    return get_model_class(model_type).build_model(use_custom_kernels=True)


def get_dataset(model_type: ModelT) -> Dataset:
    return get_model_class(model_type).load_dummy_dataset()


def get_optimizer_cls_and_kwargs(
    model_type: ModelT, using_deepspeed: bool
) -> tuple[type[torch.optim.Optimizer], dict[str, Any]] | None:
    model_class = get_model_class(model_type)

    if using_deepspeed and model_class.optimizer in [
        torch.optim.Adam,
        torch.optim.AdamW,
    ]:  # use Deepspeed's Adam optimizer
        return None

    return (model_class.optimizer, model_class.optimizer_kwargs)


def train(output_dir: str, model_type: ModelT, training_arguments: dict[str, Any]):
    if check_cuda_p2p_ib_support() is False:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"

    model = get_model(model_type)
    train_dataset = get_dataset(model_type)
    optimizer_cls_and_kwargs = get_optimizer_cls_and_kwargs(
        model_type, using_deepspeed=(training_arguments.get("deepspeed") is not None)
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            **training_arguments,
        ),
        train_dataset=train_dataset,
        optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
    )

    trainer.train()


def run(launcher: torchrunx.Launcher, output_dir: str, model_type: ModelT, training_arguments: Path):
    training_arguments = json.load(open(training_arguments, "r"))
    launcher.run(
        func=train,
        func_kwargs=dict(
            output_dir=output_dir,
            model_type=model_type,
            training_arguments=training_arguments,
        ),
    )


if __name__ == "__main__":
    tyro.cli(run)
