import json
from pathlib import Path

import tyro
from experiments.config import TrainingConfig


def save_arguments_to_file(
    output: Path,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    config: TrainingConfig,
) -> None:
    training_class = config.training_class(
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    training_arguments = training_class._to_huggingface_args_dict()

    output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(training_arguments, open(output, "w"))


if __name__ == "__main__":
    tyro.cli(save_arguments_to_file, config=[tyro.conf.OmitArgPrefixes])
