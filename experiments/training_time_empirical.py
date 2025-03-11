import dataclasses
import math
import tempfile
from dataclasses import dataclass
from typing import Any, TypedDict

import torch
import torchrunx
from src.benchmarking.max_batch_size import find_max_mbs_pow2
from src.benchmarking.step_time import estimate_step_time
from src.benchmarking.utils import ManualTrainer
from tango import Step

from experiments import Experiment, SlurmJob, step
from experiments.config import TrainingConfig


def build_benchmarking_trainer(config: TrainingConfig, disable_compile: bool = False) -> ManualTrainer:
    training_class = config.training_class(
        num_training_steps=1,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
    )
    if training_class.compile and disable_compile:
        training_class = dataclasses.replace(training_class, compile=False)

    model_class = config.model_class()
    model = model_class.build_model(use_custom_kernels=config.free_lunch)
    train_dataset = model_class.load_dummy_dataset()

    trainer = training_class.build_trainer(
        model=model,
        train_dataset=train_dataset,
        hf_training_args_overrides=dict(
            output_dir=tempfile.mkdtemp(),
            save_strategy="no",
            report_to="none",
        ),
    )

    return ManualTrainer.from_trainer(trainer)


def find_largest_batch_size_worker(config: TrainingConfig, limit: int):
    try:
        trainer = build_benchmarking_trainer(config, disable_compile=True)
    except torch.cuda.OutOfMemoryError:
        return 0
    return find_max_mbs_pow2(trainer, limit=limit)


@step(cacheable=True, version="001")
def find_largest_batch_size(config: TrainingConfig, limit: int) -> int:
    return (
        torchrunx.Launcher(workers_per_host=config.gpus_per_node)
        .run(find_largest_batch_size_worker, config=config, limit=limit)
        .rank(0)
    )


class BenchmarkingResults(TypedDict):
    micro_batch_size: int
    step_time: float
    compile_disabled: bool


def benchmark_step_time_worker(
    config: TrainingConfig,
    disable_compile: bool,
    micro_batch_size: int,
    target_micro_batch_size: int,
    num_benchmarking_steps: int,
) -> BenchmarkingResults | None:
    try:
        trainer = build_benchmarking_trainer(config, disable_compile=disable_compile)
        step_time = estimate_step_time(trainer, micro_batch_size, target_micro_batch_size, num_benchmarking_steps)
        return BenchmarkingResults(
            micro_batch_size=micro_batch_size,
            step_time=step_time,
            compile_disabled=disable_compile,
        )
    except torch.cuda.OutOfMemoryError:
        return None


@step(cacheable=True, version="001")
def benchmark_step_time(
    config: TrainingConfig,
    max_micro_batch_size: int,
    target_micro_batch_size: int,
    num_benchmarking_steps: int,
    trial: int = 0,
) -> BenchmarkingResults | None:
    micro_batch_size = max_micro_batch_size

    while micro_batch_size > 0:
        try:
            benchmark_results = (
                torchrunx.Launcher(workers_per_host=config.gpus_per_node)
                .run(
                    benchmark_step_time_worker,
                    config=config,
                    disable_compile=False,
                    micro_batch_size=micro_batch_size,
                    target_micro_batch_size=target_micro_batch_size,
                    num_benchmarking_steps=num_benchmarking_steps,
                )
                .rank(0)
            )
        except RuntimeError:
            if config.free_lunch:
                print("Possible time-out during compile, trying again without compiling...")
                benchmark_results = (
                    torchrunx.Launcher(workers_per_host=config.gpus_per_node)
                    .run(
                        benchmark_step_time_worker,
                        config=config,
                        disable_compile=True,
                        micro_batch_size=micro_batch_size,
                        target_micro_batch_size=target_micro_batch_size,
                        num_benchmarking_steps=num_benchmarking_steps,
                    )
                    .rank(0)
                )
            else:
                raise

        if benchmark_results is not None:
            return benchmark_results

        micro_batch_size //= 2

    return None


@step(cacheable=True, version="001")
def compute_training_days(benchmarking_results: BenchmarkingResults | None, num_steps: int) -> float | None:
    """total training time in days"""
    if benchmarking_results is None:
        return None
    return (num_steps * benchmarking_results["step_time"]) / (24 * 60 * 60)


## Experiment


@dataclass
class TrainingTimeEmpirical(Experiment):
    config: TrainingConfig
    benchmarking_steps: int = 3
    trial: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial": self.trial,
            **self.config.__dict__,
            "benchmarking_steps": self.benchmarking_steps,
        }

    def __post_init__(self):
        self.model_class = self.config.model_class()
        self.training_class = self.config.training_class()

    def is_valid(self) -> bool:
        if any(
            [
                self.benchmarking_steps <= 0,
                self.trial < 0,
                # model batch size should be evenly divisible by total GPUs
                self.model_class.batch_size % (self.config.num_nodes * self.config.gpus_per_node) > 0,
                # batch size per gpu should be power of 2
                not math.log2(
                    self.model_class.batch_size // (self.config.num_nodes * self.config.gpus_per_node)
                ).is_integer(),
                # if activation checkpointing is enabled, model should support it
                self.config.activation_checkpointing and (not self.model_class.supports_activation_checkpointing),
                # data types for ampere or newer GPUs
                self.model_class.mixed_precision == "bf16" and not self.config.ampere_or_newer_gpu(),
                # don't shard for a single GPU (no-op)
                self.config.num_nodes == 1
                and self.config.gpus_per_node == 1
                and self.config.sharding != ""
                and not self.config.offloading,
                # offloading requires sharding
                (self.config.offloading and self.config.sharding == ""),
            ]
        ):
            return False
        return self.training_class.is_valid()

    @property
    def target_micro_batch_size(self) -> int:
        return self.model_class.batch_size // (self.config.num_nodes * self.config.gpus_per_node)

    @property
    def step_dict(self) -> dict[str, Step]:
        steps = {}
        steps["max_micro_batch_size"] = find_largest_batch_size(config=self.config, limit=self.target_micro_batch_size)
        steps["benchmarking_results"] = benchmark_step_time(
            config=self.config,
            max_micro_batch_size=steps["max_micro_batch_size"],
            target_micro_batch_size=self.target_micro_batch_size,
            num_benchmarking_steps=self.benchmarking_steps,
            trial=self.trial,
        )
        steps["training_days"] = compute_training_days(
            benchmarking_results=steps["benchmarking_results"],
            num_steps=self.model_class.training_steps,
        )
        return steps

    @property
    def slurm_job(self) -> SlurmJob | None:
        return SlurmJob(
            time_min=60,
            num_nodes=self.config.num_nodes,
            mem_per_node=(64 * self.config.gpus_per_node),
            cpus_per_node=(4 * self.config.gpus_per_node),
            gpus_per_node=self.config.gpus_per_node,
            gpu_type=self.config.gpu_type,
        )

    def results(self):
        return {
            "max_micro_batch_size": self.step_result("max_micro_batch_size"),
            **(self.step_result("benchmarking_results") or {}),
            "training_days": self.step_result("training_days"),
        }


if __name__ == "__main__":
    TrainingTimeEmpirical.cli()
