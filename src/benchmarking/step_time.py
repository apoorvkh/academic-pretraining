import gc
import logging
import time
from contextlib import contextmanager

import torch.cuda
from transformers import Trainer

from .utils import ManualTrainer

logger = logging.getLogger("academic-pretraining")


@contextmanager
def perf_timer():
    t1 = t2 = time.perf_counter()
    yield lambda: t2 - t1
    t2 = time.perf_counter()


@contextmanager
def get_train_dataloader(trainer: Trainer, micro_batch_size: int):
    _original_trainer_mbs = trainer.args.per_device_train_batch_size
    try:
        trainer.args.__dict__.update(per_device_train_batch_size=micro_batch_size)
        trainer._train_batch_size = trainer.args.train_batch_size
        yield iter(trainer.get_train_dataloader())
    finally:
        trainer.args.__dict__.update(per_device_train_batch_size=_original_trainer_mbs)
        trainer._train_batch_size = trainer.args.train_batch_size


@torch._dynamo.config.patch(verbose=True, suppress_errors=True, cache_size_limit=64)  # pyright: ignore [reportAttributeAccessIssue]
def benchmark_acc_optim_times(
    trainer: ManualTrainer,
    micro_batch_size: int,
    training_steps: int = 1,
    accumulations: int = 1,
    warmup: bool = False,
) -> tuple[float, float]:
    gc.collect()
    torch.cuda.empty_cache()

    accumulation_times = []
    optimization_times = []

    if warmup:
        training_steps += 1

    model = trainer.model_wrapped
    with get_train_dataloader(trainer, micro_batch_size) as train_dataloader:
        for _ in range(training_steps):
            for _ in range(accumulations):
                inputs = next(train_dataloader)
                with perf_timer() as t:
                    trainer.manual_training_step(model, inputs)
                accumulation_times.append(t())

            with perf_timer() as t:
                trainer.manual_optimization_step(model)
            optimization_times.append(t())

    if warmup:
        accumulation_times = accumulation_times[1:]
        optimization_times = optimization_times[1:]

    logger.info(f"Accumulation times: {accumulation_times}")
    logger.info(f"Optimization times: {optimization_times}")

    mean_acc_time = sum(accumulation_times) / len(accumulation_times)
    mean_optim_time = sum(optimization_times) / len(optimization_times)
    return mean_acc_time, mean_optim_time


def estimate_step_time(
    trainer: ManualTrainer,
    micro_batch_size: int,
    target_micro_batch_size: int,
    num_benchmarking_steps: int,
) -> float:
    accumulation_steps = target_micro_batch_size // micro_batch_size

    logger.info(
        f"Estimating step time for MBS = {micro_batch_size}, ACC = {accumulation_steps}",
    )

    mean_acc_time, mean_optim_time = benchmark_acc_optim_times(
        trainer,
        micro_batch_size,
        training_steps=num_benchmarking_steps,
        accumulations=1,
        warmup=True,
    )

    mean_step_time = mean_acc_time * accumulation_steps + mean_optim_time

    return mean_step_time
