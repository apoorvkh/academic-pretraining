import logging

import torch.cuda

from .step_time import benchmark_acc_optim_times
from .utils import ManualTrainer

logger = logging.getLogger("academic-pretraining")


def find_max_mbs_pow2(trainer: ManualTrainer, limit: int) -> int:
    mbs = 1

    while mbs <= limit:
        logger.info(f"Running 1 training step with MBS = {mbs} ...")
        try:
            benchmark_acc_optim_times(trainer=trainer, micro_batch_size=mbs, training_steps=1, accumulations=1)
        except torch.cuda.OutOfMemoryError:
            break
        mbs *= 2

    # here, mbs will either have failed or be > limit
    # so (mbs // 2) should be last working size or zero

    return mbs // 2
