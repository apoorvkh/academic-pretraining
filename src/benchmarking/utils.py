from typing import Self

from accelerate.utils import DummyOptim, DummyScheduler
from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback


class ForceExit(Exception):
    pass


class ExitBeforeSteps(TrainerCallback):
    def on_step_begin(self, *args, **kwargs):
        raise ForceExit()


##


def has_deepspeed_engine(trainer: Trainer) -> bool:
    return (
        hasattr(trainer.accelerator, "deepspeed_engine_wrapped")
        and trainer.accelerator.deepspeed_engine_wrapped is not None
    )


class DeepSpeedEngineWrapperManualStep(DeepSpeedEngineWrapper):
    def backward(self, loss, **kwargs):
        self.engine.backward(loss, **kwargs)
        # Modification: doesn't run engine.step()

    def step(self):
        self.engine.step()


##


class ManualTrainer(Trainer):
    @classmethod
    def from_trainer(cls, trainer: Trainer) -> Self:
        ## prepare trainer by running train() but stop before first training step
        cb = ExitBeforeSteps()
        trainer.add_callback(cb)
        try:
            trainer.train()
        except ForceExit:
            pass
        trainer.remove_callback(cb)
        ##

        ## manually step DeepSpeed optimizer (i.e. not in accelerator.backwards)
        if has_deepspeed_engine(trainer):
            trainer.accelerator.deepspeed_engine_wrapped.__class__ = DeepSpeedEngineWrapperManualStep  # pyright: ignore [reportOptionalMemberAccess]
        ##

        trainer.__class__ = cls
        return trainer  # pyright: ignore [reportReturnType]

    def manual_training_step(self, model, inputs):
        with self.accelerator.accumulate(model):
            self.training_step(model, inputs)

    def manual_optimization_step(self, model):
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            _grad_norm = self.accelerator.clip_grad_norm_(
                model.parameters(),
                self.args.max_grad_norm,
            )

        if has_deepspeed_engine(self):
            self.accelerator.deepspeed_engine_wrapped.step()  # pyright: ignore

        if self.optimizer is not None and not isinstance(self.optimizer, DummyOptim):
            self.optimizer.step()
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, DummyScheduler):
            self.lr_scheduler.step()

        model.zero_grad()
