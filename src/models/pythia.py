from typing import Any, Literal

import torch.optim
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedModel,
    SchedulerType,
)

from . import LanguageModelClass, PythiaT


class PythiaModelClass(LanguageModelClass[PythiaT]):
    def build_model(self, use_custom_kernels: bool = True) -> PreTrainedModel:
        # to ensure same initialization as original Pythia training, use:
        # GPTNeoXForCausalLM.from_pretrained(revision="step0", ...)
        config = GPTNeoXConfig.from_pretrained(
            f"EleutherAI/{self.model_type}",
            attn_implementation=("sdpa" if use_custom_kernels else "eager"),
        )
        return GPTNeoXForCausalLM(config)

    @property
    def batch_size(self) -> int:
        return 1024

    @property
    def training_steps(self) -> int:
        return 143000

    @property
    def mixed_precision(self) -> Literal[None, "bf16", "fp16"]:
        """
        From Github README: "All models are trained with mixed precision, using fp16 for all models except
        EleutherAI/pythia-1b which trained with bf16, because in fp16 the model experienced an irreconcilable loss
        spike late in training."
        """
        if self.model_type == "pythia-1b":
            return "bf16"
        return "fp16"

    @property
    def optimizer(self) -> type[torch.optim.Optimizer]:
        return torch.optim.Adam

    @property
    def optimizer_kwargs(self) -> dict[str, Any]:
        match self.model_type:
            case "pythia-14m" | "pythia-31m" | "pythia-70m":
                lr = 1.0e-3
            case "pythia-160m":
                lr = 6.0e-4
            case "pythia-410m" | "pythia-1b":
                lr = 3.0e-4
            case "pythia-1.4b":
                lr = 2.0e-4
            case "pythia-2.8b":
                lr = 1.6e-4
            case "pythia-6.9b" | "pythia-12b":
                lr = 1.2e-4
        return {
            "lr": lr,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.01,
        }

    @property
    def scheduler_type(self) -> SchedulerType:
        return SchedulerType.COSINE_WITH_MIN_LR

    @property
    def scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "num_warmup_steps": int(0.01 * self.training_steps),
            "min_lr_rate": 0.1,
        }

    @property
    def max_grad_norm(self) -> float:
        return 1.0

    @property
    def hf_training_args(self) -> dict[str, Any]:
        return {}

    @property
    def fsdp_layers_to_wrap(self) -> list[str]:
        return ["GPTNeoXLayer"]

    @property
    def vocab_size(self) -> int:
        return 50304

    @property
    def sequence_length(self) -> int:
        return 2049
