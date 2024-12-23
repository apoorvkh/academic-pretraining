import gc
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch._dynamo.config
import torch.cuda
import torch.nn as nn
from accelerate.utils import check_cuda_p2p_ib_support
from torch.utils.data import Dataset
from transformers import SchedulerType, Trainer, TrainingArguments
from transformers.trainer_utils import FSDPOption


@dataclass
class TrainingClass:
    num_training_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool = False
    bf16: bool = False
    fp16: bool = False
    tf32: bool = False
    compile: bool = False

    optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler_type: SchedulerType = SchedulerType.LINEAR
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)

    FsdpShardingT = Literal["no_shard", "shard_grad_op", "full_shard", "hybrid_shard_zero2", "hybrid_shard"]
    fsdp_sharding: FsdpShardingT = "no_shard"
    fsdp_layers_to_wrap: list[str] = field(default_factory=list)
    fsdp_offload: bool = False

    ZeroStageT = Literal["0", "1", "2", "3", "3++"]
    zero_stage: ZeroStageT = "0"
    zero_offload_optimizer: bool = False
    zero_offload_params: bool = False

    max_grad_norm: float = 1.0
    hf_training_args_overrides: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        return not (
            self.num_training_steps <= 0
            or self.micro_batch_size <= 0
            or self.gradient_accumulation_steps <= 0
            or (self.bf16 and self.fp16)
            or (self.fsdp_sharding != "no_shard" and self.zero_stage != "0")
            or (self.fsdp_offload and self.fsdp_sharding == "no_shard")
            or (self.zero_offload_optimizer and self.zero_stage == "0")
            or (self.zero_offload_params and self.zero_stage not in ["3", "3++"])
        )

    def build_trainer(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        hf_training_args_overrides: dict[str, Any] = {},
        hf_trainer_kwargs_overrides: dict[str, Any] = {},
    ) -> Trainer:
        gc.collect()
        torch.cuda.empty_cache()
        torch.compiler.reset()

        # 4000-series GPUs do not support P2P/IB GPU communications
        if check_cuda_p2p_ib_support() is False:
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_IB_DISABLE"] = "1"

        return self.trainer_cls(
            model=model,
            args=self.to_huggingface_args(**hf_training_args_overrides),
            train_dataset=train_dataset,
            **hf_trainer_kwargs_overrides,
        )

    @property
    def trainer_cls(self) -> type[Trainer]:
        # If using Deepspeed, use Deepspeed's Adam optimizer
        if self.zero_stage != "0" and self.optimizer in [torch.optim.Adam, torch.optim.AdamW]:
            return Trainer

        # TODO: For now PyTorch fused Adam optimizer seems to break often
        # if self.optimizer in [torch.optim.Adam, torch.optim.AdamW]
        # self.optimizer_kwargs["fused"] = True

        class CustomOptimizerTrainer(Trainer):
            @staticmethod
            def get_optimizer_cls_and_kwargs(
                args: TrainingArguments, model=None
            ) -> tuple[type[torch.optim.Optimizer], dict[str, Any]]:
                return self.optimizer, self.optimizer_kwargs

            # Can remove this after transformers==4.43.0
            def create_optimizer(self):
                trainer_get_optimizer_fn = Trainer.get_optimizer_cls_and_kwargs
                Trainer.get_optimizer_cls_and_kwargs = self.get_optimizer_cls_and_kwargs
                optimizer = super().create_optimizer()
                Trainer.get_optimizer_cls_and_kwargs = trainer_get_optimizer_fn
                return optimizer

        return CustomOptimizerTrainer

    def to_huggingface_args(self, **hf_training_args_overrides) -> TrainingArguments:
        return TrainingArguments(**self._to_huggingface_args_dict(**hf_training_args_overrides))

    def _to_huggingface_args_dict(self, **hf_training_args_overrides) -> dict:
        fsdp_options, fsdp_config = self._build_fsdp_config()
        ds_config = self._build_deepspeed_config()

        # DDPOptimizer + activation checkpointing not supported
        # [https://github.com/pytorch/pytorch/issues/104674]
        if self.gradient_checkpointing:
            torch._dynamo.config.optimize_ddp = False

        scheduler_warmup_steps = self.scheduler_kwargs.pop("num_warmup_steps", 0)

        return dict(
            max_steps=self.num_training_steps,
            per_device_train_batch_size=self.micro_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            lr_scheduler_type=self.scheduler_type.value,
            lr_scheduler_kwargs=self.scheduler_kwargs,
            warmup_steps=scheduler_warmup_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            bf16=self.bf16,
            fp16=self.fp16,
            tf32=self.tf32,
            fsdp=fsdp_options,
            fsdp_config=fsdp_config,
            deepspeed=ds_config,
            ddp_find_unused_parameters=False,
            torch_compile=self.compile,
            max_grad_norm=self.max_grad_norm,
            **self.hf_training_args_overrides,
            **hf_training_args_overrides,
        )

    def _build_fsdp_config(self) -> tuple[list[FSDPOption], dict[str, Any]] | tuple[str, None]:
        if self.fsdp_sharding == "no_shard":
            return "", None

        fsdp_options = [FSDPOption(self.fsdp_sharding), FSDPOption.AUTO_WRAP]
        if self.fsdp_offload:
            fsdp_options += [FSDPOption.OFFLOAD]
        fsdp_config = {
            "transformer_layer_cls_to_wrap": self.fsdp_layers_to_wrap,
        }
        return fsdp_options, fsdp_config

    def _build_deepspeed_config(self) -> dict | None:
        config = {
            "fp16": {
                "enabled": "auto",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
        }

        if self.optimizer in [torch.optim.Adam, torch.optim.AdamW]:
            config["optimizer"] = {
                "type": "Adam",
                "params": {
                    "lr": "auto",
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto",
                    "adam_w_mode": self.optimizer == torch.optim.AdamW,
                },
            }

        match self.zero_stage:
            case "0":
                return None
            case "1":
                config["zero_optimization"] = {"stage": 1}
            case "2":
                config["zero_optimization"] = {
                    "stage": 2,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True,
                }
            case "3" | "3++":
                config["zero_optimization"] = {
                    "stage": 3,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "sub_group_size": 1e9,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_gather_16bit_weights_on_model_save": True,
                }

                if self.zero_stage == "3++":
                    config["zero_optimization"].update(
                        zero_quantized_weights=True,
                        zero_hpz_partition_size=torch.cuda.device_count(),
                        zero_quantized_gradients=True,
                    )

        if self.zero_offload_optimizer:
            config["zero_optimization"]["offload_optimizer"] = {  # pyright: ignore [reportArgumentType]
                "device": "cpu",
                "pin_memory": True,
            }

        if self.zero_offload_params:
            config["zero_optimization"]["offload_param"] = {  # pyright: ignore [reportArgumentType]
                "device": "cpu",
                "pin_memory": True,
            }

        return config
