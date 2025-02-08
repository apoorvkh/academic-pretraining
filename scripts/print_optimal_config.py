import polars as pl
import tyro
from experiments.config import GpuT
from experiments.training_time_empirical_sweep import TrainingTimeEmpiricalSweep
from src.models import ModelT, get_model_class


def print_optimal_config(num_nodes: int, gpus_per_node: int, gpu_type: GpuT, model: ModelT) -> None:
    pl.Config(tbl_cols=20).__enter__()

    results = TrainingTimeEmpiricalSweep(
        search_space=dict(
            num_nodes=[num_nodes],
            gpus_per_node=[gpus_per_node],
            gpu_type=[gpu_type],
            model=[model],
            free_lunch=[False, True],
            activation_checkpointing=[False, True],
            sharding=["", "zero_1", "zero_2", "zero_3", "fsdp_shard_grad_op", "fsdp_full_shard"],
            offloading=[False, True],
        )
    ).results()

    batch_size = get_model_class(model_type=model).batch_size

    min_training_time = (
        results.filter(pl.col("training_days").is_not_null())
        .sort("training_days")
        .head(1)
        .with_columns(grad_acc_steps=(batch_size // (pl.col("micro_batch_size") * pl.col("gpus_per_node"))))
        .select(
            [
                "num_nodes",
                "gpus_per_node",
                "gpu_type",
                "model",
                "free_lunch",
                "activation_checkpointing",
                "sharding",
                "offloading",
                "micro_batch_size",
                "grad_acc_steps",
                "training_days",
            ]
        )
    )

    print(min_training_time)


if __name__ == "__main__":
    tyro.cli(print_optimal_config)
