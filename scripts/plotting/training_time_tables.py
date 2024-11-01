import marimo

__generated_with = "0.8.18"
app = marimo.App(width="medium")


@app.cell
def __():
    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, mo, pl


@app.cell
def __():
    from experiments.training_time_empirical_sweep import TrainingTimeEmpiricalSweep
    from scripts.plotting import process_training_time_results

    return TrainingTimeEmpiricalSweep, process_training_time_results


@app.cell
def __(pl):
    def training_times_to_table(results):
        _results = (
            results.with_columns(
                pl.col("model").replace(
                    {
                        "pythia-160m": "Pythia (160M)",
                        "pythia-410m": "Pythia (410M)",
                        "pythia-1b": "Pythia (1B)",
                        "pythia-2.8b": "Pythia (2.8B)",
                        "pythia-6.9b": "Pythia (6.9B)",
                        "roberta": "RoBERTa",
                        "mamba": "Mamba",
                        "convnext-xlarge-22k": "ConvNeXt",
                        "vit": "ViT",
                    }
                ),
                pl.col("gpu_type").replace(
                    {
                        "geforce3090": "RTX 3090 (24 GB)",
                        "a6000": "A6000 (48 GB)",
                        "a100": "A100 (80 GB)",
                        "h100": "H100 (80 GB)",
                    }
                ),
            )
            .filter(pl.col("num_nodes") == 1)
            .rename({"gpus_per_node": "num_gpus"})
        )

        _table = _results.select(["num_gpus", "gpu_type", "model", "training_days"])
        _table = _table.with_columns(pl.col("training_days").round(0))
        _table_pd = _table.to_pandas().pivot_table(
            index="model", columns=("gpu_type", "num_gpus"), values="training_days"
        )

        _table_pd = _table_pd.reindex(
            columns=[
                "RTX 3090 (24 GB)",
                "A6000 (48 GB)",
                "A100 (80 GB)",
                "H100 (80 GB)",
            ],
            level="gpu_type",
        )

        _table_pd = _table_pd.reindex(
            [
                "Pythia (160M)",
                "Pythia (410M)",
                "Pythia (1B)",
                "Pythia (2.8B)",
                "Pythia (6.9B)",
                "RoBERTa",
                "Mamba",
                "ConvNeXt",
                "ViT",
            ]
        )

        _latex_table = _table_pd.to_latex(float_format="%d", na_rep="---", escape=True)

        return _latex_table

    return (training_times_to_table,)


@app.cell
def __(TrainingTimeEmpiricalSweep, process_training_time_results):
    optimal_results = process_training_time_results(
        results=TrainingTimeEmpiricalSweep(
            search_space="experiments/sweep_configs/training_time_empirical/all_optimized.json"
        ).results(),
        select_min=True,
    )
    return (optimal_results,)


@app.cell
def __(optimal_results, training_times_to_table):
    print(training_times_to_table(optimal_results))
    return


@app.cell
def __(TrainingTimeEmpiricalSweep, process_training_time_results):
    naive_results = process_training_time_results(
        results=TrainingTimeEmpiricalSweep(
            search_space="experiments/sweep_configs/training_time_empirical/all_naive.json"
        ).results()
    )
    return (naive_results,)


@app.cell
def __(naive_results, training_times_to_table):
    print(training_times_to_table(naive_results))
    return


@app.cell
def __(mo):
    mo.md(r"""# Optimal Settings""")
    return


@app.cell
def __():
    from src.models import get_model_class

    return (get_model_class,)


@app.cell
def __(get_model_class):
    batch_sizes = {
        model: get_model_class(model_name=model).batch_size
        for model in [
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-2.8b",
            "pythia-6.9b",
            "roberta",
            "mamba",
            "convnext-xlarge-22k",
            "vit",
        ]
    }
    return (batch_sizes,)


@app.cell
def __(batch_sizes, pl, results):
    _table = (
        results.with_columns(
            pl.col("model")
            .replace(
                {
                    "pythia-160m": "Pythia (160M)",
                    "pythia-410m": "Pythia (410M)",
                    "pythia-1b": "Pythia (1B)",
                    "pythia-2.8b": "Pythia (2.8B)",
                    "pythia-6.9b": "Pythia (6.9B)",
                    "roberta": "RoBERTa",
                    "mamba": "Mamba",
                    "convnext-xlarge-22k": "ConvNeXt",
                    "vit": "ViT",
                }
            )
            .cast(
                pl.Enum(
                    [
                        "Pythia (160M)",
                        "Pythia (410M)",
                        "Pythia (1B)",
                        "Pythia (2.8B)",
                        "Pythia (6.9B)",
                        "RoBERTa",
                        "Mamba",
                        "ConvNeXt",
                        "ViT",
                    ]
                )
            ),
            pl.col("gpu_type")
            .replace(
                {
                    "geforce3090": "RTX 3090",
                    "a6000": "A6000",
                    "a100": "A100",
                    "h100": "H100",
                }
            )
            .cast(pl.Enum(["RTX 3090", "A6000", "A100", "H100"])),
            batch_size=pl.col("model").replace(batch_sizes).cast(int),
        )
        .filter(pl.col("num_nodes") == 1)
        .with_columns(grad_acc_steps=(pl.col("batch_size") // (pl.col("micro_batch_size") * pl.col("gpus_per_node"))))
        .drop(
            [
                "trial",
                "num_nodes",
                "benchmarking_steps",
                "batch_size",
                "max_micro_batch_size",
                "step_time",
                "compile_disabled",
                "training_days",
            ]
        )
        .sort(["model", "gpu_type", "gpus_per_node"])
        .select(
            [
                "model",
                "gpu_type",
                "gpus_per_node",
                "free_lunch",
                "activation_checkpointing",
                "sharding",
                "offloading",
                "micro_batch_size",
                "grad_acc_steps",
            ]
        )
        .rename(
            {
                "gpus_per_node": "# GPUs",
                "gpu_type": "GPU Type",
                "model": "Model",
                "free_lunch": "Free Lunch",
                "activation_checkpointing": "Checkpointing",
                "sharding": "Sharding",
                "offloading": "Offloading",
                "micro_batch_size": "MBS",
                "grad_acc_steps": "GAS",
            }
        )
    )

    def _bool_to_mark(x):
        return "\\cmark" if x else "\\xmark"

    _sharding = {
        "": "---",
        "zero\\_1": "Zero (1)",
        "zero\\_2": "Zero (2)",
        "zero\\_3": "Zero (3)",
        "fsdp\\_shard\\_grad\\_op": "FSDP (2)",
        "fsdp\\_full\\_shard": "FSDP (3)",
    }.get

    _table.to_pandas().to_latex(
        index=False,
        escape=True,
        longtable=True,
        column_format="llc|cccccc",
        formatters={
            "Free Lunch": _bool_to_mark,
            "Checkpointing": _bool_to_mark,
            "Sharding": _sharding,
            "Offloading": _bool_to_mark,
        },
    )
    return


if __name__ == "__main__":
    app.run()
