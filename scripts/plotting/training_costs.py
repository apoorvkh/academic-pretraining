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
def __(TrainingTimeEmpiricalSweep, process_training_time_results):
    results = process_training_time_results(
        results=TrainingTimeEmpiricalSweep(
            search_space="experiments/sweep_configs/training_time_empirical/all_optimized.json"
        ).results(),
        select_min=True,
    )
    return (results,)


@app.cell
def __(pl, results):
    _results = results.drop(
        [
            "free_lunch",
            "activation_checkpointing",
            "sharding",
            "offloading",
            "benchmarking_steps",
            "max_micro_batch_size",
            "micro_batch_size",
            "step_time",
            "compile_disabled",
        ]
    )

    _results = _results.with_columns((pl.col("num_nodes") * pl.col("gpus_per_node")).alias("num_gpus")).drop(
        "num_nodes", "gpus_per_node"
    )

    _base_machine_costs = {1: 1000, 2: 1500, 4: 7500, 8: 10500}
    _gpu_costs = {"geforce3090": 1300, "a6000": 4800, "a100": 19000, "h100": 30000}

    df = _results.with_columns(
        (
            pl.col("num_gpus").replace(_base_machine_costs)
            + (pl.col("num_gpus") * pl.col("gpu_type").replace(_gpu_costs).cast(pl.Int64))
        ).alias("cost")
    ).with_columns((pl.col("cost") * (pl.col("training_days") / (5 * 365))).alias("normalized_cost"))

    df = df.with_columns(
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
        pl.col("gpu_type").replace({"geforce3090": "RTX 3090", "a6000": "A6000", "a100": "A100", "h100": "H100"}),
    )
    return (df,)


@app.cell
def __(df):
    df
    return


@app.cell
def __(alt):
    def build_cost_chart(data):
        return (
            alt.Chart(data)
            .mark_circle()
            .encode(
                alt.X("cost", axis=alt.Axis(format="$.2~s", tickCount=20), title="Cost (USD)"),
                alt.Y("training_days", title="Training Time (days)").scale(type="log"),
                alt.Color(
                    "gpu_type",
                    title="GPU Type",
                    scale=alt.Scale(domain=["H100", "A100", "A6000", "RTX 3090"]),
                    legend=alt.Legend(symbolSize=400),
                ),
                alt.Size("num_gpus:O", title="Num. GPUs", scale=alt.Scale(range=[150, 1200])),
                alt.Column(
                    "model",
                    sort=[
                        "Pythia (160M)",
                        "Pythia (410M)",
                        "Pythia (1B)",
                        "Pythia (2.8B)",
                        "Pythia (6.9B)",
                        "RoBERTa",
                        "Mamba",
                        "ConvNeXt",
                        "ViT",
                    ],
                    header=alt.Header(labelFontSize=32),
                    title="",
                ),
            )
            .properties(width=400, height=400)
            .resolve_scale(x="independent", y="independent")
            .configure_title(fontSize=34)
            .configure_axis(
                titleFontSize=30,
                labelFontSize=32,
            )
            .configure_legend(
                titleFontSize=28,
                labelFontSize=28,
            )
        )

    return (build_cost_chart,)


@app.cell
def __(alt):
    def build_normalized_cost_chart(data):
        return (
            alt.Chart(data)
            .mark_circle()
            .encode(
                alt.X("normalized_cost", axis=alt.Axis(format="$.2~s", tickCount=20), title="Cost (USD)"),
                alt.Y("training_days", title="Training Time (days)").scale(type="log"),
                alt.Color(
                    "gpu_type",
                    title="GPU Type",
                    scale=alt.Scale(domain=["H100", "A100", "A6000", "RTX 3090"]),
                    legend=alt.Legend(symbolSize=400),
                ),
                alt.Size("num_gpus:O", title="Num. GPUs", scale=alt.Scale(range=[150, 1200])),
                alt.Column(
                    "model",
                    sort=[
                        "Pythia (160M)",
                        "Pythia (410M)",
                        "Pythia (1B)",
                        "Pythia (2.8B)",
                        "Pythia (6.9B)",
                        "RoBERTa",
                        "Mamba",
                        "ConvNeXt",
                        "ViT",
                    ],
                    header=alt.Header(labelFontSize=32),
                    title="",
                ),
            )
            .properties(width=400, height=400)
            .resolve_scale(x="independent", y="independent")
            .configure_title(fontSize=34)
            .configure_axis(
                titleFontSize=30,
                labelFontSize=32,
            )
            .configure_legend(
                titleFontSize=28,
                labelFontSize=28,
            )
        )

    return (build_normalized_cost_chart,)


@app.cell
def __(build_cost_chart, df, pl):
    _df = df.filter(
        pl.col("model").is_in(
            [
                "Pythia (160M)",
                "Pythia (410M)",
                "Pythia (1B)",
                "Pythia (2.8B)",
                "Pythia (6.9B)",
            ]
        )
    )
    _chart = build_cost_chart(_df)
    _chart.save("artifacts/plots/cost_pythia.pdf")
    _chart.interactive()
    return


@app.cell
def __(build_cost_chart, df, pl):
    _df = df.filter(
        pl.col("model").is_in(
            [
                "RoBERTa",
                "Mamba",
                "ConvNeXt",
                "ViT",
            ]
        )
    )
    _chart = build_cost_chart(_df)
    _chart.save("artifacts/plots/cost_other.pdf")
    _chart.interactive()
    return


@app.cell
def __(build_normalized_cost_chart, df, pl):
    _df = df.filter(
        pl.col("model").is_in(
            [
                "Pythia (160M)",
                "Pythia (410M)",
                "Pythia (1B)",
                "Pythia (2.8B)",
                "Pythia (6.9B)",
            ]
        )
    )
    _chart = build_normalized_cost_chart(_df)
    _chart.save("artifacts/plots/normaized_cost_pythia.pdf")
    _chart.interactive()
    return


@app.cell
def __(build_normalized_cost_chart, df, pl):
    _df = df.filter(
        pl.col("model").is_in(
            [
                "RoBERTa",
                "Mamba",
                "ConvNeXt",
                "ViT",
            ]
        )
    )
    _chart = build_normalized_cost_chart(_df)
    _chart.save("artifacts/plots/normaized_cost_other.pdf")
    _chart.interactive()
    return


@app.cell
def __(alt, df, pl):
    _df_1b = df.filter(pl.col("model") == "Pythia (1B)")

    _chart = (
        alt.Chart(_df_1b, title="Overall Hardware Cost")
        .mark_circle(clip=True)
        .encode(
            alt.X("cost", axis=alt.Axis(format="$.2~s", tickCount=20), title=""),
            alt.Y("training_days", scale=alt.Scale(domain=[0, 80]), title="Training Time (days)"),
            alt.Color(
                "gpu_type",
                title="GPU Type",
                scale=alt.Scale(domain=["H100", "A100", "A6000", "RTX 3090"]),
                legend=alt.Legend(symbolSize=400),
            ),
            alt.Size(
                "num_gpus:O",
                title="Num. GPUs",
                scale=alt.Scale(range=[150, 1200]),
            ),
        )
        .properties(width=400, height=400)
    ) & (
        alt.Chart(_df_1b, title="Experiment Cost")
        .mark_circle(clip=True)
        .encode(
            alt.X(
                "normalized_cost",
                scale=alt.Scale(domain=[300, 850]),
                axis=alt.Axis(format="$.2~s", tickCount=5),
                title="",
            ),
            alt.Y("training_days", scale=alt.Scale(domain=[0, 85]), title="Training Time (days)"),
            alt.Color(
                "gpu_type",
                title="GPU Type",
                scale=alt.Scale(domain=["H100", "A100", "A6000", "RTX 3090"]),
            ),
            alt.Size("num_gpus:O", title="Num. GPUs", scale=alt.Scale(range=[150, 1200])),
        )
        .properties(width=400, height=400)
    )

    _chart = (
        _chart.configure_title(fontSize=34)
        .configure_axis(
            titleFontSize=32,
            labelFontSize=32,
        )
        .configure_legend(
            titleFontSize=28,
            labelFontSize=28,
        )
    )

    _chart.save("artifacts/plots/cost_pythia-1b.pdf")

    _chart.interactive()
    return


if __name__ == "__main__":
    app.run()
