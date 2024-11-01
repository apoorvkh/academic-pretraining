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
def __(mo):
    mo.md(r"""# Pythia""")
    return


@app.cell
def __(TrainingTimeEmpiricalSweep, process_training_time_results):
    _optimized = TrainingTimeEmpiricalSweep(
        search_space="sweep_configs/training_time_empirical/pythia_optimized.json"
    ).results()

    free_lunch_results = process_training_time_results(
        results=_optimized,
        free_lunch_only=True,
    )

    memory_saving_results = process_training_time_results(results=_optimized, mem_saving_only=True, select_min=True)
    return free_lunch_results, memory_saving_results


@app.cell
def __(free_lunch_results, memory_saving_results):
    _columns = ["num_nodes", "gpus_per_node", "gpu_type", "model", "training_days"]

    _free_lunch = free_lunch_results.select(_columns).rename({"training_days": "training_days_free_lunch"})
    _memory_saving = memory_saving_results.select(_columns).rename({"training_days": "training_days_memory_saving"})

    _join_kwargs = {
        "on": ["num_nodes", "gpus_per_node", "gpu_type", "model"],
        "how": "inner",
        "coalesce": True,
    }
    merged_results = _free_lunch.join(_memory_saving, **_join_kwargs)
    return (merged_results,)


@app.cell
def __(merged_results, pl):
    normalized_results = (
        merged_results.filter(
            pl.col("training_days_free_lunch").is_not_null() & pl.col("training_days_memory_saving").is_not_null()
        )
        .with_columns(
            (
                (pl.col("training_days_free_lunch") - pl.col("training_days_memory_saving"))
                / pl.col("training_days_free_lunch")
            ).alias("memory_saving_percent_gains")
        )
        .drop(["training_days_free_lunch", "training_days_memory_saving"])
    )

    normalized_results = normalized_results.with_columns(
        pl.col("gpu_type").replace({"geforce3090": "RTX 3090", "a6000": "A6000", "a100": "A100", "h100": "H100"}),
        pl.col("model").replace(
            {
                "pythia-160m": "160M",
                "pythia-410m": "410M",
                "pythia-1b": "1B",
                "pythia-2.8b": "2.8B",
                "pythia-6.9b": "6.9B",
            },
        ),
    )
    return (normalized_results,)


@app.cell
def __(alt, normalized_results):
    _chart1 = (
        alt.Chart(normalized_results)
        .transform_aggregate(
            mean_memory_saving="mean(memory_saving_percent_gains)",
            groupby=["gpus_per_node", "gpu_type"],
        )
        .mark_rect(stroke="black", strokeWidth=1)
        .encode(
            alt.X("gpus_per_node:O", title="Number of GPUs", axis=alt.Axis(labelAngle=0)),
            alt.Y(
                "gpu_type",
                title="",
                scale=alt.Scale(domain=["RTX 3090", "A6000", "A100", "H100"], reverse=True),
            ),
            alt.Color("mean_memory_saving:Q", legend=None, title="", scale=alt.Scale(domain=[-1, 1])),
        )
        .properties(
            width=600,
            height=600,
        )
    )

    _chart1 += _chart1.mark_text(baseline="middle", fontSize=45).encode(
        alt.X("gpus_per_node:O"),
        alt.Y("gpu_type:O"),
        text=alt.Text("mean_memory_saving:Q", format=".0%"),
        color=alt.condition(alt.datum["mean_memory_saving"] < 0, alt.value("black"), alt.value("white")),
    )

    _chart2 = (
        alt.Chart(
            normalized_results,
        )
        .transform_filter(alt.datum.gpus_per_node > 1)
        .transform_aggregate(mean_memory_saving="mean(memory_saving_percent_gains)", groupby=["gpu_type", "model"])
        .mark_rect(stroke="black", strokeWidth=1)
        .encode(
            alt.X(
                "model",
                title="Model Size (Pythia)",
                axis=alt.Axis(labelAngle=0),
                scale=alt.Scale(domain=["160M", "410M", "1B", "2.8B"]),
            ),
            alt.Y(
                "gpu_type",
                title="",
                scale=alt.Scale(domain=["RTX 3090", "A6000", "A100", "H100"], reverse=True),
            ),
            alt.Color("mean_memory_saving:Q", title="", legend=None, scale=alt.Scale(domain=[-1, 1])),
        )
        .properties(
            width=600,
            height=600,
        )
    )

    _chart2 += _chart2.mark_text(baseline="middle", fontSize=45).encode(
        alt.X("model"),
        alt.Y("gpu_type"),
        text=alt.Text("mean_memory_saving:Q", format=".0%"),
        color=alt.value("white"),
    )

    _chart = (
        (_chart1 & _chart2)
        .properties(
            title=["Speedups from Optimal", "Memory-Saving Methods"],
        )
        .configure_view(fill="lightgray")
        .configure_axis(labelFontSize=32, titleFontSize=32)
        .configure_title(fontSize=44, anchor="middle", dx=80)
    )

    _chart.save("artifacts/plots/optimal-memory-saving-speedups_pythia.pdf")

    _chart.interactive()
    return


@app.cell
def __(mo):
    mo.md(r"""# All models""")
    return


@app.cell
def __(TrainingTimeEmpiricalSweep, process_training_time_results):
    _optimized = TrainingTimeEmpiricalSweep(
        search_space="sweep_configs/training_time_empirical/all_optimized.json"
    ).results()

    all_free_lunch_results = process_training_time_results(
        results=_optimized,
        free_lunch_only=True,
    )

    all_memory_saving_results = process_training_time_results(results=_optimized, mem_saving_only=True, select_min=True)
    return all_free_lunch_results, all_memory_saving_results


@app.cell
def __(all_free_lunch_results, all_memory_saving_results, pl):
    _columns = ["num_nodes", "gpus_per_node", "gpu_type", "model", "training_days"]

    _free_lunch = all_free_lunch_results.select(_columns).rename({"training_days": "training_days_free_lunch"})
    _memory_saving = all_memory_saving_results.select(_columns).rename({"training_days": "training_days_memory_saving"})

    _join_kwargs = {
        "on": ["num_nodes", "gpus_per_node", "gpu_type", "model"],
        "how": "inner",
        "coalesce": True,
    }
    _merged_results = _free_lunch.join(_memory_saving, **_join_kwargs)

    all_normalized_results = (
        _merged_results.filter(
            pl.col("training_days_free_lunch").is_not_null() & pl.col("training_days_memory_saving").is_not_null()
        )
        .with_columns(
            (
                (pl.col("training_days_free_lunch") - pl.col("training_days_memory_saving"))
                / pl.col("training_days_free_lunch")
            ).alias("memory_saving_percent_gains")
        )
        .drop(["training_days_free_lunch", "training_days_memory_saving"])
    )

    all_normalized_results = all_normalized_results.with_columns(
        pl.col("gpu_type").replace({"geforce3090": "RTX 3090", "a6000": "A6000", "a100": "A100", "h100": "H100"}),
        pl.col("model").replace(
            {
                "pythia-160m": "(160M)",
                "pythia-410m": "(410M)",
                "pythia-1b": "(1B)",
                "pythia-2.8b": "(2.8B)",
                "pythia-6.9b": "(6.9B)",
                "roberta": "RoBERTa",
                "mamba": "Mamba",
                "convnext-xlarge-22k": "ConvNeXt",
                "vit": "ViT",
            },
        ),
        pl.col("gpus_per_node")
        .cast(str)
        .replace(
            {
                "1": "1 GPU",
                "2": "2 GPUs",
                "4": "4 GPUs",
                "8": "8 GPUs",
            }
        ),
    )
    return (all_normalized_results,)


@app.cell
def __(all_normalized_results, alt):
    _chart = (
        alt.Chart(
            all_normalized_results,
        )
        .mark_rect(stroke="black", strokeWidth=1)
        .encode(
            alt.X(
                "model",
                title="",
                axis=alt.Axis(labelAngle=0),
                scale=alt.Scale(
                    domain=[
                        "(160M)",
                        "(410M)",
                        "(1B)",
                        "(2.8B)",
                        "(6.9B)",
                        "RoBERTa",
                        "Mamba",
                        "ConvNeXt",
                        "ViT",
                    ]
                ),
            ),
            alt.Y(
                "gpu_type",
                title="",
                scale=alt.Scale(domain=["RTX 3090", "A6000", "A100", "H100"], reverse=True),
            ),
            alt.Color("memory_saving_percent_gains:Q", title="", legend=None, scale=alt.Scale(domain=[-1, 1])),
        )
        .properties(
            width=1200,
            height=600,
        )
    )

    _chart += _chart.mark_text(baseline="middle", fontSize=40).encode(
        alt.X("model"),
        alt.Y("gpu_type"),
        text=alt.Text("memory_saving_percent_gains:Q", format=".0%"),
        color=alt.condition(alt.datum["memory_saving_percent_gains"] < 0, alt.value("black"), alt.value("white")),
    )

    _chart = _chart.facet(row=alt.Row("gpus_per_node", title=""), columns=1)

    _chart = (
        _chart.properties(
            title="Speedups from Optimal Memory-Saving Methods",
        )
        .configure_view(fill="lightgray")
        .configure_axis(labelFontSize=32, titleFontSize=32)
        .configure_header(labelFontSize=36, labelAngle=0, labelOrient="top")
        .configure_title(fontSize=44, anchor="middle")
    )

    _chart.save("artifacts/plots/optimal-memory-saving-speedups_all.pdf")

    _chart.interactive()
    return


if __name__ == "__main__":
    app.run()
