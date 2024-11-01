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
    naive_results = process_training_time_results(
        results=TrainingTimeEmpiricalSweep(
            search_space="experiments/sweep_configs/training_time_empirical/pythia_naive.json"
        ).results()
    )

    _optimized = TrainingTimeEmpiricalSweep(
        search_space="experiments/sweep_configs/training_time_empirical/pythia_optimized.json"
    ).results()

    free_lunch_results = process_training_time_results(
        results=_optimized,
        free_lunch_only=True,
    )

    memory_saving_results = process_training_time_results(
        results=_optimized,
        mem_saving_only=True,
        select_min=True,
    )

    optimized_results = process_training_time_results(
        results=_optimized,
        select_min=True,
    )
    return (
        free_lunch_results,
        memory_saving_results,
        naive_results,
        optimized_results,
    )


@app.cell
def __(
    free_lunch_results,
    memory_saving_results,
    naive_results,
    optimized_results,
    pl,
):
    def _preprocess(df, label):
        return df.with_columns(
            pl.lit(label).alias("method"), pl.col("training_days").is_not_null().alias("fits_in_memory")
        ).select(["num_nodes", "gpus_per_node", "gpu_type", "model", "method", "fits_in_memory"])

    _naive = _preprocess(naive_results, "naive")
    _free_lunch = _preprocess(free_lunch_results, "free_lunch")
    _memory_saving = _preprocess(memory_saving_results, "memory_saving")
    _optimized = _preprocess(optimized_results, "optimized")

    merged_results = pl.concat([_naive, _free_lunch, _memory_saving, _optimized]).filter(
        pl.col("fits_in_memory") == True
    )

    merged_results = merged_results.with_columns(
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
        pl.col("method").replace(
            {
                "naive": "Naive",
                "free_lunch": "Free Lunch",
                "memory_saving": "Memory Saving",
                "optimized": "Optimized",
            }
        ),
    )
    return (merged_results,)


@app.cell
def __(alt, pl):
    def build_chart(_results, _methods):
        _results = _results.filter(pl.col("method").is_in(_methods))
        return (
            alt.Chart(_results, title="Feasibility of Empirical Training Methods")
            .mark_circle()
            .encode(
                x=alt.X(
                    "model",
                    title="Model Size",
                    axis=alt.Axis(grid=False, labelAngle=0),
                    scale=alt.Scale(
                        domain=[
                            "160M",
                            "410M",
                            "1B",
                            "2.8B",
                            "6.9B",
                        ],
                    ),
                ),
                y=alt.Y(
                    "gpu_type",
                    title="",
                    axis=alt.Axis(grid=False),
                    scale=alt.Scale(domain=["RTX 3090", "A6000", "A100", "H100"], reverse=True),
                ),
                color=alt.Color(
                    "method",
                    title="",
                    scale=alt.Scale(domain=_methods),
                    sort=_methods,
                ),
                xOffset=alt.XOffset("method", scale=alt.Scale(domain=_methods), sort=_methods),
            )
        )

    return (build_chart,)


@app.cell
def __(build_chart, merged_results, pl):
    _results = merged_results.filter((pl.col("num_nodes") == 1) & (pl.col("gpus_per_node") == 2)).drop(["num_nodes"])

    _chart = build_chart(_results, ["Naive", "Optimized"])

    _chart = (
        _chart.configure_circle(size=1500)
        .configure_axis(labelFontSize=25, titleFontSize=25)
        .configure_legend(labelFontSize=25, titleFontSize=25, symbolSize=400, orient="top", labelLimit=200)
        .properties(width=600, height=250)
        .configure_title(fontSize=30)
        .configure_range(category=["#f58518", "#e45756"])
    )

    _chart.save("artifacts/plots/feasibility_pythia-2gpu.pdf")

    _chart
    return


@app.cell
def __(mo):
    mo.md(r"""# All models""")
    return


@app.cell
def __(TrainingTimeEmpiricalSweep, process_training_time_results):
    all_naive_results = process_training_time_results(
        results=TrainingTimeEmpiricalSweep(
            search_space="experiments/sweep_configs/training_time_empirical/all_naive.json"
        ).results()
    )

    _optimized = TrainingTimeEmpiricalSweep(
        search_space="experiments/sweep_configs/training_time_empirical/all_optimized.json"
    ).results()

    all_free_lunch_results = process_training_time_results(
        results=_optimized,
        free_lunch_only=True,
    )

    all_optimized_results = process_training_time_results(
        results=_optimized,
        select_min=True,
    )
    return all_free_lunch_results, all_naive_results, all_optimized_results


@app.cell
def __(
    all_free_lunch_results,
    all_naive_results,
    all_optimized_results,
    pl,
):
    def _preprocess(df, label):
        return df.with_columns(
            pl.lit(label).alias("method"), pl.col("training_days").is_not_null().alias("fits_in_memory")
        ).select(["num_nodes", "gpus_per_node", "gpu_type", "model", "method", "fits_in_memory"])

    _naive = _preprocess(all_naive_results, "naive")
    _free_lunch = _preprocess(all_free_lunch_results, "free_lunch")
    _optimized = _preprocess(all_optimized_results, "optimized")

    all_merged_results = pl.concat([_naive, _free_lunch, _optimized]).filter(pl.col("fits_in_memory") == True)

    all_merged_results = all_merged_results.with_columns(
        pl.col("gpu_type").replace({"geforce3090": "RTX 3090", "a6000": "A6000", "a100": "A100", "h100": "H100"}),
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
            },
        ),
        pl.col("method").replace(
            {
                "naive": "Naive",
                "free_lunch": "Free Lunch",
                "optimized": "Optimized",
            }
        ),
    )
    return (all_merged_results,)


@app.cell
def __(all_merged_results):
    all_merged_results
    return


@app.cell
def __(alt, pl):
    def all_build_chart(_results, _methods):
        _results = _results.filter(pl.col("method").is_in(_methods))
        return (
            alt.Chart(_results, title="Feasibility of Empirical Training Methods")
            .mark_circle()
            .encode(
                x=alt.X(
                    "model",
                    title="Model Size",
                    axis=alt.Axis(grid=False, labelAngle=0),
                    scale=alt.Scale(
                        domain=[
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
                    ),
                ),
                y=alt.Y(
                    "gpu_type",
                    title="",
                    axis=alt.Axis(grid=False),
                    scale=alt.Scale(domain=["RTX 3090", "A6000", "A100", "H100"]),
                ),
                color=alt.Color(
                    "method",
                    title="",
                    scale=alt.Scale(domain=_methods),
                    sort=_methods,
                ),
                xOffset=alt.XOffset("method", scale=alt.Scale(domain=_methods), sort=_methods),
                row=alt.Row("gpus_per_node", title="", header=alt.Header(labels=False), sort=["1", "2+"]),
            )
        )

    return (all_build_chart,)


@app.cell
def __(all_build_chart, all_merged_results, pl):
    _results = (
        all_merged_results.filter((pl.col("num_nodes") == 1) & (pl.col("gpus_per_node").is_in([1, 2])))
        .drop(["num_nodes"])
        .with_columns(pl.col("gpus_per_node").cast(str).replace({"1": "1", "2": "2+"}))
    )

    _chart = all_build_chart(_results, ["Naive", "Free Lunch", "Optimized"])

    _chart = (
        _chart.configure_circle(size=1500)
        .configure_axis(labelFontSize=25, titleFontSize=25)
        .configure_legend(labelFontSize=25, titleFontSize=25, symbolSize=400, orient="top", labelLimit=200)
        .properties(width=1600, height=250)
        .configure_title(fontSize=30, anchor="middle")
    )

    _chart.save("artifacts/plots/feasibility_all.pdf")

    _chart.interactive()
    return


if __name__ == "__main__":
    app.run()
