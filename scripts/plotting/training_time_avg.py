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
    from experiments.training_time_analytic_sweep import TrainingTimeAnalyticSweep
    from experiments.training_time_empirical_sweep import TrainingTimeEmpiricalSweep
    from scripts.plotting import process_training_time_results

    return (
        TrainingTimeAnalyticSweep,
        TrainingTimeEmpiricalSweep,
        process_training_time_results,
    )


@app.cell
def __(TrainingTimeAnalyticSweep):
    flops_results = TrainingTimeAnalyticSweep(search_space="experiments/sweep_configs/training_time_analytic/pythia.json").results()
    return (flops_results,)


@app.cell
def __(TrainingTimeEmpiricalSweep, process_training_time_results):
    naive_results = process_training_time_results(
        results=TrainingTimeEmpiricalSweep(
            search_space="experiments/sweep_configs/training_time_empirical/pythia_naive.json"
        ).results()
    )

    optimized_results = process_training_time_results(
        results=TrainingTimeEmpiricalSweep(
            search_space="experiments/sweep_configs/training_time_empirical/pythia_optimized.json"
        ).results(),
        select_min=True,
    )
    return naive_results, optimized_results


@app.cell
def __(flops_results, naive_results, optimized_results):
    _columns = ["num_nodes", "gpus_per_node", "gpu_type", "model", "training_days"]

    _flops = flops_results.select(_columns).rename({"training_days": "training_days_flops"})
    _naive = naive_results.select(_columns).rename({"training_days": "training_days_naive"})
    _optimized = optimized_results.select(_columns).rename({"training_days": "training_days_optimized"})

    _join_kwargs = {
        "on": ["num_nodes", "gpus_per_node", "gpu_type", "model"],
        "how": "inner",
        "coalesce": True,
    }
    merged_results = _flops.join(_naive, **_join_kwargs).join(_optimized, **_join_kwargs)
    return (merged_results,)


@app.cell
def __(merged_results, pl):
    normalized_results = (
        merged_results.filter(
            pl.col("training_days_flops").is_not_null()
            & pl.col("training_days_naive").is_not_null()
            & pl.col("training_days_optimized").is_not_null()
        )
        .with_columns(
            (pl.col("training_days_naive") / pl.col("training_days_flops")).alias("Analytic"),
            (pl.col("training_days_naive") / pl.col("training_days_naive")).alias("Naive"),
            (pl.col("training_days_naive") / pl.col("training_days_optimized")).alias("Optimized"),
        )
        .drop("training_days_flops", "training_days_naive", "training_days_optimized")
    )

    normalized_results = normalized_results.unpivot(
        index=["num_nodes", "gpus_per_node", "gpu_type", "model"],
        variable_name="method",
        value_name="speedup",
    )
    return (normalized_results,)


@app.cell
def __(normalized_results):
    normalized_results
    return


@app.cell
def __(alt, normalized_results):
    _chart = (
        alt.Chart(normalized_results, title="Relative Training Speed")
        .mark_bar()
        .encode(
            alt.X("method", title="", axis=alt.Axis(labelAngle=0)),
            alt.Y(
                "mean(speedup)",
                title="",
                axis=alt.Axis(tickCount=6, labelExpr=alt.datum.value + "x"),
            ),
            alt.Color("method", legend=None),
        )
    )

    _chart += _chart.mark_text(align="center", baseline="middle", dx=50, dy=-15, fontSize=40).encode(
        text=alt.Text("mean(speedup):Q", format=".2f")
    )

    _chart += (
        alt.Chart(normalized_results)
        .mark_errorbar(extent="ci")
        .encode(
            alt.X("method", title=""),
            alt.Y(
                "speedup",
                title="",
            ),
            strokeWidth=alt.value(3),
        )
    )

    _chart += alt.Chart().mark_rule(strokeDash=[15, 6], strokeWidth=5, color="red").encode(x=alt.value(600 // 3))

    _chart = (
        _chart.properties(
            width=600,
            height=600,
        )
        .configure_title(fontSize=40)
        .configure_axis(labelFontSize=36, titleFontSize=36)
    )

    _chart.save("artifacts/plots/avg-training-time-pythia.pdf")

    _chart.interactive()

    # could facet by gpus_per_node, model
    return


if __name__ == "__main__":
    app.run()
