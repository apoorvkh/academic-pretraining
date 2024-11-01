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
def __(mo):
    mo.md("""# Loading results""")
    return


@app.cell
def __():
    from experiments.training_time_empirical_sweep import TrainingTimeEmpiricalSweep
    from scripts.plotting import process_training_time_results

    return TrainingTimeEmpiricalSweep, process_training_time_results


@app.cell
def __(TrainingTimeEmpiricalSweep, process_training_time_results):
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
    )
    return free_lunch_results, memory_saving_results


@app.cell
def __(free_lunch_results, memory_saving_results, pl):
    _group_columns = ["num_nodes", "gpus_per_node", "gpu_type", "model"]

    _results_free_lunch = (
        free_lunch_results.drop_nulls("training_days")
        .group_by(_group_columns)
        .agg(pl.col("training_days").min())
        .rename({"training_days": "Free Lunch"})
    )

    _grouped_results = memory_saving_results.drop_nulls("training_days").group_by(_group_columns)

    _results_optimal = _grouped_results.agg(pl.col("training_days").min()).rename({"training_days": "Optimal"})
    _results_median = _grouped_results.agg(pl.col("training_days").median()).rename({"training_days": "Median"})
    _results_worst = _grouped_results.agg(pl.col("training_days").max()).rename({"training_days": "Worst"})

    aggregated_results = (
        _results_optimal.join(_results_median, on=_group_columns, how="inner")
        .join(_results_worst, on=_group_columns, how="inner")
        .join(_results_free_lunch, on=_group_columns, how="left")
    )
    return (aggregated_results,)


@app.cell
def __(aggregated_results):
    aggregated_results
    return


@app.cell
def __(aggregated_results, pl):
    normalized_results = aggregated_results.with_columns(
        (pl.col("Worst") / pl.col("Optimal")).alias("Optimal"),
        (pl.col("Worst") / pl.col("Median")).alias("Median"),
        (pl.col("Worst") / pl.col("Worst")).alias("Worst"),
        (pl.col("Worst") / pl.col("Free Lunch")).alias("Free Lunch"),
    ).unpivot(
        index=["num_nodes", "gpus_per_node", "gpu_type", "model"],
        variable_name="statistic",
        value_name="speedup",
    )
    return (normalized_results,)


@app.cell
def __(alt, normalized_results):
    _base = alt.Chart(normalized_results, title="Comparing Memory-Saving Methods").transform_filter(
        alt.datum.gpus_per_node > 1
    )
    _chart = _base.mark_bar().encode(
        alt.X(
            "statistic",
            title="",
            axis=alt.Axis(labelAngle=0, labelLimit=200),
            scale=alt.Scale(domain=["Optimal", "Median", "Worst", "Free Lunch"]),
        ),
        alt.Y(
            "mean(speedup)",
            title="Speedup",
            axis=alt.Axis(tickCount=5, labelExpr=alt.datum.value + "x"),
        ),
        alt.Color("statistic", legend=None),
    )

    _chart += _chart.mark_text(align="center", baseline="middle", dx=35, dy=-15, fontSize=32).encode(
        text=alt.Text("mean(speedup):Q", format=".2f")
    )

    _chart += _base.mark_errorbar(extent="ci").encode(
        alt.X("statistic", title=""),
        alt.Y(
            "speedup",
            title="",
        ),
        strokeWidth=alt.value(3),
    )

    _chart += alt.Chart().mark_rule(strokeDash=[15, 6], strokeWidth=5, color="red").encode(x=alt.value(int(600 * 0.75)))

    _chart = (
        _chart.properties(
            width=600,
            height=600,
        )
        .configure_axis(labelFontSize=33, titleFontSize=38)
        .configure_title(fontSize=40)
    )

    _chart.save("artifacts/plots/comparing-memory-saving.pdf")

    _chart.interactive()
    return


if __name__ == "__main__":
    app.run()
