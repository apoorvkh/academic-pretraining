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
def __():
    original_text = "Originally reported"
    ours_text = "Optimal settings\n(on 8 x 80GB A100s)"
    return original_text, ours_text


@app.cell
def __(original_text, ours_text, pl, results):
    # Process optimal GPU-days

    _results = results.filter(pl.col("gpu_type") == "a100", pl.col("num_nodes") == 1, pl.col("gpus_per_node") == 8)

    _results = _results.with_columns((pl.col("training_days") * 8).alias(ours_text).cast(pl.Int64)).select(
        "model", ours_text
    )

    # Process original GPU-days

    _original = (
        pl.DataFrame(
            {
                "model": [
                    "pythia-160m",
                    "pythia-410m",
                    "pythia-1b",
                    "pythia-2.8b",
                    "pythia-6.9b",
                    "roberta",
                    "convnext-xlarge-22k",
                    "vit",
                ],
                "original_num_gpus": [32, 32, 64, 64, 128, 1024, 128, 8],
                "original_days": [1, 3, 3, 9, 10, 1, 3, 30],
            }
        )
        .with_columns((pl.col("original_num_gpus") * pl.col("original_days")).alias(original_text))
        .select("model", original_text)
    )

    gpu_days_df = _results.join(_original, on="model", how="inner").unpivot(
        index="model", variable_name="method", value_name="GPU-days"
    )

    gpu_days_df = gpu_days_df.with_columns(
        pl.col("model").replace(
            {
                "pythia-160m": "160M",
                "pythia-410m": "410M",
                "pythia-1b": "1B",
                "pythia-2.8b": "2.8B",
                "pythia-6.9b": "6.9B",
                "roberta": "RoBERTa",
                "convnext-xlarge-22k": "ConvNeXt",
                "vit": "ViT",
            }
        )
    )
    return (gpu_days_df,)


@app.cell
def __(alt, gpu_days_df, original_text, ours_text):
    # Note that: ⎯ Pythia ⎯ has different spacing in generated image

    _chart = (
        alt.Chart(gpu_days_df, title="Pre-training Resources")
        .mark_bar()
        .encode(
            x=alt.X(
                "model:N",
                axis=alt.Axis(labelAngle=0),
                scale=alt.Scale(
                    domain=["160M", "410M", "1B", "2.8B", "6.9B", "RoBERTa", "ConvNeXt", "ViT"],
                ),
                title="⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯   Pythia   ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯",
            ),
            y=alt.Y("GPU-days:Q", title="GPU–days", axis=alt.Axis(tickCount=7)),
            color=alt.Color(
                "method:N",
                legend=alt.Legend(orient="none", title="", padding=5, labelExpr="split(datum.label, '\\n')"),
                scale=alt.Scale(domain=[original_text, ours_text]),
            ),
            xOffset=alt.XOffset("method:N", scale=alt.Scale(domain=[original_text, ours_text])),
        )
    )

    _chart = (
        _chart.properties(
            width=850,
            height=400,
        )
        .configure_title(fontSize=30)
        .configure_axis(labelFontSize=22, titleFontSize=28)
        .configure_legend(labelFontSize=32, labelLimit=400)
        .configure_axisX(titleAlign="left", titleX=20)
    )

    _chart.save("artifacts/plots/gpu-days.pdf")

    _chart.interactive()
    return


if __name__ == "__main__":
    app.run()
