import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from experiments.count_flops_sweep import CountFlopsSweep

    return (CountFlopsSweep,)


@app.cell
def _(CountFlopsSweep):
    results = CountFlopsSweep(search_space="experiments/sweep_configs/count_flops/all.json").results()
    return (results,)


@app.cell
def _():
    import math

    import polars as pl

    return math, pl


@app.cell
def _(pl, results):
    _results = results.select(["model", "training_flops"])

    _results = _results.with_columns(
        pl.col("model").replace(
            {
                "mamba": "Mamba",
                "roberta": "RoBERTa",
                "pythia-160m": "Pythia (160M)",
                "pythia-410m": "Pythia (410M)",
                "pythia-1b": "Pythia (1B)",
                "pythia-2.8b": "Pythia (2.8B)",
                "pythia-6.9b": "Pythia (6.9B)",
                "convnext-xlarge-22k": "ConvNeXt",
                "vit": "ViT",
            }
        )
    )

    _latex_table = (
        _results.style.cols_label({"model": "Model", "training_flops": "Training FLOPs"})
        .cols_align(align="right", columns=["model"])
        .cols_align(align="center", columns=["training_flops"])
        .fmt_scientific(columns="training_flops", decimals=1, exp_style="x10n")
        .as_latex()
    )

    _latex_table = _latex_table[_latex_table.find("\\begin{tabular*}") : _latex_table.find("\n\\end{table}")]
    _latex_table = _latex_table.replace(
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}", "\\begin{tabular}{"
    ).replace("\\end{tabular*}", "\\end{tabular}")

    _latex_table
    return


if __name__ == "__main__":
    app.run()
