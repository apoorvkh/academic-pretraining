import marimo

__generated_with = "0.8.18"
app = marimo.App(width="medium")


@app.cell
def __():
    from experiments.count_flops_sweep import CountFlopsSweep

    return (CountFlopsSweep,)


@app.cell
def __(CountFlopsSweep):
    results = CountFlopsSweep(search_space="sweep_configs/count_flops/all.json").results()
    return (results,)


@app.cell
def __():
    import math

    import polars as pl

    return math, pl


@app.cell
def __(math, pl, results):
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

    def _sci_not(x):
        exponent = int(math.log10(abs(x)))
        coefficient = x / 10**exponent
        return f"${coefficient:.1f} \\times 10^{{{exponent}}}$"

    _table = _results.to_pandas().to_latex(
        header=["Model", "Training FLOPs"],
        index=False,
        column_format="rc",
        float_format=_sci_not,
        escape=True,
    )

    _table
    return


if __name__ == "__main__":
    app.run()
