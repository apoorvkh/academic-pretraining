import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import polars as pl
import tyro

from experiments import Sweep
from experiments.config import GpuT
from experiments.count_flops import CountFlopsExperiment


@dataclass
class CountFlopsSweep(Sweep[CountFlopsExperiment]):
    search_space: Annotated[dict[str, list] | Path | str, tyro.conf.arg(constructor=Path)]
    slurm_gpu: GpuT | None = None

    def __post_init__(self) -> None:
        if isinstance(self.search_space, (Path, str)):
            with open(self.search_space) as f:
                self.search_space: dict[str, list] = json.load(f)

    @property
    def experiments(self) -> list[CountFlopsExperiment]:
        return [
            CountFlopsExperiment(
                **config_kwargs,
                slurm_gpu=self.slurm_gpu,
            )
            for config_kwargs in self._kwargs_product(**self.search_space)
        ]

    def results(self) -> pl.DataFrame:
        results = [{**e.to_dict(), **e.results()} for e in self.experiments if e.is_cached()]
        for r in results:
            r["training_flops"] = float(r["training_flops"])  # avoid overflows in Polars
        return pl.DataFrame(results)


if __name__ == "__main__":
    CountFlopsSweep.cli()
