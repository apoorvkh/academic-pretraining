import itertools
import multiprocessing as mp
import os
import sys
import tomllib
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, Callable, Generic, Literal, Self, TypeVar

import polars as pl
import submitit
import tango.cli
import torchrunx
import tyro
from tango import Step, StepGraph, StepState, Workspace
from tango.cli import tango_cli
from tango.settings import TangoGlobalSettings
from tqdm import tqdm

from experiments.__tango__ import TangoStringHash, step, tango_executor, tango_settings, tango_workspace
from experiments.__torchrunx__ import distribute
from experiments.config import GpuT

__all__ = ["SlurmJob", "Experiment", "Sweep", "distribute", "TangoStringHash", "step"]

@dataclass(unsafe_hash=True)  # for batching jobs
class SlurmJob:
    time_min: int = 60
    num_nodes: int = 1
    mem_per_node: int = 64
    cpus_per_node: int = 8
    gpus_per_node: int = 0
    gpu_type: GpuT | None = None

    def __post_init__(self):
        with open("slurm.toml", "rb") as f:
            slurm_config = tomllib.load(f)

        if self.gpu_type not in slurm_config:
            raise ValueError(f"GPU type '{self.gpu_type}' not found in slurm.toml")

        gpu_config = slurm_config[self.gpu_type]
        self.constraint = gpu_config.get("constraint") or None
        self.partition = gpu_config.get("partition") or None
        self.account = gpu_config.get("account") or None
        self.nodelist = gpu_config.get("nodelist") or None
        self.reservation = gpu_config.get("reservation") or None

    def to_parameters(self) -> dict:
        return dict(
            stderr_to_stdout=True,
            use_srun=False,
            time=self.time_min,
            nodes=self.num_nodes,
            ntasks_per_node=1,
            mem=f"{self.mem_per_node}G",
            cpus_per_task=self.cpus_per_node,
            gpus_per_node=self.gpus_per_node,
            constraint=self.gpu_type,
            partition=self.partition,
            account=self.account,
            nodelist=self.nodelist,
            additional_parameters=({"reservation": self.reservation} if self.reservation else {}),
        )


@dataclass
class Experiment(ABC):
    @property
    @abstractmethod
    def step_dict(self) -> dict[str, Step]:
        raise NotImplementedError

    @property
    @abstractmethod
    def slurm_job(self) -> SlurmJob | None:
        raise NotImplementedError

    @property
    def dependencies(self) -> Sequence["Experiment"]:
        return []

    def results(self) -> Any:
        return None

    def print_results(self) -> None:
        print(self.results())

    ###

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    @property
    def step_graph(self) -> StepGraph:
        return StepGraph(self.step_dict)

    def step_result(self, step_name: str) -> Any:
        return self.step_dict[step_name].result(workspace=tango_workspace)

    def check_dependencies(self):
        print("\n" "Checking dependencies ... " "\n")
        dependent_experiments = self.dependencies
        not_cached = sum([not e.is_cached() for e in dependent_experiments])
        if not_cached > 0:
            print(f"{not_cached} / {len(dependent_experiments)}" " dependent experiments need to be run first")
            sys.exit(1)

    def _execute_step_graph(self) -> None:
        # avoid "RuntimeError: context has already been set"
        # if CLI was already initialized
        mp.set_start_method(None, force=True)

        try:
            with tango_cli(tango_settings):
                tango.cli.execute_step_graph(
                    step_graph=self.step_graph,
                    workspace=tango_workspace,
                    executor=tango_executor,
                )
        except tango.common.exceptions.CliRunError:
            pass

    def is_cached(self) -> bool:
        for s in self.step_dict.values():
            if s.CACHEABLE and s not in tango_workspace.step_cache:
                return False
        return True

    def is_running(self) -> bool:
        return any([tango_workspace.step_info(s).state == StepState.RUNNING for s in self.step_dict.values()])

    def run(self) -> None:
        self.check_dependencies()
        print("\n" f"Running experiment: {self}" "\n")
        self._execute_step_graph()
        if self.is_cached():
            print("\n" "Results:")
            self.print_results()

    def launch(
        self,
        slurm_executor: submitit.SlurmExecutor | None = None,
        update_executor: bool = True,
        rsync: bool = False,
    ) -> None:
        if slurm_executor is not None:
            slurm_job = self.slurm_job
            if slurm_job is None:
                raise NotImplementedError(f"{self}.slurm_job is None")
            with (
                submitit.helpers.RsyncSnapshot(snapshot_dir=Path(os.environ["SLURM_SNAPSHOT_ROOT"], uuid.uuid4().hex))
                if rsync
                else nullcontext()
            ):
                if update_executor:
                    slurm_executor.update_parameters(**slurm_job.to_parameters())
                slurm_executor.submit(self.run)
        else:
            self.run()

    @classmethod
    @tyro.conf.configure(tyro.conf.OmitArgPrefixes, tyro.conf.OmitSubcommandPrefixes)
    def launch_cli(cls, experiment: Self, slurm: bool = False) -> None:
        slurm_executor = None
        rsync = False
        if slurm:
            slurm_executor = submitit.SlurmExecutor(folder=os.environ["SLURM_OUTPUT_DIR"])
            rsync = True
        return experiment.launch(slurm_executor=slurm_executor, update_executor=True, rsync=rsync)

    @classmethod
    def cli(cls) -> None:
        tyro.cli(cls.launch_cli)


ExperimentT = TypeVar("ExperimentT", bound=Experiment)


class Sweep(Generic[ExperimentT], ABC):
    @property
    @abstractmethod
    def experiments(self) -> list[ExperimentT]:
        raise NotImplementedError

    def results(self) -> pl.DataFrame:
        return pl.DataFrame([{**e.to_dict(), **e.results()} for e in self.experiments if e.is_cached()])

    def print_results(self) -> None:
        print(self.results())

    ###

    @staticmethod
    def _args_product(*args):
        return itertools.product(*args)

    @staticmethod
    def _kwargs_product(**kwargs):
        # from https://stackoverflow.com/a/5228294
        keys = kwargs.keys()
        for instance in itertools.product(*kwargs.values()):
            yield dict(zip(keys, instance))

    @property
    def num_cached(self) -> int:
        return sum([e.is_cached() for e in self.experiments])

    def cached_experiments(self) -> pl.DataFrame:
        return pl.DataFrame([{**e.to_dict(), "cached": e.is_cached()} for e in self.experiments])

    def print_incomplete(self) -> None:
        print("\n" "The following experiments are incomplete and are not currently running:" "\n")
        for e in self.experiments:
            if not e.is_cached() and not e.is_running():
                print(e)

    def sweep(self, slurm: bool = False) -> None:
        if slurm:
            rsync_cm = submitit.helpers.RsyncSnapshot(
                snapshot_dir=Path(os.environ["SLURM_SNAPSHOT_ROOT"], uuid.uuid4().hex)
            )

            slurm_executor = submitit.SlurmExecutor(folder=os.environ["SLURM_OUTPUT_DIR"])

            experiments_by_slurm_job = defaultdict(list)
            for e in self.experiments:
                experiments_by_slurm_job[e.slurm_job].append(e)
            experiment_batches = list(experiments_by_slurm_job.values())
        else:
            rsync_cm = nullcontext()
            slurm_executor = None
            experiment_batches = [[e] for e in self.experiments]

        launch_counter = 0
        pbar = tqdm(total=len(self.experiments)).__enter__()
        rsync_cm.__enter__()

        for batch in experiment_batches:
            slurm_batch_mode = slurm and len(batch) > 1

            if slurm_batch_mode:
                slurm_job = batch[0].slurm_job
                if slurm_job is None:
                    raise NotImplementedError(f"{batch[0]}.slurm_job is None")
                assert slurm_executor is not None  # slurm_executor: submitit.SlurmExecutor
                slurm_executor.update_parameters(**slurm_job.to_parameters())
                batch_cm = slurm_executor.batch()
            else:
                batch_cm = nullcontext()

            batch_cm.__enter__()

            for e in batch:
                pbar.update(1)
                if not e.is_cached() and not e.is_running():
                    e.launch(slurm_executor=slurm_executor, update_executor=(not slurm_batch_mode), rsync=False)
                    launch_counter += 1

            batch_cm.__exit__(None, None, None)

        pbar.__exit__(None, None, None)
        rsync_cm.__exit__(None, None, None)

        if slurm:
            print(f"Submitted {launch_counter} jobs to SLURM.")

    @classmethod
    @tyro.conf.configure(tyro.conf.OmitArgPrefixes, tyro.conf.OmitSubcommandPrefixes, tyro.conf.SuppressFixed)
    def run(
        cls,
        experiment_sweep: Self,
        cmd: Literal["run", "count", "print-incomplete", "print-results"] = "run",
        slurm: bool = False,
    ) -> None:
        pl.Config(tbl_cols=20, tbl_rows=100).__enter__()

        match cmd:
            case "run":
                experiment_sweep.sweep(slurm=slurm)
            case "count":
                print("# cached experiments: " f"{experiment_sweep.num_cached} / {len(experiment_sweep.experiments)}")
            case "print-incomplete":
                experiment_sweep.print_incomplete()
            case "print-results":
                experiment_sweep.print_results()

    @classmethod
    def cli(cls) -> None:
        tyro.cli(cls.run)
