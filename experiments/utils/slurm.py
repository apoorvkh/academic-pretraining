import tomllib
from dataclasses import dataclass

from experiments.config import GpuT

__all__ = ["SlurmJob"]


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
