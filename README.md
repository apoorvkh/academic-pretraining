# [$100K or 100 Days: Trade-offs when Pre-Training with Academic Resources](https://arxiv.org/abs/2410.23261)

Apoorv Khandelwal, Tian Yun, Nihal V. Nayak, Jack Merullo, Stephen H. Bach, Chen Sun, Ellie Pavlick

Published at COLM 2025

---

Use this repository to:

1. determine the *best* HuggingFace Trainer settings for training your model on your hardware
2. determine how long training will take

Refer to [our paper](https://arxiv.org/abs/2410.23261) for further insights about the current state of academic compute, more training times (for several models and GPUs), and for help deciding which GPUs to buy.

## Installation

```bash
git clone https://github.com/apoorvkh/academic-pretraining.git
cd academic-pretraining

# Install pixi (like conda)
curl -fsSL https://pixi.sh/install.sh | bash

# Install project dependencies (may take a few minutes the first time)
pixi shell
```

## Activate your virtual environment: `pixi shell`

## Run our benchmark on your model / hardware

<details><summary>[expand] <code>python scripts/benchmark.py --help</code></summary>

```bash
╭─ options ───────────────────────────────────────────────╮
│ -h, --help              show this help message and exit │
│ --num-nodes INT         (required)                      │
│ --gpus-per-node INT     (required)                      │
│ --gpu-type {geforce3090,v100,a6000,a40,l40,a100,h100}   │
│                         (required)                      │
│ --model {roberta,pythia-160m,pythia-410m,pythia-1b,...} │
│                         (required)                      │
│ --methods {naive,free-lunch,all}                        │
│                         (default: all)                  │
│ --cmd {run,count,print-incomplete,print-results}        │
│                         (default: run)                  │
│ --slurm, --no-slurm     (default: False)                │
╰─────────────────────────────────────────────────────────╯
# truncated output (run for full lists)
```
</details>

You can first test `--methods naive` and `--methods free-lunch` (approx. 10 minutes). If these fail due to memory constraints or you would like to try to reduce training time, you can test `--methods all` (approx. 2 hours).

`--methods all` searches our space of efficient training methods (Sec. 3.2.1) and is likely to find gains when the model is large or the GPU memory is small.

For example (run this on specified hardware): `python scripts/benchmark.py --num-nodes 1 --gpus-per-node 4 --gpu-type a100 --model pythia-1b --methods all --cmd run`

After your results are computed, you can run our scripts to generate the optimal [`TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).

<details><summary>[expand] <code>python scripts/print_optimal_config.py --num-nodes 1 --gpus-per-node 4 --gpu-type a100 --model pythia-1b</code></summary>

```bash
┌───────────┬───────────────┬──────────┬───────────┬────────────┬──────────────────────────┬──────────┬────────────┬──────────────────┬────────────────┬───────────────┐
│ num_nodes ┆ gpus_per_node ┆ gpu_type ┆ model     ┆ free_lunch ┆ activation_checkpointing ┆ sharding ┆ offloading ┆ micro_batch_size ┆ grad_acc_steps ┆ training_days │
│ ---       ┆ ---           ┆ ---      ┆ ---       ┆ ---        ┆ ---                      ┆ ---      ┆ ---        ┆ ---              ┆ ---            ┆ ---           │
│ i64       ┆ i64           ┆ str      ┆ str       ┆ bool       ┆ bool                     ┆ str      ┆ bool       ┆ i64              ┆ i64            ┆ f64           │
╞═══════════╪═══════════════╪══════════╪═══════════╪════════════╪══════════════════════════╪══════════╪════════════╪══════════════════╪════════════════╪═══════════════╡
│ 1         ┆ 4             ┆ a100     ┆ pythia-1b ┆ true       ┆ false                    ┆ zero_1   ┆ false      ┆ 16               ┆ 16             ┆ 17.571102     │
└───────────┴───────────────┴──────────┴───────────┴────────────┴──────────────────────────┴──────────┴────────────┴──────────────────┴────────────────┴───────────────┘
```
</details>

<details><summary>[expand] <code>python scripts/to_training_arguments.py --output artifacts/training_arguments.json --num-nodes 1 --gpus-per-node 4 --gpu-type a100 --model pythia-1b --free-lunch --sharding zero_1 --micro-batch-size 16 --gradient-accumulation-steps 16</code></summary>

`artifacts/training_arguments.json`

```json
{
    "max_steps": 143000,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 16,
    "lr_scheduler_type": "cosine_with_min_lr",
    "lr_scheduler_kwargs": {
        "min_lr_rate": 0.1
    },
    "warmup_steps": 1430,
    "gradient_checkpointing": false,
    "bf16": true,
    "fp16": false,
    "tf32": true,
    "fsdp": "",
    "fsdp_config": null,
    "deepspeed": {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
                "adam_w_mode": false
            }
        },
        "zero_optimization": {
            "stage": 1
        }
    },
    "ddp_find_unused_parameters": false,
    "torch_compile": true,
    "max_grad_norm": 1.0
}
```
</details>

## Training script

We provide a boilerplate training script that can be used to train models using arguments from the above steps. This script operates using `transformers.Trainer` and works with our codebase, but can be customized further or migrated to your own project (after copying in necessary code). This script loads a dummy dataset, which you should change to one of your own interest. This script automatically runs multi-GPU training in the local (or SLURM) environment: this behavior can be further customized using the [torchrunx](https://torchrunx.readthedocs.io/stable) `--launcher` CLI arguments.

For example: `python scripts/training.py --output-dir output --model-type pythia-410m --training-arguments artifacts/training_arguments.json`

## Extending this codebase

You can add models via [`src/models/__init__.py`](./src/models/__init__.py) and GPUs via [`src/gpus.py`](./src/gpus.py). These should automatically populate in the CLI commands. You should run `pyright` to check for any missing implementations. You should then sanity check simple settings (run `--methods naive` and `--methods free-lunch` in our [benchmark](#run-our-benchmark-on-your-model--hardware)).

## Miscellaneous

> [!NOTE]
> This repository is written as an abstraction over the [Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://github.com/pytorch/pytorch) libraries (currently pinned to `transformers==4.42.3` and `torch==2.3.1`). We automatically handle all dependencies, multi-node/GPU environments, and experiment caching/execution (with SLURM support). Our codebase should also be easy to extend for new GPUs and models. We plan to continually update this repository with new features. We provide all [artifacts](#artifacts) from our survey and experiments (pinned to previous revisions of our repository/paper).

### Caching

All experiment results are automatically cached (using [AI2 Tango](https://ai2-tango.readthedocs.io)) in the `tango_workspace/` directory. Accordingly, if you run the same experiment twice (redundantly), the second run will simply retrieve the result from the cache. You can set `export TANGO_WORKSPACE_DIR=` or delete `tango_workspace/` to invalidate this cache.

### SLURM integration

Our codebase natively features support for running experiments via SLURM! Simply pass the `--slurm` argument when running experiments. Then, your experiments will automatically be submitted to the SLURM queue with the specified hardware. Logs from your SLURM jobs can be found at `.cache/slurm_outputs`. You must adjust `slurm.toml` to specify your own cluster's partitons/etc per GPU type. You may also want to further adjust the specifications in `experiments/training_time_empirical.py: TrainingTimeEmpirical.slurm_job`.

### Plotting

We provide all plotting scripts used to generate the figures in our paper at `scripts/plotting`. You can adjust these to visualize your own experiments/results. You can load these scripts as [Marimo](https://marimo.io/) notebooks, e.g. with

```bash
marimo edit scripts/plotting/optimal_table.py
```

## Artifacts

We provide all artifacts from our paper and experiments as `artifacts.tar` under certain [releases](https://github.com/apoorvkh/academic-pretraining/releases).

| Paper Revision | Release Tag |
| --- | --- |
| [arxiv-v1](https://arxiv.org/abs/2410.23261v1) | v1.0.0 |

Our artifacts include:

- anonymized results from our survey (`artifacts/survey.csv`)
- the [Tango](http://ai2-tango.readthedocs.io) workspace (`artifacts/tango_workspace.tgz`) with cached results from all our experiments

You can exactly reproduce all plots in our paper ([#plotting](#plotting)) using this workspace.

You can checkout a specific release and its artifacts via:

```bash
RELEASE_TAG=v1.0.0

git clone https://github.com/apoorvkh/academic-pretraining.git --branch $RELEASE_TAG --single-branch academic-pretraining-$RELEASE_TAG
cd academic-pretraining-$RELEASE_TAG

# download artifacts
curl -fsSL https://github.com/apoorvkh/academic-pretraining/releases/download/$RELEASE_TAG/artifacts.tar | tar xvf -

# unpack Tango workspace (many files; may take a few minutes)
tar xzf artifacts/tango_workspace.tgz
```

## Citation

```bibtex
@inproceedings{
    khandelwal2025:100k,
    title={\$100K or 100 Days: Trade-offs when Pre-Training with Academic Resources},
    author={Apoorv Khandelwal and Tian Yun and Nihal V. Nayak and Jack Merullo and Stephen Bach and Chen Sun and Ellie Pavlick},
    booktitle={Second Conference on Language Modeling},
    year={2025},
    url={https://arxiv.org/abs/2410.23261}
}
```
