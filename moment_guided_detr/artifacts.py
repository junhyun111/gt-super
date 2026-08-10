from __future__ import annotations

from pathlib import Path

from gt_aux.config import ExperimentConfig

from .model import EXPERIMENT_NAME


def output_dir(config: ExperimentConfig) -> Path:
    path = config.root / "outputs" / "moment_guided_detr"
    path.mkdir(parents=True, exist_ok=True)
    return path


def history_path(config: ExperimentConfig, seed: int) -> Path:
    return output_dir(config) / f"history_{config.run_mode}_{EXPERIMENT_NAME}_seed{seed}.csv"


def gradients_path(config: ExperimentConfig, seed: int) -> Path:
    return output_dir(config) / f"gradients_{config.run_mode}_{EXPERIMENT_NAME}_seed{seed}.csv"


def checkpoint_path(config: ExperimentConfig, seed: int) -> Path:
    path = config.root / "cache" / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"checkpoint_{config.run_mode}_{EXPERIMENT_NAME}_seed{seed}.pt"


def evaluation_dir(config: ExperimentConfig) -> Path:
    path = output_dir(config) / "evaluation"
    path.mkdir(parents=True, exist_ok=True)
    return path
