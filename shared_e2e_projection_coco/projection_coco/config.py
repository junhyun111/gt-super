from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_NAME = "shared_e2e_rep_projected"


@dataclass
class TrainConfig:
    data_root: Path
    output_root: Path
    model_name: str = "SenseTime/deformable-detr"
    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 8
    image_min_size: int = 800
    image_max_size: int = 1333
    lr: float = 2e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    aux_weight: float = 0.5
    feature_level: int = 0
    horizontal_flip_p: float = 0.5
    seed: int = 42
    train_limit: int | None = None
    val_limit: int | None = None
    eval_every: int = 1
    save_every: int = 5
    gradient_log_every: int = 100
    amp: bool = True
    deterministic: bool = False
    disable_custom_kernels: bool = False
    offline: bool = False
    skip_initial_eval: bool = False

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.output_root = Path(self.output_root).expanduser().resolve()
        positive = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "image_min_size": self.image_min_size,
            "image_max_size": self.image_max_size,
            "lr": self.lr,
            "backbone_lr": self.backbone_lr,
            "grad_clip": self.grad_clip,
            "eval_every": self.eval_every,
            "save_every": self.save_every,
            "gradient_log_every": self.gradient_log_every,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These settings must be positive: {', '.join(invalid)}")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.image_min_size > self.image_max_size:
            raise ValueError("image_min_size must not exceed image_max_size")
        if not 0.0 <= self.horizontal_flip_p <= 1.0:
            raise ValueError("horizontal_flip_p must be between 0 and 1")
        if self.aux_weight <= 0:
            raise ValueError("aux_weight must be positive for the projection model")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        for name in ("train_limit", "val_limit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")

    @property
    def image_size(self) -> dict[str, int]:
        return {
            "shortest_edge": self.image_min_size,
            "longest_edge": self.image_max_size,
        }

    @property
    def run_dir(self) -> Path:
        return self.output_root / EXPERIMENT_NAME / f"seed_{self.seed}"

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def history_path(self) -> Path:
        return self.run_dir / "history.csv"

    @property
    def gradients_path(self) -> Path:
        return self.run_dir / "projection_gradients.csv"

    def create_output_dirs(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict:
        values = asdict(self)
        values["data_root"] = str(self.data_root)
        values["output_root"] = str(self.output_root)
        values["experiment"] = EXPERIMENT_NAME
        return values

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().flatten()[:32].numpy().tobytes())
    return digest.hexdigest()[:16]


def configure_huggingface_cache(cache_path: str | Path | None) -> None:
    if cache_path is None:
        return
    resolved = str(Path(cache_path).expanduser().resolve())
    os.environ.setdefault("HF_HOME", resolved)
