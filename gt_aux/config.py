from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
)
LABEL2ID = {name: index for index, name in enumerate(VOC_CLASSES)}
ID2LABEL = {index: name for name, index in LABEL2ID.items()}


@dataclass
class ExperimentConfig:
    root: Path
    run_mode: str = "smoke"
    checkpoint: str = "SenseTime/deformable-detr"
    seed: int = 42
    data_seed: int = 42
    full_train_images: int = 3000
    full_val_images: int = 750
    smoke_train_images: int = 400
    smoke_val_images: int = 100
    smoke_epochs: int = 7
    full_epochs: int = 7
    batch_size: int | None = None
    num_workers: int = 0
    image_min_size_smoke: int = 384
    image_max_size_smoke: int = 640
    image_min_size_full: int = 640
    image_max_size_full: int = 1000
    lr: float = 2e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    base_aux_weight: float = 0.5
    feature_level: int = 0
    horizontal_flip_p: float = 0.5
    use_amp: bool | None = None
    disable_custom_kernels: bool | None = None
    deterministic: bool = True
    save_epoch_checkpoints: bool = False
    experiments: list[str] = field(default_factory=lambda: [
        "baseline", "shared_detach", "shared_e2e",
    ])

    @classmethod
    def for_run(
        cls,
        root: Path,
        *,
        run_mode: str,
        train_images: int,
        val_images: int,
        epochs: int,
        image_min_size: int,
        image_max_size: int,
        **kwargs,
    ) -> "ExperimentConfig":
        """Build one run from explicit notebook-facing settings.

        The dataclass keeps smoke/full defaults for backwards compatibility,
        while notebooks can configure the active run without editing this file.
        """
        if run_mode not in {"smoke", "full"}:
            raise ValueError("run_mode must be 'smoke' or 'full'")
        active_values = {
            f"{run_mode}_train_images": train_images,
            f"{run_mode}_val_images": val_images,
            f"{run_mode}_epochs": epochs,
            f"image_min_size_{run_mode}": image_min_size,
            f"image_max_size_{run_mode}": image_max_size,
        }
        return cls(root=root, run_mode=run_mode, **active_values, **kwargs)

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        if self.run_mode not in {"smoke", "full"}:
            raise ValueError("run_mode must be 'smoke' or 'full'")
        if self.batch_size is None:
            self.batch_size = 2 if torch.cuda.is_available() else 1
        if self.use_amp is None:
            self.use_amp = torch.cuda.is_available()
        if self.disable_custom_kernels is None:
            self.disable_custom_kernels = os.name == "nt" or not torch.cuda.is_available()
        positive_values = {
            "train_images": self.train_images,
            "val_images": self.val_images,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "image_min_size": self.image_size["shortest_edge"],
            "image_max_size": self.image_size["longest_edge"],
            "lr": self.lr,
            "backbone_lr": self.backbone_lr,
            "grad_clip": self.grad_clip,
        }
        invalid = [name for name, value in positive_values.items() if value is None or value <= 0]
        if invalid:
            raise ValueError(f"These settings must be positive: {', '.join(invalid)}")
        if self.image_size["shortest_edge"] > self.image_size["longest_edge"]:
            raise ValueError("image_min_size must not exceed image_max_size")
        if not 0.0 <= self.horizontal_flip_p <= 1.0:
            raise ValueError("horizontal_flip_p must be between 0 and 1")
        if self.base_aux_weight < 0:
            raise ValueError("base_aux_weight must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.train_images > self.full_train_images:
            raise ValueError("train_images must not exceed full_train_images")
        if self.val_images > self.full_val_images:
            raise ValueError("val_images must not exceed full_val_images")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def voc_root(self) -> Path:
        return self.data_dir / "VOCdevkit" / "VOC2007"

    @property
    def output_dir(self) -> Path:
        path = self.root / "outputs" / "detr_gt_auxiliary"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoint_dir(self) -> Path:
        # Model/seed subdirectories below this root keep epoch snapshots from
        # different experiments out of one large flat directory.
        path = self.root / "cache" / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def train_images(self) -> int:
        return self.smoke_train_images if self.run_mode == "smoke" else self.full_train_images

    @property
    def val_images(self) -> int:
        return self.smoke_val_images if self.run_mode == "smoke" else self.full_val_images

    @property
    def epochs(self) -> int:
        return self.smoke_epochs if self.run_mode == "smoke" else self.full_epochs

    @property
    def image_size(self) -> dict[str, int]:
        if self.run_mode == "smoke":
            return {"shortest_edge": self.image_min_size_smoke, "longest_edge": self.image_max_size_smoke}
        return {"shortest_edge": self.image_min_size_full, "longest_edge": self.image_max_size_full}

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def history_path(self, experiment: str, seed: int | None = None) -> Path:
        seed = self.seed if seed is None else seed
        return self.output_dir / f"history_{self.run_mode}_{experiment}_seed{seed}.csv"

    def gradients_path(self, experiment: str, seed: int | None = None) -> Path:
        seed = self.seed if seed is None else seed
        return self.output_dir / f"gradients_{self.run_mode}_{experiment}_seed{seed}.csv"

    def checkpoint_path(self, experiment: str, seed: int | None = None) -> Path:
        seed = self.seed if seed is None else seed
        model_seed_dir = self.checkpoint_dir / experiment / f"seed_{seed}"
        model_seed_dir.mkdir(parents=True, exist_ok=True)
        return model_seed_dir / f"checkpoint_{self.run_mode}_{experiment}_seed{seed}.pt"

    def as_dict(self) -> dict:
        return {
            "root": str(self.root), "run_mode": self.run_mode,
            "checkpoint": self.checkpoint,
            "train_images": self.train_images, "val_images": self.val_images,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "num_workers": self.num_workers, "image_size": self.image_size,
            "lr": self.lr, "backbone_lr": self.backbone_lr,
            "weight_decay": self.weight_decay, "grad_clip": self.grad_clip,
            "base_aux_weight": self.base_aux_weight,
            "feature_level": self.feature_level,
            "horizontal_flip_p": self.horizontal_flip_p,
            "use_amp": self.use_amp,
            "deterministic": self.deterministic,
            "save_epoch_checkpoints": self.save_epoch_checkpoints,
            "device": str(self.device), "experiments": list(self.experiments),
            "seed": self.seed, "data_seed": self.data_seed,
        }


def seed_everything(seed: int, deterministic: bool | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic is not None:
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=True)


def model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().flatten()[:32].numpy().tobytes())
    return digest.hexdigest()[:16]
