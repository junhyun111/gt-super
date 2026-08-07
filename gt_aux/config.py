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
    experiments: list[str] = field(default_factory=lambda: [
        "baseline", "shared_detach", "shared_e2e",
    ])

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
        # Keep epoch checkpoints in the project cache so training artifacts are
        # available immediately after every epoch without mixing them with
        # result CSVs and plots under ``outputs``.
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
        return self.checkpoint_dir / f"checkpoint_{self.run_mode}_{experiment}_seed{seed}.pt"

    def as_dict(self) -> dict:
        return {
            "root": str(self.root), "run_mode": self.run_mode,
            "train_images": self.train_images, "val_images": self.val_images,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "device": str(self.device), "experiments": list(self.experiments),
            "seed": self.seed,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().flatten()[:32].numpy().tobytes())
    return digest.hexdigest()[:16]
