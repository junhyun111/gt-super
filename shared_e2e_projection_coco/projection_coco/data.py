from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoImageProcessor

from .config import TrainConfig
from .distributed import DistributedContext


@dataclass
class DataBundle:
    processor: AutoImageProcessor
    train_dataset: "CocoDetectionDataset"
    val_dataset: "CocoDetectionDataset"
    category_ids: set[int]


class CocoDetectionDataset(Dataset):
    """COCO detection dataset that preserves the original sparse category IDs.

    SenseTime/deformable-detr has a 91-logit COCO head whose valid targets use
    the original IDs (1..90 with gaps). Compacting them to 0..79 would silently
    train the wrong pretrained classifier rows.
    """

    def __init__(
        self,
        image_root: Path,
        annotation_file: Path,
        *,
        training: bool,
        horizontal_flip_p: float,
        limit: int | None,
        selection_seed: int,
    ) -> None:
        self.image_root = Path(image_root)
        self.annotation_file = Path(annotation_file)
        self.training = training
        self.horizontal_flip_p = horizontal_flip_p
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Missing COCO image directory: {self.image_root}")
        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"Missing COCO annotation file: {self.annotation_file}")

        self.coco = COCO(str(self.annotation_file))
        ids = sorted(self.coco.getImgIds())
        if limit is not None and limit < len(ids):
            rng = np.random.default_rng(selection_seed)
            ids = sorted(int(value) for value in rng.choice(ids, size=limit, replace=False))
        self.image_ids = ids
        self.category_ids = set(int(value) for value in self.coco.getCatIds())
        if not self.image_ids:
            raise ValueError(f"No images found in {self.annotation_file}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def _valid_annotations(self, image_id: int, width: int, height: int) -> list[dict]:
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        annotations = []
        for original in self.coco.loadAnns(annotation_ids):
            annotation = copy.deepcopy(original)
            x, y, width_box, height_box = map(float, annotation["bbox"])
            x1 = min(max(x, 0.0), float(width))
            y1 = min(max(y, 0.0), float(height))
            x2 = min(max(x + width_box, 0.0), float(width))
            y2 = min(max(y + height_box, 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            annotation["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            annotation["area"] = (x2 - x1) * (y2 - y1)
            annotation["iscrowd"] = 0
            annotations.append(annotation)
        return annotations

    def __getitem__(self, index: int):
        image_id = int(self.image_ids[index])
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.image_root / image_info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing COCO image: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        annotations = self._valid_annotations(image_id, width, height)

        if self.training and random.random() < self.horizontal_flip_p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for annotation in annotations:
                x, y, box_width, box_height = annotation["bbox"]
                annotation["bbox"] = [width - x - box_width, y, box_width, box_height]

        boxes = []
        labels = []
        for annotation in annotations:
            x, y, box_width, box_height = annotation["bbox"]
            boxes.append([x, y, x + box_width, y + box_height])
            labels.append(int(annotation["category_id"]))
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        processor_target = {
            "image_id": image_id,
            "annotations": annotations,
        }
        eval_target = {
            "image_id": image_id,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "orig_size": torch.tensor([height, width], dtype=torch.long),
        }
        return image, processor_target, eval_target


def collate_detection_batch(batch, *, processor, image_size):
    images, processor_targets, eval_targets = zip(*batch)
    encoded = processor(
        images=list(images),
        annotations=list(processor_targets),
        return_tensors="pt",
        size=image_size,
    )
    return {
        "pixel_values": encoded["pixel_values"],
        "pixel_mask": encoded["pixel_mask"],
        "labels": encoded["labels"],
        "eval_targets": list(eval_targets),
    }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_data(config: TrainConfig) -> DataBundle:
    train_images = config.data_root / "train2017"
    val_images = config.data_root / "val2017"
    annotations = config.data_root / "annotations"
    train_annotations = annotations / "instances_train2017.json"
    val_annotations = annotations / "instances_val2017.json"

    try:
        processor = AutoImageProcessor.from_pretrained(
            config.model_name, local_files_only=config.offline
        )
    except OSError as error:
        if config.offline:
            raise RuntimeError(
                f"Model processor is not cached for offline use: {config.model_name}"
            ) from error
        raise

    train_dataset = CocoDetectionDataset(
        train_images,
        train_annotations,
        training=True,
        horizontal_flip_p=config.horizontal_flip_p,
        limit=config.train_limit,
        selection_seed=config.seed,
    )
    val_dataset = CocoDetectionDataset(
        val_images,
        val_annotations,
        training=False,
        horizontal_flip_p=0.0,
        limit=config.val_limit,
        selection_seed=config.seed + 1,
    )
    if train_dataset.category_ids != val_dataset.category_ids:
        raise ValueError("COCO train/validation category sets do not match")
    return DataBundle(
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        category_ids=train_dataset.category_ids,
    )


def make_train_loader(
    config: TrainConfig,
    bundle: DataBundle,
    context: DistributedContext,
):
    sampler = None
    if context.distributed:
        sampler = DistributedSampler(
            bundle.train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
    generator = torch.Generator().manual_seed(config.seed + context.rank)
    collate = partial(
        collate_detection_batch,
        processor=bundle.processor,
        image_size=config.image_size,
    )
    kwargs = {}
    if config.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(
        bundle.train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=context.device.type == "cuda",
        collate_fn=collate,
        worker_init_fn=_seed_worker,
        **kwargs,
    )
    return loader, sampler


def make_val_loader(config: TrainConfig, bundle: DataBundle, context: DistributedContext):
    collate = partial(
        collate_detection_batch,
        processor=bundle.processor,
        image_size=config.image_size,
    )
    kwargs = {}
    if config.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        bundle.val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=context.device.type == "cuda",
        collate_fn=collate,
        worker_init_fn=_seed_worker,
        **kwargs,
    )
