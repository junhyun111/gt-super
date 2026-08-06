from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor
from tqdm.auto import tqdm

from .config import ExperimentConfig, LABEL2ID, seed_everything


@dataclass
class DataBundle:
    processor: AutoImageProcessor
    train_records: list[dict]
    val_records: list[dict]
    full_train_records: list[dict]
    full_val_records: list[dict]


def parse_voc_record(config: ExperimentConfig, image_id: str) -> dict:
    annotation_path = config.voc_root / "Annotations" / f"{image_id}.xml"
    root = ET.parse(annotation_path).getroot()
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in LABEL2ID:
            continue
        box = obj.find("bndbox")
        x1 = max(0.0, float(box.findtext("xmin")) - 1.0)
        y1 = max(0.0, float(box.findtext("ymin")) - 1.0)
        x2 = min(float(width), float(box.findtext("xmax")))
        y2 = min(float(height), float(box.findtext("ymax")))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            labels.append(LABEL2ID[name])
    return {
        "image_id": image_id,
        "image_path": str(config.voc_root / "JPEGImages" / f"{image_id}.jpg"),
        "width": width, "height": height,
        "boxes_xyxy": boxes, "labels": labels,
    }


def prepare_data(config: ExperimentConfig) -> DataBundle:
    assert (config.voc_root / "JPEGImages").exists(), f"Missing {config.voc_root / 'JPEGImages'}"
    assert (config.voc_root / "Annotations").exists(), f"Missing {config.voc_root / 'Annotations'}"
    trainval_file = config.voc_root / "ImageSets" / "Main" / "trainval.txt"
    all_ids = [line.strip() for line in trainval_file.read_text().splitlines() if line.strip()]
    assert len(all_ids) == 5011, f"Expected 5011 VOC trainval images, got {len(all_ids)}"

    selection_rng = np.random.default_rng(config.seed)
    selected_ids = selection_rng.permutation(all_ids)[:config.full_train_images + config.full_val_images]
    records = [
        parse_voc_record(config, str(image_id))
        for image_id in tqdm(selected_ids, desc="VOC XML")
    ]
    records = [record for record in records if record["boxes_xyxy"]]
    assert len(records) == config.full_train_images + config.full_val_images

    split_rng = np.random.default_rng(config.seed + 1)
    order = split_rng.permutation(len(records))
    full_train_records = [records[index] for index in order[:config.full_train_images]]
    full_val_records = [
        records[index]
        for index in order[config.full_train_images:config.full_train_images + config.full_val_images]
    ]
    train_object_count = sum(len(record["boxes_xyxy"]) for record in full_train_records)
    val_object_count = sum(len(record["boxes_xyxy"]) for record in full_val_records)
    assert (train_object_count, val_object_count) == (9180, 2530)

    manifest = {
        "seed": config.seed, "selection_seed": config.seed,
        "split_seed": config.seed + 1,
        "full_train_ids": [record["image_id"] for record in full_train_records],
        "full_val_ids": [record["image_id"] for record in full_val_records],
    }
    (config.output_dir / "voc2007_split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Full split: train={len(full_train_records)} ({train_object_count} objects), "
        f"val={len(full_val_records)} ({val_object_count} objects)"
    )
    print(f"Current run: train={config.train_images}, val={config.val_images}")

    try:
        processor = AutoImageProcessor.from_pretrained(config.checkpoint, local_files_only=True)
    except OSError:
        processor = AutoImageProcessor.from_pretrained(config.checkpoint)
    return DataBundle(
        processor=processor,
        train_records=full_train_records[:config.train_images],
        val_records=full_val_records[:config.val_images],
        full_train_records=full_train_records,
        full_val_records=full_val_records,
    )


class VOCDataset(Dataset):
    def __init__(self, records: list[dict], config: ExperimentConfig, training: bool = False):
        self.records = records
        self.config = config
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        boxes = torch.tensor(record["boxes_xyxy"], dtype=torch.float32)
        labels = torch.tensor(record["labels"], dtype=torch.long)
        width, height = image.size
        if self.training and random.random() < self.config.horizontal_flip_p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            old_x1, old_x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
            boxes[:, 0] = width - old_x2
            boxes[:, 2] = width - old_x1
        return image, {
            "image_id": int(record["image_id"]),
            "boxes_xyxy": boxes, "labels": labels,
            "orig_size": torch.tensor([height, width], dtype=torch.long),
        }


def collate_detection_batch(batch, processor, config: ExperimentConfig):
    images, targets = zip(*batch)
    annotations = []
    for target in targets:
        boxes = target["boxes_xyxy"]
        boxes_xywh = boxes.clone()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
        areas = boxes_xywh[:, 2] * boxes_xywh[:, 3]
        annotations.append({
            "image_id": target["image_id"],
            "annotations": [
                {
                    "id": target["image_id"] * 1000 + index,
                    "image_id": target["image_id"],
                    "category_id": int(label), "bbox": box.tolist(),
                    "area": float(area), "iscrowd": 0,
                }
                for index, (box, label, area) in enumerate(
                    zip(boxes_xywh, target["labels"], areas)
                )
            ],
        })
    encoded = processor(
        images=list(images), annotations=annotations, return_tensors="pt",
        size=config.image_size,
    )
    return {
        "pixel_values": encoded["pixel_values"],
        "pixel_mask": encoded["pixel_mask"],
        "labels": encoded["labels"],
        "eval_targets": [
            {
                "boxes": target["boxes_xyxy"], "labels": target["labels"],
                "orig_size": target["orig_size"], "image_id": target["image_id"],
            }
            for target in targets
        ],
    }


def make_loaders(config: ExperimentConfig, bundle: DataBundle, seed: int):
    generator = torch.Generator().manual_seed(seed)
    collate = lambda batch: collate_detection_batch(batch, bundle.processor, config)
    train_loader = DataLoader(
        VOCDataset(bundle.train_records, config, training=True),
        batch_size=config.batch_size, shuffle=True, generator=generator,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        VOCDataset(bundle.val_records, config, training=False),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    return train_loader, val_loader
