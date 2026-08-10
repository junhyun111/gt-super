from __future__ import annotations

import contextlib
import gc
import io
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm

from gt_aux.config import ID2LABEL, ExperimentConfig, model_fingerprint
from gt_aux.data import DataBundle
from gt_aux.eval import load_checkpoint as load_baseline_checkpoint

from .artifacts import checkpoint_path
from .model import EXPERIMENT_NAME, make_moment_model
from .train import MomentTrainingSettings, move_labels_to_device


IOU_CURVE = [round(0.50 + 0.05 * index, 2) for index in range(10)]
ERROR_TYPES = [
    "TP",
    "Duplicate FP",
    "Localization FP",
    "Classification FP",
    "Background FP",
    "Missed GT",
]


def release_model(model) -> None:
    model.close()
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_moment_checkpoint(config: ExperimentConfig, seed: int):
    path = checkpoint_path(config, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(f"Unexpected experiment in {path}: {checkpoint.get('experiment')}")
    if int(checkpoint.get("seed")) != int(seed):
        raise ValueError(f"Checkpoint seed={checkpoint.get('seed')}, expected {seed}")
    settings = MomentTrainingSettings(**checkpoint.get("moment_settings", {}))
    model, _ = make_moment_model(
        config,
        seed,
        center_weight=settings.center_weight,
        covariance_weight=settings.covariance_weight,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = {
        "experiment": checkpoint.get("experiment"),
        "seed": checkpoint.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "config": checkpoint.get("config"),
        "moment_settings": checkpoint.get("moment_settings"),
        "path": str(path),
    }
    del checkpoint
    return model, metadata, model_fingerprint(model.detector)


def load_comparison_model(
    config: ExperimentConfig,
    bundle: DataBundle,
    experiment: str,
    seed: int,
):
    if experiment == "baseline":
        model, checkpoint, _ = load_baseline_checkpoint(
            config, bundle, experiment, seed
        )
        metadata = {
            "experiment": checkpoint.get("experiment"),
            "seed": checkpoint.get("seed"),
            "epoch": checkpoint.get("epoch"),
            "config": checkpoint.get("config"),
            "path": str(config.checkpoint_path(experiment, seed)),
        }
        del checkpoint
        fingerprint = model_fingerprint(model.detector)
        return model, metadata, fingerprint
    if experiment == EXPERIMENT_NAME:
        return load_moment_checkpoint(config, seed)
    raise ValueError(f"Only baseline and {EXPERIMENT_NAME} are supported, got {experiment!r}")


def _valid_mean(values: torch.Tensor) -> float:
    values = values[values >= 0]
    return float(values.mean()) if values.numel() else float("nan")


@torch.inference_mode()
def collect_evaluation_run(
    config: ExperimentConfig,
    bundle: DataBundle,
    val_loader,
    experiment: str,
    seed: int,
):
    """Run inference once and return overall, scale, and raw prediction rows."""

    model, metadata, fingerprint = load_comparison_model(
        config, bundle, experiment, seed
    )
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        extended_summary=True,
        iou_thresholds=IOU_CURVE,
    )
    prediction_rows = []
    start = time.perf_counter()
    try:
        for batch in tqdm(
            val_loader, desc=f"{experiment}/seed{seed}", leave=False
        ):
            result = model(
                pixel_values=batch["pixel_values"].to(config.device, non_blocking=True),
                pixel_mask=batch["pixel_mask"].to(config.device, non_blocking=True),
                labels=None,
            )
            if result.get("moment_executed", False):
                raise AssertionError("Moment path ran during inference")
            sizes = torch.stack(
                [target["orig_size"] for target in batch["eval_targets"]]
            ).to(config.device)
            predictions = bundle.processor.post_process_object_detection(
                result["outputs"], threshold=0.0, target_sizes=sizes
            )
            predictions_cpu = [
                {key: value.detach().cpu() for key, value in prediction.items()}
                for prediction in predictions
            ]
            targets_cpu = [
                {"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()}
                for target in batch["eval_targets"]
            ]
            metric.update(predictions_cpu, targets_cpu)

            for prediction, target in zip(predictions_cpu, batch["eval_targets"]):
                image_id = int(target["image_id"])
                for score, label, box in zip(
                    prediction["scores"], prediction["labels"], prediction["boxes"]
                ):
                    x1, y1, x2, y2 = map(float, box.tolist())
                    prediction_rows.append(
                        {
                            "experiment": experiment,
                            "seed": seed,
                            "image_id": image_id,
                            "category_id": int(label) + 1,
                            "score": float(score),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )

        elapsed = time.perf_counter() - start
        values = metric.compute()
        overall_row = {
            "experiment": experiment,
            "seed": seed,
            "checkpoint_epoch": metadata.get("epoch"),
            "fingerprint": fingerprint,
            "mAP": float(values["map"]),
            "AP50": float(values["map_50"]),
            "AP75": float(values["map_75"]),
            "AR100": float(values["mar_100"]),
            "inference_seconds": elapsed,
        }
        precision = values["precision"]
        scale_rows = []
        for scale, suffix, area_index in (
            ("small", "small", 1),
            ("medium", "medium", 2),
            ("large", "large", 3),
        ):
            ap = float(values[f"map_{suffix}"])
            ar = float(values[f"mar_{suffix}"])
            scale_rows.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "scale": scale,
                    "AP": ap if ap >= 0 else np.nan,
                    "AP50": _valid_mean(
                        precision[IOU_CURVE.index(0.50), :, :, area_index, -1]
                    ),
                    "AP75": _valid_mean(
                        precision[IOU_CURVE.index(0.75), :, :, area_index, -1]
                    ),
                    "AR": ar if ar >= 0 else np.nan,
                }
            )
        return overall_row, scale_rows, prediction_rows
    finally:
        release_model(model)


def load_geometry_model(
    config: ExperimentConfig,
    experiment: str,
    seed: int,
):
    """Load either checkpoint into the moment wrapper for read-only diagnostics."""

    if experiment == EXPERIMENT_NAME:
        return load_moment_checkpoint(config, seed)
    if experiment != "baseline":
        raise ValueError(f"Unsupported geometry experiment: {experiment!r}")

    path = config.checkpoint_path("baseline", seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != "baseline":
        raise ValueError(f"Unexpected experiment in {path}: {checkpoint.get('experiment')}")
    model, _ = make_moment_model(config, seed)
    detector_state = {
        key.removeprefix("detector."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("detector.")
    }
    model.detector.load_state_dict(detector_state)
    metadata = {
        "experiment": "baseline",
        "seed": seed,
        "epoch": checkpoint.get("epoch"),
        "config": checkpoint.get("config"),
        "path": str(path),
    }
    del checkpoint
    return model, metadata, model_fingerprint(model.detector)


@torch.inference_mode()
def collect_attention_geometry(
    config: ExperimentConfig,
    bundle: DataBundle,
    val_loader,
    experiment: str,
    seed: int,
) -> list[dict]:
    """Measure matched-query moment errors without changing model weights."""

    model, metadata, _ = load_geometry_model(config, experiment, seed)
    model.eval()
    rows = []
    try:
        for batch in tqdm(
            val_loader, desc=f"{experiment} geometry/seed{seed}", leave=False
        ):
            labels = move_labels_to_device(batch["labels"], config.device)
            result = model(
                pixel_values=batch["pixel_values"].to(config.device, non_blocking=True),
                pixel_mask=batch["pixel_mask"].to(config.device, non_blocking=True),
                labels=labels,
                return_moment_details=True,
            )
            for detail in result["moment_details"]:
                target = batch["eval_targets"][detail.pop("batch_index")]
                area = detail["target_area_pixels"]
                scale = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
                rows.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "checkpoint_epoch": metadata.get("epoch"),
                        "image_id": int(target["image_id"]),
                        "scale": scale,
                        **detail,
                    }
                )
        return rows
    finally:
        release_model(model)


def collect_moment_geometry(
    config: ExperimentConfig,
    bundle: DataBundle,
    val_loader,
    seed: int,
) -> list[dict]:
    """Backward-compatible alias for Moment-Guided geometry collection."""

    return collect_attention_geometry(
        config, bundle, val_loader, EXPERIMENT_NAME, seed
    )


@dataclass
class CocoContext:
    ground_truth: COCO
    image_ids: list[int]
    category_ids: list[int]
    large_gt_by_image: dict[int, list[dict]]


def build_coco_context(records: list[dict]) -> CocoContext:
    images, annotations = [], []
    annotation_id = 1
    for record in records:
        image_id = int(record["image_id"])
        images.append(
            {
                "id": image_id,
                "width": int(record["width"]),
                "height": int(record["height"]),
            }
        )
        for box, label in zip(record["boxes_xyxy"], record["labels"]):
            x1, y1, x2, y2 = map(float, box)
            width, height = x2 - x1, y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id + 1, "name": ID2LABEL[class_id]}
            for class_id in sorted(ID2LABEL)
        ],
    }
    ground_truth = COCO()
    ground_truth.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth.createIndex()
    image_ids = [image["id"] for image in images]
    category_ids = [category["id"] for category in dataset["categories"]]
    large_gt_by_image = {
        image_id: [
            annotation
            for annotation in annotations
            if annotation["image_id"] == image_id and annotation["area"] >= 96**2
        ]
        for image_id in image_ids
    }
    return CocoContext(ground_truth, image_ids, category_ids, large_gt_by_image)


def _xywh_to_xyxy(box):
    x, y, width, height = map(float, box)
    return np.array([x, y, x + width, y + height], dtype=np.float64)


def _iou_xyxy(box1, box2):
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])
    wh = np.maximum(right_bottom - left_top, 0)
    intersection = wh[0] * wh[1]
    area1 = np.prod(np.maximum(box1[2:] - box1[:2], 0))
    area2 = np.prod(np.maximum(box2[2:] - box2[:2], 0))
    return intersection / max(area1 + area2 - intersection, 1e-12)


def _coco_results(predictions: pd.DataFrame) -> list[dict]:
    return [
        {
            "image_id": int(row.image_id),
            "category_id": int(row.category_id),
            "bbox": [row.x1, row.y1, row.x2 - row.x1, row.y2 - row.y1],
            "score": float(row.score),
        }
        for row in predictions.itertuples(index=False)
    ]


def large_error_decomposition(
    prediction_df: pd.DataFrame,
    records: list[dict],
    *,
    iou_thresholds: tuple[float, ...] = (0.50, 0.75),
    background_iou: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """COCO evaluator-level Large TP/FP/FN decomposition and PR curves."""

    context = build_coco_context(records)
    all_events, all_metrics, all_pr = [], [], []
    for (experiment, seed), run_predictions in prediction_df.groupby(
        ["experiment", "seed"], sort=True
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            detections = context.ground_truth.loadRes(_coco_results(run_predictions))
        evaluator = COCOeval(context.ground_truth, detections, iouType="bbox")
        evaluator.params.imgIds = context.image_ids
        evaluator.params.catIds = context.category_ids
        evaluator.params.iouThrs = np.asarray(iou_thresholds, dtype=np.float64)
        evaluator.params.maxDets = [1, 10, 100]
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()

        large_index = evaluator.params.areaRngLbl.index("large")
        max_det_index = evaluator.params.maxDets.index(100)
        precision_tensor = evaluator.eval["precision"]
        recall_tensor = evaluator.eval["recall"]

        for threshold_index, threshold in enumerate(iou_thresholds):
            precision_values = precision_tensor[
                threshold_index, :, :, large_index, max_det_index
            ]
            valid_precision = precision_values[precision_values >= 0]
            ap_large = float(valid_precision.mean()) if valid_precision.size else np.nan
            recall_values = recall_tensor[
                threshold_index, :, large_index, max_det_index
            ]
            valid_recall = recall_values[recall_values >= 0]
            coco_recall = float(valid_recall.mean()) if valid_recall.size else np.nan
            for recall_index, recall_threshold in enumerate(evaluator.params.recThrs):
                class_precisions = precision_tensor[
                    threshold_index, recall_index, :, large_index, max_det_index
                ]
                valid = class_precisions[class_precisions >= 0]
                all_pr.append(
                    {
                        "experiment": experiment,
                        "seed": int(seed),
                        "iou_threshold": threshold,
                        "recall_threshold": float(recall_threshold),
                        "precision": float(valid.mean()) if valid.size else np.nan,
                    }
                )

            threshold_events = []
            for eval_image in evaluator.evalImgs:
                if eval_image is None or eval_image["maxDet"] != 100:
                    continue
                if list(eval_image["aRng"]) != list(
                    evaluator.params.areaRng[large_index]
                ):
                    continue
                image_id = int(eval_image["image_id"])
                category_id = int(eval_image["category_id"])
                detection_ids = np.asarray(eval_image["dtIds"], dtype=np.int64)
                gt_ids = np.asarray(eval_image["gtIds"], dtype=np.int64)
                detection_matches = np.asarray(
                    eval_image["dtMatches"][threshold_index]
                )
                gt_matches = np.asarray(eval_image["gtMatches"][threshold_index])
                detection_ignore = np.asarray(
                    eval_image["dtIgnore"][threshold_index], dtype=bool
                )
                gt_ignore = np.asarray(eval_image["gtIgnore"], dtype=bool)
                matched_large_gt_ids = {
                    int(gt_id)
                    for gt_id, matched, ignored in zip(gt_ids, gt_matches, gt_ignore)
                    if matched > 0 and not ignored
                }

                for detection_index, detection_id in enumerate(detection_ids):
                    if detection_ignore[detection_index]:
                        continue
                    detection = detections.anns[int(detection_id)]
                    base_event = {
                        "experiment": experiment,
                        "seed": int(seed),
                        "iou_threshold": threshold,
                        "image_id": image_id,
                        "category_id": category_id,
                        "score": float(detection["score"]),
                        "detection_id": int(detection_id),
                    }
                    if detection_matches[detection_index] > 0:
                        gt_id = int(detection_matches[detection_index])
                        gt = context.ground_truth.anns[gt_id]
                        threshold_events.append(
                            {
                                **base_event,
                                "event_type": "TP",
                                "gt_id": gt_id,
                                "best_iou": _iou_xyxy(
                                    _xywh_to_xyxy(detection["bbox"]),
                                    _xywh_to_xyxy(gt["bbox"]),
                                ),
                            }
                        )
                        continue

                    detection_box = _xywh_to_xyxy(detection["bbox"])
                    overlaps = [
                        (gt, _iou_xyxy(detection_box, _xywh_to_xyxy(gt["bbox"])))
                        for gt in context.large_gt_by_image[image_id]
                    ]
                    same_class = [
                        item for item in overlaps if item[0]["category_id"] == category_id
                    ]
                    wrong_class = [
                        item for item in overlaps if item[0]["category_id"] != category_id
                    ]
                    duplicates = [
                        item
                        for item in same_class
                        if item[1] >= threshold
                        and item[0]["id"] in matched_large_gt_ids
                    ]
                    classifications = [
                        item for item in wrong_class if item[1] >= threshold
                    ]
                    localizations = [
                        item
                        for item in same_class
                        if background_iou <= item[1] < threshold
                    ]
                    if duplicates:
                        event_type, candidate = "Duplicate FP", max(
                            duplicates, key=lambda item: item[1]
                        )
                    elif classifications:
                        event_type, candidate = "Classification FP", max(
                            classifications, key=lambda item: item[1]
                        )
                    elif localizations:
                        event_type, candidate = "Localization FP", max(
                            localizations, key=lambda item: item[1]
                        )
                    else:
                        event_type = "Background FP"
                        candidate = (
                            None,
                            max([overlap for _, overlap in overlaps], default=0.0),
                        )
                    candidate_gt, best_iou = candidate
                    threshold_events.append(
                        {
                            **base_event,
                            "event_type": event_type,
                            "gt_id": (
                                int(candidate_gt["id"])
                                if candidate_gt is not None
                                else np.nan
                            ),
                            "best_iou": float(best_iou),
                        }
                    )

                for gt_id, matched, ignored in zip(gt_ids, gt_matches, gt_ignore):
                    if matched == 0 and not ignored:
                        threshold_events.append(
                            {
                                "experiment": experiment,
                                "seed": int(seed),
                                "iou_threshold": threshold,
                                "image_id": image_id,
                                "category_id": category_id,
                                "event_type": "Missed GT",
                                "score": np.nan,
                                "detection_id": np.nan,
                                "gt_id": int(gt_id),
                                "best_iou": np.nan,
                            }
                        )

            all_events.extend(threshold_events)
            counts = pd.Series(
                [event["event_type"] for event in threshold_events], dtype="object"
            ).value_counts()
            tp = int(counts.get("TP", 0))
            missed = int(counts.get("Missed GT", 0))
            fp = sum(int(counts.get(name, 0)) for name in ERROR_TYPES[1:5])
            all_metrics.append(
                {
                    "experiment": experiment,
                    "seed": int(seed),
                    "iou_threshold": threshold,
                    **{name: int(counts.get(name, 0)) for name in ERROR_TYPES},
                    "Precision_micro": tp / max(tp + fp, 1),
                    "Recall_micro": tp / max(tp + missed, 1),
                    "Recall_COCO_macro": coco_recall,
                    "AP_L": ap_large,
                }
            )
    return (
        pd.DataFrame(all_events),
        pd.DataFrame(all_metrics),
        pd.DataFrame(all_pr),
    )
