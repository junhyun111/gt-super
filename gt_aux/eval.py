from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.loss.loss_deformable_detr import DeformableDetrHungarianMatcher

from .config import ID2LABEL, ExperimentConfig
from .data import DataBundle, VOCDataset, collate_detection_batch, make_loaders
from .model import box_iou, make_model


def class_agnostic_items(predictions, targets):
    """Remove semantic labels while preserving boxes and ranking scores."""
    agnostic_predictions = [
        {
            "boxes": prediction["boxes"],
            "scores": prediction["scores"],
            "labels": torch.zeros_like(prediction["labels"]),
        }
        for prediction in predictions
    ]
    agnostic_targets = [
        {
            "boxes": target["boxes"],
            "labels": torch.zeros_like(target["labels"]),
        }
        for target in targets
    ]
    return agnostic_predictions, agnostic_targets


@torch.inference_mode()
def evaluate_main(model, val_loader, processor, config: ExperimentConfig):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    calls_before = model.aux_forward_calls
    start = time.perf_counter()
    for batch in tqdm(
        val_loader,
        desc="main-only validation",
        leave=False,
        mininterval=0.5,
    ):
        pixel_values = batch["pixel_values"].to(config.device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(config.device, non_blocking=True)
        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=None)
        assert result["aux_executed"] is False
        target_sizes = torch.stack([target["orig_size"] for target in batch["eval_targets"]]).to(config.device)
        predictions = processor.post_process_object_detection(
            result["outputs"], threshold=0.0, target_sizes=target_sizes
        )
        predictions = [{key: value.detach().cpu() for key, value in prediction.items()}
                       for prediction in predictions]
        targets = [{"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()}
                   for target in batch["eval_targets"]]
        if model.is_localization_only:
            predictions, targets = class_agnostic_items(predictions, targets)
        metric.update(predictions, targets)
    assert model.aux_forward_calls == calls_before
    values = metric.compute()
    return {
        "map": float(values["map"]), "map50": float(values["map_50"]),
        "map75": float(values["map_75"]), "mar100": float(values["mar_100"]),
        "val_seconds": time.perf_counter() - start,
    }


@torch.inference_mode()
def collect_localization_predictions(model, val_loader, processor, config: ExperimentConfig):
    """Collect main-path predictions and class-agnostic targets for localization evaluation."""
    model.eval()
    predictions, targets = [], []
    calls_before = model.aux_forward_calls
    for batch in tqdm(val_loader, desc="class-agnostic localization", leave=False, mininterval=0.5):
        result = model(
            pixel_values=batch["pixel_values"].to(config.device, non_blocking=True),
            pixel_mask=batch["pixel_mask"].to(config.device, non_blocking=True),
            labels=None,
        )
        assert result["aux_executed"] is False
        target_sizes = torch.stack([
            target["orig_size"] for target in batch["eval_targets"]
        ]).to(config.device)
        batch_predictions = processor.post_process_object_detection(
            result["outputs"], threshold=0.0, target_sizes=target_sizes
        )
        batch_predictions = [
            {key: value.detach().cpu() for key, value in prediction.items()}
            for prediction in batch_predictions
        ]
        batch_targets = [
            {
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
                "orig_size": target["orig_size"].cpu(),
                "image_id": target["image_id"],
            }
            for target in batch["eval_targets"]
        ]
        agnostic_predictions, agnostic_targets = class_agnostic_items(
            batch_predictions, batch_targets
        )
        predictions.extend(agnostic_predictions)
        for agnostic, original in zip(agnostic_targets, batch_targets):
            agnostic["orig_size"] = original["orig_size"]
            agnostic["image_id"] = original["image_id"]
            targets.append(agnostic)
    assert model.aux_forward_calls == calls_before
    return predictions, targets


def localization_metrics(predictions, targets):
    """Compute ranking-aware class-agnostic COCO localization metrics."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(
        predictions,
        [{"boxes": target["boxes"], "labels": target["labels"]} for target in targets],
    )
    values = metric.compute()
    keys = {
        "loc_map": "map", "loc_ap50": "map_50", "loc_ap75": "map_75",
        "loc_ap_small": "map_small", "loc_ap_medium": "map_medium",
        "loc_ap_large": "map_large", "loc_ar100": "mar_100",
    }
    result = {}
    for name, source in keys.items():
        value = float(values[source])
        result[name] = value if value >= 0 else float("nan")
    return result


def localization_geometry_metrics(predictions, targets, max_detections: int = 100):
    """Measure one-to-one box geometry, independent of semantic correctness.

    Predictions are first limited by score, then greedily matched by descending
    IoU. This complements AP by exposing center/size quality directly.
    """
    matched_ious, center_errors, size_errors = [], [], []
    gt_count = 0
    for prediction, target in zip(predictions, targets):
        gt_boxes = target["boxes"].float()
        gt_count += len(gt_boxes)
        order = prediction["scores"].argsort(descending=True)[:max_detections]
        pred_boxes = prediction["boxes"][order].float()
        if not len(gt_boxes) or not len(pred_boxes):
            continue
        pairwise_iou = box_iou(pred_boxes, gt_boxes)[0]
        available_predictions = torch.ones(len(pred_boxes), dtype=torch.bool)
        available_targets = torch.ones(len(gt_boxes), dtype=torch.bool)
        pairs = []
        for _ in range(min(len(pred_boxes), len(gt_boxes))):
            candidate = pairwise_iou.clone()
            candidate[~available_predictions] = -1
            candidate[:, ~available_targets] = -1
            flat_index = int(candidate.argmax())
            pred_index = flat_index // len(gt_boxes)
            target_index = flat_index % len(gt_boxes)
            if candidate[pred_index, target_index] < 0:
                break
            pairs.append((pred_index, target_index))
            available_predictions[pred_index] = False
            available_targets[target_index] = False
        if not pairs:
            continue
        pred_indices = torch.tensor([pair[0] for pair in pairs])
        target_indices = torch.tensor([pair[1] for pair in pairs])
        matched_pred = pred_boxes[pred_indices]
        matched_gt = gt_boxes[target_indices]
        ious = pairwise_iou[pred_indices, target_indices]
        height, width = target["orig_size"].float()
        scale = gt_boxes.new_tensor([width, height])
        pred_centers = (matched_pred[:, :2] + matched_pred[:, 2:]) / 2 / scale
        gt_centers = (matched_gt[:, :2] + matched_gt[:, 2:]) / 2 / scale
        pred_sizes = (matched_pred[:, 2:] - matched_pred[:, :2]) / scale
        gt_sizes = (matched_gt[:, 2:] - matched_gt[:, :2]) / scale
        matched_ious.extend(ious.tolist())
        center_errors.extend((pred_centers - gt_centers).norm(dim=-1).tolist())
        size_errors.extend((pred_sizes - gt_sizes).abs().sum(dim=-1).tolist())
    matched = len(matched_ious)
    iou_tensor = torch.tensor(matched_ious) if matched else torch.empty(0)
    return {
        "matched_iou": float(iou_tensor.mean()) if matched else float("nan"),
        "center_error_l2": float(torch.tensor(center_errors).mean()) if matched else float("nan"),
        "size_error_l1": float(torch.tensor(size_errors).mean()) if matched else float("nan"),
        "geometry_recall50": float((iou_tensor >= 0.50).sum()) / max(gt_count, 1),
        "geometry_recall75": float((iou_tensor >= 0.75).sum()) / max(gt_count, 1),
        "matched_boxes": matched,
        "gt_boxes": gt_count,
    }


@torch.inference_mode()
def classification_diagnostics(
    model,
    val_loader,
    processor,
    config: ExperimentConfig,
    *,
    score_threshold: float = 0.1,
    iou_threshold: float = 0.5,
):
    """Measure semantic behavior separately from box-localization quality.

    Matched-query accuracy uses the same Hungarian cost as model training.
    Classification false positives are score-filtered detections whose closest
    ground-truth box overlaps at ``iou_threshold`` but has another class.
    """
    if model.is_localization_only:
        raise ValueError("Classification diagnostics require a semantic detector")
    matcher = DeformableDetrHungarianMatcher(
        class_cost=model.detector.config.class_cost,
        bbox_cost=model.detector.config.bbox_cost,
        giou_cost=model.detector.config.giou_cost,
    )
    model.eval()
    calls_before = model.aux_forward_calls
    loss_sum = 0.0
    loss_batches = matched_count = matched_correct = 0
    matched_gt_score_sum = 0.0
    detection_count = classification_fp = background_fp = image_count = 0

    for batch in tqdm(
        val_loader, desc="classification diagnostics", leave=False, mininterval=0.5
    ):
        labels = [
            {key: value.to(config.device) for key, value in target.items()}
            for target in batch["labels"]
        ]
        result = model(
            pixel_values=batch["pixel_values"].to(config.device, non_blocking=True),
            pixel_mask=batch["pixel_mask"].to(config.device, non_blocking=True),
            labels=labels,
            aux_weight=0.0,
        )
        outputs = result["outputs"]
        loss_sum += float(outputs.loss_dict["loss_ce"].detach())
        loss_batches += 1
        assignments = matcher(
            {"logits": outputs.logits, "pred_boxes": outputs.pred_boxes}, labels
        )
        probabilities = outputs.logits.sigmoid()
        for batch_index, (query_indices, target_indices) in enumerate(assignments):
            if not len(query_indices):
                continue
            query_indices = query_indices.to(probabilities.device)
            target_indices = target_indices.to(probabilities.device)
            matched_probabilities = probabilities[batch_index, query_indices]
            matched_targets = labels[batch_index]["class_labels"][target_indices]
            matched_predictions = matched_probabilities.argmax(dim=-1)
            matched_correct += int((matched_predictions == matched_targets).sum())
            matched_count += int(len(query_indices))
            matched_gt_score_sum += float(
                matched_probabilities.gather(1, matched_targets[:, None]).sum()
            )

        target_sizes = torch.stack([
            target["orig_size"] for target in batch["eval_targets"]
        ]).to(config.device)
        predictions = processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )
        for prediction, target in zip(predictions, batch["eval_targets"]):
            image_count += 1
            predicted_boxes = prediction["boxes"].detach().cpu()
            predicted_labels = prediction["labels"].detach().cpu()
            target_boxes = target["boxes"].cpu()
            target_labels = target["labels"].cpu()
            detection_count += len(predicted_boxes)
            if not len(predicted_boxes):
                continue
            if not len(target_boxes):
                background_fp += len(predicted_boxes)
                continue
            overlaps = box_iou(predicted_boxes, target_boxes)[0]
            best_iou, best_target = overlaps.max(dim=1)
            foreground = best_iou >= iou_threshold
            classification_fp += int(
                (
                    foreground
                    & (predicted_labels != target_labels[best_target])
                ).sum()
            )
            background_fp += int((~foreground).sum())

    assert model.aux_forward_calls == calls_before
    return {
        "mean_cls_loss": loss_sum / max(loss_batches, 1),
        "matched_query_class_accuracy": matched_correct / max(matched_count, 1),
        "matched_query_mean_gt_score": matched_gt_score_sum / max(matched_count, 1),
        "matched_queries": matched_count,
        "classification_fp_count": classification_fp,
        "classification_fp_per_image": classification_fp / max(image_count, 1),
        "classification_fp_rate": classification_fp / max(detection_count, 1),
        "background_fp_count": background_fp,
        "background_fp_per_image": background_fp / max(image_count, 1),
        "detections_at_threshold": detection_count,
        "classification_score_threshold": score_threshold,
        "classification_iou_threshold": iou_threshold,
    }


def load_checkpoint(config: ExperimentConfig, bundle: DataBundle, experiment: str, seed: int | None = None):
    seed = config.seed if seed is None else seed
    model, fingerprint = make_model(config, experiment, seed)
    checkpoint_path = config.checkpoint_path(experiment, seed)
    # Evaluation only needs model weights.  Keep optimizer/scheduler tensors in
    # the returned checkpoint on CPU instead of spending GPU memory on them.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    assert checkpoint.get("experiment") == experiment
    assert checkpoint.get("model_state_dict")
    return model, checkpoint, fingerprint


def evaluate_saved_experiment(config: ExperimentConfig, bundle: DataBundle, experiment: str, seed: int | None = None):
    _, val_loader = make_loaders(config, bundle, config.seed if seed is None else seed)
    model, checkpoint, fingerprint = load_checkpoint(config, bundle, experiment, seed)
    metrics = evaluate_main(model, val_loader, bundle.processor, config)
    model.close()
    del model
    return {"experiment": experiment, "seed": config.seed if seed is None else seed,
            "fingerprint": fingerprint, "checkpoint_epoch": checkpoint.get("epoch"), **metrics}


def load_histories(config: ExperimentConfig, experiments: list[str] | None = None, seed: int | None = None):
    experiments = config.experiments if experiments is None else experiments
    frames = []
    for experiment in experiments:
        path = config.history_path(experiment, seed)
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_histories(config: ExperimentConfig, history_df: pd.DataFrame):
    if history_df.empty:
        return pd.DataFrame()
    final_epoch = history_df["epoch"].max()
    final_rows = history_df[history_df["epoch"] == final_epoch]
    summary = final_rows.groupby("experiment").agg(
        map_mean=("map", "mean"), map_sd=("map", "std"),
        map50_mean=("map50", "mean"), map50_sd=("map50", "std"),
        map75_mean=("map75", "mean"), train_seconds_mean=("train_seconds", "mean"),
        aux_coverage_mean=("aux_coverage", "mean"),
        collision_rate_mean=("collision_rate", "mean"),
    ).sort_values("map50_mean", ascending=False)
    summary.to_csv(config.output_dir / f"summary_{config.run_mode}.csv")
    return summary


def plot_histories(config: ExperimentConfig, history_df: pd.DataFrame):
    if history_df.empty:
        return None
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.lineplot(data=history_df, x="epoch", y="map", hue="experiment", marker="o", ax=axes[0])
    axes[0].set_title("Main-only validation mAP")
    sns.lineplot(data=history_df, x="epoch", y="map50", hue="experiment", marker="o", ax=axes[1])
    axes[1].set_title("Main-only validation AP@0.5")
    trained_rows = history_df[history_df["epoch"] > 0]
    sns.lineplot(data=trained_rows, x="epoch", y="main_giou_loss", hue="experiment", marker="o", ax=axes[2])
    axes[2].set_title("Main Hungarian GIoU loss")
    for axis in axes:
        axis.axvline(2, color="black", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = config.output_dir / f"main_convergence_{config.run_mode}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_gradient_history(config: ExperimentConfig, gradient_df: pd.DataFrame):
    if gradient_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=gradient_df, x="experiment", y="cosine", hue="epoch", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Main vs auxiliary gradient cosine")
    sns.barplot(data=gradient_df, x="experiment", y="norm_ratio", hue="epoch", ax=axes[1])
    axes[1].set_title("Weighted auxiliary/main gradient norm")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path = config.output_dir / f"gradient_conflict_{config.run_mode}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


@torch.inference_mode()
def visualize_main_predictions(config: ExperimentConfig, bundle: DataBundle, experiment: str,
                               count: int = 4, threshold: float = 0.4, seed: int | None = None):
    seed = config.seed if seed is None else seed
    model, _, _ = load_checkpoint(config, bundle, experiment, seed)
    dataset = VOCDataset(bundle.val_records[:count], config, training=False)
    collate = lambda batch: collate_detection_batch(batch, bundle.processor, config)
    loader = DataLoader(dataset, batch_size=count, shuffle=False, num_workers=0, collate_fn=collate)
    batch = next(iter(loader))
    result = model(
        pixel_values=batch["pixel_values"].to(config.device),
        pixel_mask=batch["pixel_mask"].to(config.device), labels=None,
    )
    target_sizes = torch.stack([target["orig_size"] for target in batch["eval_targets"]]).to(config.device)
    predictions = bundle.processor.post_process_object_detection(
        result["outputs"], threshold=threshold, target_sizes=target_sizes
    )
    fig, axes = plt.subplots(1, count, figsize=(6 * count, 6))
    axes = np.atleast_1d(axes)
    for axis, record, prediction in zip(axes, bundle.val_records[:count], predictions):
        image = Image.open(record["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in record["boxes_xyxy"]:
            draw.rectangle(box, outline="lime", width=3)
        for score, label, box in zip(prediction["scores"].cpu(), prediction["labels"].cpu(), prediction["boxes"].cpu()):
            coords = box.tolist()
            draw.rectangle(coords, outline="orange", width=2)
            label_name = "object" if model.is_localization_only else ID2LABEL[int(label)]
            draw.text((coords[0], coords[1]), f"{label_name} {float(score):.2f}", fill="orange")
        axis.imshow(image)
        axis.set_title("green=GT, orange=main query")
        axis.axis("off")
    plt.tight_layout()
    path = config.output_dir / f"main_predictions_{config.run_mode}_{experiment}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    model.close()
    return fig
