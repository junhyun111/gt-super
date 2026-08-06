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

from .config import ID2LABEL, ExperimentConfig
from .data import DataBundle, VOCDataset, collate_detection_batch, make_loaders
from .model import make_model


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
        metric.update(predictions, targets)
    assert model.aux_forward_calls == calls_before
    values = metric.compute()
    return {
        "map": float(values["map"]), "map50": float(values["map_50"]),
        "map75": float(values["map_75"]), "mar100": float(values["mar_100"]),
        "val_seconds": time.perf_counter() - start,
    }


def load_checkpoint(config: ExperimentConfig, bundle: DataBundle, experiment: str, seed: int | None = None):
    seed = config.seed if seed is None else seed
    model, fingerprint = make_model(config, experiment, seed)
    checkpoint_path = config.checkpoint_path(experiment, seed)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
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
            draw.text((coords[0], coords[1]), f"{ID2LABEL[int(label)]} {float(score):.2f}", fill="orange")
        axis.imshow(image)
        axis.set_title("green=GT, orange=main query")
        axis.axis("off")
    plt.tight_layout()
    path = config.output_dir / f"main_predictions_{config.run_mode}_{experiment}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    model.close()
    return fig
