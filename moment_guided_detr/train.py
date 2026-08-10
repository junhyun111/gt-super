from __future__ import annotations

import gc
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from gt_aux.config import ExperimentConfig, seed_everything
from gt_aux.data import DataBundle, make_loaders

from .artifacts import checkpoint_path, gradients_path, history_path
from .model import EXPERIMENT_NAME, MomentGuidedDeformableDetr, make_moment_model


@dataclass(frozen=True)
class MomentTrainingSettings:
    """Notebook-facing settings specific to Moment-Guided training."""

    moment_weight: float = 1.0
    center_weight: float = 1.0
    covariance_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


GRADIENT_COLUMNS = [
    "experiment",
    "seed",
    "data_seed",
    "epoch",
    "cosine",
    "main_grad_norm",
    "weighted_moment_grad_norm",
    "norm_ratio",
]


def move_labels_to_device(labels, device):
    return [{key: value.to(device) for key, value in target.items()} for target in labels]


def unique_parameters(parameters):
    seen, result = set(), []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def make_optimizer(model: MomentGuidedDeformableDetr, config: ExperimentConfig):
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if "detector.model.backbone" in name else other).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": unique_parameters(other), "lr": config.lr},
            {"params": unique_parameters(backbone), "lr": config.backbone_lr},
        ],
        weight_decay=config.weight_decay,
    )


def gradient_cosine(main_loss, moment_loss, parameters):
    main_grads = torch.autograd.grad(
        main_loss, parameters, retain_graph=True, allow_unused=True
    )
    moment_grads = torch.autograd.grad(
        moment_loss, parameters, retain_graph=True, allow_unused=True
    )
    paired = [
        (main, moment)
        for main, moment in zip(main_grads, moment_grads)
        if main is not None and moment is not None
    ]
    if not paired:
        return float("nan"), float("nan"), float("nan")
    main_vector = torch.cat([main.detach().float().flatten() for main, _ in paired])
    moment_vector = torch.cat(
        [moment.detach().float().flatten() for _, moment in paired]
    )
    main_norm, moment_norm = main_vector.norm(), moment_vector.norm()
    cosine = F.cosine_similarity(main_vector, moment_vector, dim=0)
    return float(cosine), float(main_norm), float(moment_norm)


@torch.inference_mode()
def evaluate_main(model, val_loader, processor, config: ExperimentConfig):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    calls_before = model.moment_forward_calls
    start = time.perf_counter()
    for batch in tqdm(
        val_loader, desc="main-only validation", leave=False, mininterval=0.5
    ):
        pixel_values = batch["pixel_values"].to(config.device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(config.device, non_blocking=True)
        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=None)
        if result["moment_executed"]:
            raise AssertionError("Moment path ran during main-only inference")
        target_sizes = torch.stack(
            [target["orig_size"] for target in batch["eval_targets"]]
        ).to(config.device)
        predictions = processor.post_process_object_detection(
            result["outputs"], threshold=0.0, target_sizes=target_sizes
        )
        predictions = [
            {key: value.detach().cpu() for key, value in prediction.items()}
            for prediction in predictions
        ]
        targets = [
            {"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()}
            for target in batch["eval_targets"]
        ]
        metric.update(predictions, targets)
    if model.moment_forward_calls != calls_before:
        raise AssertionError("Moment path changed during main-only validation")
    values = metric.compute()
    return {
        "map": float(values["map"]),
        "map50": float(values["map_50"]),
        "map75": float(values["map_75"]),
        "mar100": float(values["mar_100"]),
        "val_seconds": time.perf_counter() - start,
    }


def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    config,
    settings,
    seed,
    epoch,
    elapsed_train,
    history,
    gradients,
):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.as_dict(),
        "moment_settings": settings.as_dict(),
        "experiment": EXPERIMENT_NAME,
        "seed": seed,
        "epoch": epoch,
        "elapsed_train": elapsed_train,
        "history": history.to_dict("records"),
        "gradients": gradients.to_dict("records"),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "loader_generator_state": train_loader.generator.get_state(),
    }
    path = checkpoint_path(config, seed)
    torch.save(state, path)
    if config.save_epoch_checkpoints:
        torch.save(state, path.with_name(path.stem + f"_epoch{epoch}.pt"))


def _load_resume_checkpoint(
    resume_from,
    model,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    config,
    settings,
    seed,
):
    path = Path(resume_from).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(f"Not a {EXPERIMENT_NAME!r} checkpoint: {path}")
    if int(checkpoint.get("seed")) != int(seed):
        raise ValueError(f"Checkpoint seed={checkpoint.get('seed')}, expected {seed}")
    if checkpoint.get("moment_settings") != settings.as_dict():
        raise ValueError(
            f"Moment settings differ: saved={checkpoint.get('moment_settings')}, "
            f"current={settings.as_dict()}"
        )
    saved_config = checkpoint.get("config", {})
    current_config = config.as_dict()
    compatibility_keys = {
        "run_mode",
        "checkpoint",
        "data_seed",
        "train_images",
        "val_images",
        "epochs",
        "batch_size",
        "image_size",
        "lr",
        "backbone_lr",
        "weight_decay",
        "grad_clip",
        "horizontal_flip_p",
        "use_amp",
        "deterministic",
    }
    mismatches = {
        key: (saved_config.get(key), current_config.get(key))
        for key in compatibility_keys
        if saved_config.get(key) != current_config.get(key)
    }
    if mismatches:
        raise ValueError(f"Resume config mismatch: {mismatches}")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    train_loader.generator.set_state(checkpoint["loader_generator_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint, path


def train_moment_experiment(
    config: ExperimentConfig,
    bundle: DataBundle,
    *,
    seed: int | None = None,
    settings: MomentTrainingSettings | None = None,
    save_checkpoint: bool = True,
    resume_from: str | Path | None = None,
):
    seed = config.seed if seed is None else seed
    settings = settings or MomentTrainingSettings()
    print(f"\n===== {EXPERIMENT_NAME} / seed={seed} =====")
    seed_everything(seed, deterministic=config.deterministic)
    train_loader, val_loader = make_loaders(config, bundle, seed)
    model, fingerprint = make_moment_model(
        config,
        seed,
        center_weight=settings.center_weight,
        covariance_weight=settings.covariance_weight,
    )
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
    history, gradient_history, elapsed_train, start_epoch = [], [], 0.0, 1

    if resume_from is not None:
        checkpoint, resume_path = _load_resume_checkpoint(
            resume_from,
            model,
            optimizer,
            scheduler,
            scaler,
            train_loader,
            config,
            settings,
            seed,
        )
        history = list(checkpoint["history"])
        gradient_history = list(checkpoint["gradients"])
        elapsed_train = float(checkpoint["elapsed_train"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[resume] loaded {resume_path}; continuing at epoch {start_epoch}")
    else:
        initial_metrics = evaluate_main(model, val_loader, bundle.processor, config)
        history.append(
            {
                "experiment": EXPERIMENT_NAME,
                "seed": seed,
                "data_seed": config.data_seed,
                "epoch": 0,
                "model_fingerprint": fingerprint,
                "train_seconds": 0.0,
                "main_loss": np.nan,
                "main_bbox_loss": np.nan,
                "main_giou_loss": np.nan,
                "moment_loss": np.nan,
                "center_loss": np.nan,
                "covariance_loss": np.nan,
                "matched_queries_per_batch": np.nan,
                "valid_attention_mass": np.nan,
                "off_diagonal_abs": np.nan,
                "predicted_variance": np.nan,
                "target_variance": np.nan,
                "moment_weight": settings.moment_weight,
                **initial_metrics,
            }
        )

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"[phase] training epoch {epoch}/{config.epochs}: {len(train_loader)} batches")
        model.train()
        sums = Counter()
        epoch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"{EXPERIMENT_NAME} e{epoch}", leave=False)
        ):
            pixel_values = batch["pixel_values"].to(config.device, non_blocking=True)
            pixel_mask = batch["pixel_mask"].to(config.device, non_blocking=True)
            labels = move_labels_to_device(batch["labels"], config.device)
            with torch.autocast(
                device_type=config.device.type,
                dtype=torch.float16,
                enabled=config.use_amp,
            ):
                result = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels,
                    moment_weight=settings.moment_weight,
                )

            if not result["moment_executed"]:
                raise AssertionError("Moment loss did not run during training")
            if step == 0:
                attention = model.final_cross_attention
                parameters = unique_parameters(
                    list(attention.sampling_offsets.parameters())
                    + list(attention.attention_weights.parameters())
                )
                cosine, main_norm, moment_norm = gradient_cosine(
                    result["main_loss"],
                    settings.moment_weight * result["moment_loss"],
                    parameters,
                )
                gradient_history.append(
                    {
                        "experiment": EXPERIMENT_NAME,
                        "seed": seed,
                        "data_seed": config.data_seed,
                        "epoch": epoch,
                        "cosine": cosine,
                        "main_grad_norm": main_norm,
                        "weighted_moment_grad_norm": moment_norm,
                        "norm_ratio": moment_norm / max(main_norm, 1e-12),
                    }
                )

            scaler.scale(result["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss_dict = result["outputs"].loss_dict or {}
            sums["batches"] += 1
            sums["main_loss"] += float(result["main_loss"].detach())
            sums["main_bbox_loss"] += float(loss_dict["loss_bbox"].detach())
            sums["main_giou_loss"] += float(loss_dict["loss_giou"].detach())
            sums["moment_loss"] += float(result["moment_loss"].detach())
            sums["center_loss"] += float(result["center_loss"].detach())
            sums["covariance_loss"] += float(result["covariance_loss"].detach())
            sums["matched_queries"] += result["matched_queries"]
            for key, value in result["moment_stats"].items():
                if key != "matched_queries":
                    sums[key] += value

        scheduler.step()
        elapsed_train += time.perf_counter() - epoch_start
        metrics = evaluate_main(model, val_loader, bundle.processor, config)
        batches = max(int(sums["batches"]), 1)
        row = {
            "experiment": EXPERIMENT_NAME,
            "seed": seed,
            "data_seed": config.data_seed,
            "epoch": epoch,
            "model_fingerprint": fingerprint,
            "train_seconds": elapsed_train,
            "main_loss": sums["main_loss"] / batches,
            "main_bbox_loss": sums["main_bbox_loss"] / batches,
            "main_giou_loss": sums["main_giou_loss"] / batches,
            "moment_loss": sums["moment_loss"] / batches,
            "center_loss": sums["center_loss"] / batches,
            "covariance_loss": sums["covariance_loss"] / batches,
            "matched_queries_per_batch": sums["matched_queries"] / batches,
            "valid_attention_mass": sums["valid_attention_mass"] / batches,
            "off_diagonal_abs": sums["off_diagonal_abs"] / batches,
            "predicted_variance": sums["predicted_variance"] / batches,
            "target_variance": sums["target_variance"] / batches,
            "moment_weight": settings.moment_weight,
            **metrics,
        }
        history.append(row)
        print(
            {
                key: round(value, 4)
                for key, value in row.items()
                if key
                in {
                    "epoch",
                    "main_loss",
                    "moment_loss",
                    "center_loss",
                    "covariance_loss",
                    "map",
                    "map50",
                    "map75",
                }
            }
        )
        history_df = pd.DataFrame(history)
        gradients_df = pd.DataFrame(gradient_history, columns=GRADIENT_COLUMNS)
        history_df.to_csv(history_path(config, seed), index=False)
        gradients_df.to_csv(gradients_path(config, seed), index=False)
        if save_checkpoint:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                train_loader,
                config,
                settings,
                seed,
                epoch,
                elapsed_train,
                history_df,
                gradients_df,
            )

    return model, pd.DataFrame(history), pd.DataFrame(
        gradient_history, columns=GRADIENT_COLUMNS
    )


def release_model(model):
    if model is None:
        return None
    model.close()
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None
