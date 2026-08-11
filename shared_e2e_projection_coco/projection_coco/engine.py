from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from .config import EXPERIMENT_NAME, TrainConfig, seed_everything
from .data import DataBundle, make_train_loader, make_val_loader
from .distributed import DistributedContext, reduce_sums
from .evaluate import evaluate_main
from .model import make_model


GRADIENT_COLUMNS = [
    "experiment",
    "seed",
    "epoch",
    "step",
    "rank",
    "projection_scope",
    "projection_vector_numel",
    "projection_vector_mb",
    "enc_cls_aux_cosine_raw",
    "enc_cls_aux_dot_raw",
    "enc_cls_aux_dot_projected",
    "enc_cls_grad_norm",
    "enc_aux_grad_norm",
    "enc_aux_grad_norm_projected",
    "projection_applied",
    "projection_removed_ratio",
    "aux_weight",
    "grad_scale",
    "optimizer_step_skipped",
]


def move_labels_to_device(labels, device: torch.device):
    return [
        {key: value.to(device, non_blocking=True) for key, value in target.items()}
        for target in labels
    ]


def unique_parameters(parameters):
    seen = set()
    result = []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def make_optimizer(model, config: TrainConfig):
    backbone = []
    other = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = backbone if "detector.model.backbone" in name else other
        target.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": unique_parameters(other), "lr": config.lr},
            {"params": unique_parameters(backbone), "lr": config.backbone_lr},
        ],
        weight_decay=config.weight_decay,
    )


def project_conflicting_gradient(
    classification_gradient: torch.Tensor,
    auxiliary_gradient: torch.Tensor,
    epsilon: float = 1e-12,
):
    cls_float = classification_gradient.detach().float()
    aux_float = auxiliary_gradient.detach().float()
    dot = (cls_float * aux_float).sum()
    cls_norm_squared = cls_float.square().sum()
    cls_norm = cls_norm_squared.sqrt()
    aux_norm = aux_float.square().sum().sqrt()
    cosine = dot / (cls_norm * aux_norm + epsilon)
    applied = bool((dot < 0).item())
    if applied:
        coefficient = dot / (cls_norm_squared + epsilon)
        projected = auxiliary_gradient - coefficient.to(
            auxiliary_gradient.dtype
        ) * classification_gradient
    else:
        projected = auxiliary_gradient
    projected_float = projected.detach().float()
    projected_norm = projected_float.square().sum().sqrt()
    projected_dot = (cls_float * projected_float).sum()
    removed_ratio = 1.0 - projected_norm / (aux_norm + epsilon)
    return projected, {
        "enc_cls_aux_cosine_raw": float(cosine),
        "enc_cls_aux_dot_raw": float(dot),
        "enc_cls_aux_dot_projected": float(projected_dot),
        "enc_cls_grad_norm": float(cls_norm),
        "enc_aux_grad_norm": float(aux_norm),
        "enc_aux_grad_norm_projected": float(projected_norm),
        "projection_applied": applied,
        "projection_removed_ratio": float(removed_ratio),
    }


def representation_projected_gradients(model, result):
    loss_dict = result["outputs"].loss_dict or {}
    if "loss_ce" not in loss_dict:
        raise KeyError("Projection requires outputs.loss_dict['loss_ce']")
    if not result["aux_executed"] or result["aux_loss"] is None:
        raise RuntimeError("Projection requires an executed auxiliary branch")
    representation = model.encoder_representation
    classification_gradient = torch.autograd.grad(
        loss_dict["loss_ce"], representation, retain_graph=True, allow_unused=False
    )[0]
    auxiliary_gradient = torch.autograd.grad(
        result["aux_loss"], representation, retain_graph=True, allow_unused=False
    )[0]
    projected_gradient, stats = project_conflicting_gradient(
        classification_gradient, auxiliary_gradient
    )
    return representation, auxiliary_gradient, projected_gradient, stats


def register_representation_gradient_correction(
    representation: torch.Tensor,
    raw_auxiliary_gradient: torch.Tensor,
    projected_auxiliary_gradient: torch.Tensor,
    *,
    auxiliary_weight: float,
    grad_scale: float,
):
    if projected_auxiliary_gradient is raw_auxiliary_gradient:
        return None
    correction = (
        (projected_auxiliary_gradient - raw_auxiliary_gradient).detach()
        * float(auxiliary_weight)
        * float(grad_scale)
    )

    def add_correction(total_gradient):
        return total_gradient + correction.to(dtype=total_gradient.dtype)

    return representation.register_hook(add_correction)


def _atomic_torch_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _checkpoint_state(
    model,
    optimizer,
    scheduler,
    scaler,
    config: TrainConfig,
    context: DistributedContext,
    epoch: int,
    elapsed_train: float,
    history: list[dict],
    gradients: list[dict],
) -> dict:
    return {
        "format_version": 1,
        "experiment": EXPERIMENT_NAME,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.as_dict(),
        "world_size": context.world_size,
        "epoch": epoch,
        "elapsed_train": elapsed_train,
        "history": history,
        "gradients": gradients,
    }


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    config: TrainConfig,
    context: DistributedContext,
    epoch: int,
    elapsed_train: float,
    history: list[dict],
    gradients: list[dict],
) -> None:
    context.barrier()
    if context.is_main:
        state = _checkpoint_state(
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            context,
            epoch,
            elapsed_train,
            history,
            gradients,
        )
        _atomic_torch_save(state, config.latest_checkpoint)
        if epoch % config.save_every == 0 or epoch == config.epochs:
            snapshot = config.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            _atomic_torch_save(state, snapshot)
    context.barrier()


def resolve_resume_path(resume_from: str | Path | None, config: TrainConfig):
    if resume_from is None:
        return None
    if str(resume_from).lower() == "auto":
        path = config.latest_checkpoint
        return path if path.is_file() else None
    return Path(resume_from).expanduser().resolve()


def load_checkpoint(
    resume_from: str | Path | None,
    model,
    optimizer,
    scheduler,
    scaler,
    config: TrainConfig,
    context: DistributedContext,
):
    path = resolve_resume_path(resume_from, config)
    if path is None:
        return 1, 0.0, [], [], None
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(
            f"Checkpoint experiment={checkpoint.get('experiment')!r}, "
            f"expected {EXPERIMENT_NAME!r}"
        )
    if int(checkpoint.get("world_size", 1)) != context.world_size:
        raise ValueError(
            "Resume must use the same world size. "
            f"Saved={checkpoint.get('world_size', 1)}, current={context.world_size}"
        )
    saved = checkpoint.get("config", {})
    current = config.as_dict()
    compatibility_keys = {
        "model_name",
        "epochs",
        "batch_size",
        "image_min_size",
        "image_max_size",
        "lr",
        "backbone_lr",
        "weight_decay",
        "grad_clip",
        "aux_weight",
        "feature_level",
        "horizontal_flip_p",
        "seed",
        "train_limit",
        "val_limit",
        "amp",
        "disable_custom_kernels",
    }
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in compatibility_keys
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}: saved={old!r}, current={new!r}"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise ValueError(f"Resume config mismatch: {details}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    seed_everything(
        config.seed + context.rank + start_epoch * 100_003,
        config.deterministic,
    )
    return (
        start_epoch,
        float(checkpoint.get("elapsed_train", 0.0)),
        list(checkpoint.get("history", [])),
        list(checkpoint.get("gradients", [])),
        path,
    )


def _empty_epoch_sums() -> Counter:
    keys = [
        "batches",
        "total_loss",
        "main_loss",
        "main_cls_loss",
        "main_bbox_loss",
        "main_giou_loss",
        "aux_loss",
        "aux_l1",
        "aux_giou",
        "aux_batches",
        "aux_weight",
        "total_objects",
        "used_objects",
        "collision_targets",
        "raw_feature_norm",
        "adapted_feature_norm",
        "decoder_feature_norm",
        "projection_steps",
        "projection_applied",
        "enc_cls_aux_cosine_raw",
        "projection_removed_ratio",
        "optimizer_steps",
        "optimizer_steps_skipped",
    ]
    return Counter({key: 0.0 for key in keys})


def train(
    config: TrainConfig,
    bundle: DataBundle,
    context: DistributedContext,
    *,
    resume_from: str | Path | None = None,
) -> None:
    config.create_output_dirs()
    seed_everything(config.seed + context.rank, config.deterministic)
    train_loader, train_sampler = make_train_loader(config, bundle, context)
    val_loader = make_val_loader(config, bundle, context) if context.is_main else None
    base_model, fingerprint = make_model(config, context.device)
    max_category = max(bundle.category_ids)
    if max_category >= base_model.detector.config.num_labels:
        raise ValueError(
            f"COCO category ID {max_category} does not fit model head with "
            f"{base_model.detector.config.num_labels} labels"
        )
    optimizer = make_optimizer(base_model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp and context.device.type == "cuda"
    )
    start_epoch, elapsed_train, history, gradient_history, resume_path = load_checkpoint(
        resume_from,
        base_model,
        optimizer,
        scheduler,
        scaler,
        config,
        context,
    )
    training_model = base_model
    if context.distributed:
        training_model = DistributedDataParallel(
            base_model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    if context.is_main:
        print(
            f"[data] train={len(bundle.train_dataset)}, val={len(bundle.val_dataset)}, "
            f"categories={len(bundle.category_ids)}"
        )
        print(
            f"[runtime] world_size={context.world_size}, "
            f"per_gpu_batch={config.batch_size}, "
            f"global_batch={config.batch_size * context.world_size}"
        )
        if resume_path is not None:
            print(f"[resume] {resume_path} -> epoch {start_epoch}")

    if resume_path is None and not config.skip_initial_eval:
        if context.is_main:
            print("[phase] initial main-only COCO validation")
            metrics = evaluate_main(base_model, val_loader, bundle.processor, context.device)
            history.append(
                {
                    "experiment": EXPERIMENT_NAME,
                    "seed": config.seed,
                    "epoch": 0,
                    "model_fingerprint": fingerprint,
                    "train_seconds": 0.0,
                    "total_loss": np.nan,
                    "main_loss": np.nan,
                    "main_cls_loss": np.nan,
                    "main_bbox_loss": np.nan,
                    "main_giou_loss": np.nan,
                    "aux_loss": np.nan,
                    "aux_l1": np.nan,
                    "aux_giou": np.nan,
                    "aux_coverage": np.nan,
                    "collision_rate": np.nan,
                    "aux_weight_mean": 0.0,
                    "projection_conflict_rate": np.nan,
                    "enc_cls_aux_cosine_raw_mean": np.nan,
                    "projection_removed_ratio_mean": np.nan,
                    "epoch_train_seconds": np.nan,
                    "iteration_seconds": np.nan,
                    "peak_cuda_mb": np.nan,
                    "optimizer_step_skip_rate": np.nan,
                    **metrics,
                }
            )
            pd.DataFrame(history).to_csv(config.history_path, index=False)
            print(f"[validation] epoch=0 mAP={metrics['map']:.4f}")
        context.barrier()

    if start_epoch > config.epochs:
        if context.is_main:
            print(
                f"[done] checkpoint already reached epoch {start_epoch - 1}/"
                f"{config.epochs}"
            )
        base_model.close()
        return

    for epoch in range(start_epoch, config.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training_model.train()
        sums = _empty_epoch_sums()
        epoch_start = time.perf_counter()
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(
            train_loader,
            desc=f"projection-v2 e{epoch}",
            leave=False,
            disable=not context.is_main,
            mininterval=0.5,
        )
        for step, batch in enumerate(iterator):
            pixel_values = batch["pixel_values"].to(
                context.device, non_blocking=True
            )
            pixel_mask = batch["pixel_mask"].to(context.device, non_blocking=True)
            labels = move_labels_to_device(batch["labels"], context.device)
            with torch.autocast(
                device_type=context.device.type,
                dtype=torch.float16,
                enabled=config.amp and context.device.type == "cuda",
            ):
                result = training_model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels,
                    aux_weight=config.aux_weight,
                )
            representation, raw_aux, projected_aux, projection_stats = (
                representation_projected_gradients(base_model, result)
            )
            grad_scale = scaler.get_scale()
            correction_hook = register_representation_gradient_correction(
                representation,
                raw_aux,
                projected_aux,
                auxiliary_weight=config.aux_weight,
                grad_scale=grad_scale,
            )
            try:
                scaler.scale(result["loss"]).backward()
            finally:
                if correction_hook is not None:
                    correction_hook.remove()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            skipped = bool(config.amp and scaler.get_scale() < grad_scale)
            optimizer.zero_grad(set_to_none=True)

            loss_dict = result["outputs"].loss_dict or {}
            sums["batches"] += 1
            sums["total_loss"] += float(result["loss"].detach())
            sums["main_loss"] += float(result["main_loss"].detach())
            sums["main_cls_loss"] += float(loss_dict["loss_ce"].detach())
            sums["main_bbox_loss"] += float(loss_dict["loss_bbox"].detach())
            sums["main_giou_loss"] += float(loss_dict["loss_giou"].detach())
            sums["aux_loss"] += float(result["aux_loss"].detach())
            sums["aux_l1"] += float(result["aux_l1"].detach())
            sums["aux_giou"] += float(result["aux_giou"].detach())
            sums["aux_batches"] += 1
            sums["aux_weight"] += config.aux_weight
            sums["total_objects"] += result["aux_total"]
            sums["used_objects"] += result["aux_used"]
            sums["collision_targets"] += result["aux_collisions"]
            for key, value in result["feature_stats"].items():
                sums[key] += value
            sums["projection_steps"] += 1
            sums["projection_applied"] += int(projection_stats["projection_applied"])
            sums["enc_cls_aux_cosine_raw"] += projection_stats[
                "enc_cls_aux_cosine_raw"
            ]
            sums["projection_removed_ratio"] += projection_stats[
                "projection_removed_ratio"
            ]
            sums["optimizer_steps"] += 1
            sums["optimizer_steps_skipped"] += int(skipped)

            if context.is_main and step % config.gradient_log_every == 0:
                gradient_history.append(
                    {
                        "experiment": EXPERIMENT_NAME,
                        "seed": config.seed,
                        "epoch": epoch,
                        "step": step,
                        "rank": context.rank,
                        "projection_scope": "encoder_output_representation",
                        "projection_vector_numel": representation.numel(),
                        "projection_vector_mb": representation.numel() * 4 / 2**20,
                        **projection_stats,
                        "aux_weight": config.aux_weight,
                        "grad_scale": grad_scale,
                        "optimizer_step_skipped": skipped,
                    }
                )

        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        global_sums = reduce_sums(dict(sums), context)
        elapsed_train += epoch_seconds
        if context.is_main:
            batches = max(global_sums["batches"], 1.0)
            aux_batches = max(global_sums["aux_batches"], 1.0)
            projection_steps = max(global_sums["projection_steps"], 1.0)
            row = {
                "experiment": EXPERIMENT_NAME,
                "seed": config.seed,
                "epoch": epoch,
                "model_fingerprint": fingerprint,
                "train_seconds": elapsed_train,
                "total_loss": global_sums["total_loss"] / batches,
                "main_loss": global_sums["main_loss"] / batches,
                "main_cls_loss": global_sums["main_cls_loss"] / batches,
                "main_bbox_loss": global_sums["main_bbox_loss"] / batches,
                "main_giou_loss": global_sums["main_giou_loss"] / batches,
                "aux_loss": global_sums["aux_loss"] / aux_batches,
                "aux_l1": global_sums["aux_l1"] / aux_batches,
                "aux_giou": global_sums["aux_giou"] / aux_batches,
                "aux_coverage": global_sums["used_objects"]
                / max(global_sums["total_objects"], 1.0),
                "collision_rate": global_sums["collision_targets"]
                / max(global_sums["total_objects"], 1.0),
                "aux_weight_mean": global_sums["aux_weight"] / batches,
                "raw_feature_norm": global_sums["raw_feature_norm"] / aux_batches,
                "adapted_feature_norm": global_sums["adapted_feature_norm"]
                / aux_batches,
                "decoder_feature_norm": global_sums["decoder_feature_norm"]
                / aux_batches,
                "projection_conflict_rate": global_sums["projection_applied"]
                / projection_steps,
                "enc_cls_aux_cosine_raw_mean": global_sums[
                    "enc_cls_aux_cosine_raw"
                ]
                / projection_steps,
                "projection_removed_ratio_mean": global_sums[
                    "projection_removed_ratio"
                ]
                / projection_steps,
                "epoch_train_seconds": epoch_seconds,
                "iteration_seconds": epoch_seconds
                / max(len(train_loader), 1),
                "peak_cuda_mb": (
                    torch.cuda.max_memory_allocated(context.device) / 2**20
                    if context.device.type == "cuda"
                    else np.nan
                ),
                "optimizer_step_skip_rate": global_sums[
                    "optimizer_steps_skipped"
                ]
                / max(global_sums["optimizer_steps"], 1.0),
                "map": np.nan,
                "map50": np.nan,
                "map75": np.nan,
                "map_small": np.nan,
                "map_medium": np.nan,
                "map_large": np.nan,
                "mar100": np.nan,
                "val_seconds": np.nan,
            }
        context.barrier()
        if epoch % config.eval_every == 0 or epoch == config.epochs:
            if context.is_main:
                metrics = evaluate_main(
                    base_model, val_loader, bundle.processor, context.device
                )
                row.update(metrics)
            context.barrier()
        if context.is_main:
            history.append(row)
            pd.DataFrame(history).to_csv(config.history_path, index=False)
            pd.DataFrame(gradient_history, columns=GRADIENT_COLUMNS).to_csv(
                config.gradients_path, index=False
            )
            print(
                f"[epoch {epoch:03d}] total={row['total_loss']:.4f} "
                f"main={row['main_loss']:.4f} aux={row['aux_loss']:.4f} "
                f"mAP={row['map']:.4f} conflict={row['projection_conflict_rate']:.3f}"
            )
        save_checkpoint(
            base_model,
            optimizer,
            scheduler,
            scaler,
            config,
            context,
            epoch,
            elapsed_train,
            history if context.is_main else [],
            gradient_history if context.is_main else [],
        )

    base_model.close()
