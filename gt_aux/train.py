from __future__ import annotations

import gc
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import ExperimentConfig, seed_everything
from .data import DataBundle, make_loaders
from .eval import evaluate_main
from .model import (
    GTDeformableDetr,
    LOCALIZATION_AUX_WEIGHTS,
    NO_AUX_MODES,
    PARAMETER_PROJECTED_MODES,
    REPRESENTATION_PROJECTED_MODE,
    make_model,
)


GRADIENT_COLUMNS = [
    "experiment", "seed", "data_seed", "epoch", "cosine", "main_grad_norm",
    "weighted_aux_grad_norm", "norm_ratio", "step", "projection_scope",
    "enc_cls_aux_cosine_raw", "enc_cls_aux_dot_raw",
    "enc_cls_aux_dot_projected", "enc_cls_grad_norm", "enc_aux_grad_norm",
    "enc_aux_grad_norm_projected", "projection_applied",
    "projection_removed_ratio", "projection_vector_numel",
    "projection_vector_mb", "grad_scale", "optimizer_step_skipped",
]


def move_labels_to_device(labels, device):
    return [{key: value.to(device) for key, value in target.items()} for target in labels]


def auxiliary_weight(experiment: str, progress: float, total_epochs: int, base_weight: float) -> float:
    if experiment in NO_AUX_MODES:
        return 0.0
    if experiment in LOCALIZATION_AUX_WEIGHTS:
        return LOCALIZATION_AUX_WEIGHTS[experiment]
    if experiment == "shared_late_decay":
        # Keep full auxiliary supervision through the early/middle phase, then
        # reduce its magnitude when the main and auxiliary objectives may begin
        # to conflict. With 7 epochs this is 0.5 (e1-e5), 0.25 (e6), 0.05 (e7).
        epoch = min(int(progress) + 1, total_epochs)
        if epoch <= max(total_epochs - 2, 1):
            return base_weight
        if epoch == max(total_epochs - 1, 1):
            return base_weight * 0.5
        return base_weight * 0.1
    if experiment != "shared_decay":
        return base_weight
    warmup_end, hold_end = 0.5, 2.0
    decay_end = max(hold_end + 1e-6, total_epochs * 0.75)
    if progress < warmup_end:
        return base_weight * progress / warmup_end
    if progress <= hold_end:
        return base_weight
    if progress >= decay_end:
        return 0.0
    ratio = (progress - hold_end) / (decay_end - hold_end)
    return base_weight * 0.5 * (1 + math.cos(math.pi * ratio))


def unique_parameters(parameters):
    seen, result = set(), []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def make_optimizer(model, config: ExperimentConfig):
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if "detector.model.backbone" in name else other).append(parameter)
    return torch.optim.AdamW([
        {"params": unique_parameters(other), "lr": config.lr},
        {"params": unique_parameters(backbone), "lr": config.backbone_lr},
    ], weight_decay=config.weight_decay)


def gradient_cosine(main_loss, aux_loss, parameters):
    main_grads = torch.autograd.grad(main_loss, parameters, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(aux_loss, parameters, retain_graph=True, allow_unused=True)
    paired = [(main, aux) for main, aux in zip(main_grads, aux_grads)
              if main is not None and aux is not None]
    if not paired:
        return float("nan"), float("nan"), float("nan")
    main_vector = torch.cat([main.detach().flatten().float() for main, _ in paired])
    aux_vector = torch.cat([aux.detach().flatten().float() for _, aux in paired])
    main_norm, aux_norm = main_vector.norm(), aux_vector.norm()
    cosine = F.cosine_similarity(main_vector, aux_vector, dim=0)
    return float(cosine), float(main_norm), float(aux_norm)


def project_conflicting_gradient(cls_grads, aux_grads, epsilon: float = 1e-12):
    """Project one global auxiliary vector off a conflicting classification vector.

    The lists are treated as chunks of one encoder-wide vector.  Auxiliary
    entries without a corresponding classification gradient are preserved.
    No tensors are concatenated, which avoids another encoder-sized allocation.
    """
    if len(cls_grads) != len(aux_grads):
        raise ValueError("Classification and auxiliary gradient lists must align")
    cls_present = [grad for grad in cls_grads if grad is not None]
    aux_present = [grad for grad in aux_grads if grad is not None]
    paired = [
        (cls_grad, aux_grad)
        for cls_grad, aux_grad in zip(cls_grads, aux_grads)
        if cls_grad is not None and aux_grad is not None
    ]
    if not cls_present or not aux_present or not paired:
        raise RuntimeError("Projected-E requires classification and auxiliary encoder gradients")

    # Accumulate the global diagnostics in float32 even under autocast.
    dot = sum(
        (cls_grad.detach().float() * aux_grad.detach().float()).sum()
        for cls_grad, aux_grad in paired
    )
    cls_norm_sq = sum(
        grad.detach().float().square().sum() for grad in cls_present
    )
    aux_norm_sq = sum(
        grad.detach().float().square().sum() for grad in aux_present
    )
    cls_norm = cls_norm_sq.sqrt()
    aux_norm = aux_norm_sq.sqrt()
    cosine = dot / (cls_norm * aux_norm + epsilon)
    applied = bool((dot < 0).item())
    coefficient = dot / (cls_norm_sq + epsilon) if applied else dot.new_zeros(())

    projected = []
    for cls_grad, aux_grad in zip(cls_grads, aux_grads):
        if aux_grad is None:
            projected.append(None)
        elif applied and cls_grad is not None:
            projected.append(aux_grad - coefficient.to(aux_grad.dtype) * cls_grad)
        else:
            projected.append(aux_grad)

    projected_present = [grad for grad in projected if grad is not None]
    projected_norm = sum(
        grad.detach().float().square().sum() for grad in projected_present
    ).sqrt()
    projected_dot = sum(
        (cls_grad.detach().float() * projected_grad.detach().float()).sum()
        for cls_grad, projected_grad in zip(cls_grads, projected)
        if cls_grad is not None and projected_grad is not None
    )
    removed_ratio = 1.0 - projected_norm / (aux_norm + epsilon)
    stats = {
        "enc_cls_aux_cosine_raw": float(cosine),
        "enc_cls_aux_dot_raw": float(dot),
        "enc_cls_aux_dot_projected": float(projected_dot),
        "enc_cls_grad_norm": float(cls_norm),
        "enc_aux_grad_norm": float(aux_norm),
        "enc_aux_grad_norm_projected": float(projected_norm),
        "projection_applied": applied,
        "projection_removed_ratio": float(removed_ratio),
    }
    return tuple(projected), stats


def _projection_losses(result):
    loss_dict = result["outputs"].loss_dict or {}
    if "loss_ce" not in loss_dict:
        raise KeyError("Gradient projection requires outputs.loss_dict['loss_ce']")
    if not result["aux_executed"] or result["aux_loss"] is None:
        raise RuntimeError("Gradient projection requires an executed auxiliary branch")
    return loss_dict["loss_ce"], result["aux_loss"]


def parameter_projected_encoder_gradients(model, result):
    """V1 ablation: extract two full Transformer-encoder parameter gradients."""
    loss_cls, loss_aux = _projection_losses(result)
    parameters = unique_parameters(model.detector.model.encoder.parameters())
    cls_grads = torch.autograd.grad(
        loss_cls, parameters, retain_graph=True, allow_unused=True
    )
    aux_grads = torch.autograd.grad(
        loss_aux, parameters, retain_graph=True, allow_unused=True
    )
    projected_aux_grads, stats = project_conflicting_gradient(cls_grads, aux_grads)
    return parameters, aux_grads, projected_aux_grads, stats


def projected_encoder_gradients(model, result):
    """Backward-compatible name for the V1 parameter-space ablation."""
    return parameter_projected_encoder_gradients(model, result)


def representation_projected_gradients(model, result):
    """V2: project gradients at the single shared encoder-output tensor.

    These two ``autograd.grad`` calls stop at the representation, so they do not
    traverse the Transformer encoder or backbone.  The normal total backward
    later propagates one corrected gradient through those shared modules.
    """
    loss_cls, loss_aux = _projection_losses(result)
    representation = model.encoder_representation
    cls_grad = torch.autograd.grad(
        loss_cls, representation, retain_graph=True, allow_unused=False
    )[0]
    aux_grad = torch.autograd.grad(
        loss_aux, representation, retain_graph=True, allow_unused=False
    )[0]
    (projected_aux_grad,), stats = project_conflicting_gradient(
        (cls_grad,), (aux_grad,)
    )
    return representation, aux_grad, projected_aux_grad, stats


def replace_encoder_auxiliary_gradient(
    parameters, raw_aux_grads, projected_aux_grads, aux_weight: float
):
    """Replace the unscaled encoder aux component after the normal backward."""
    with torch.no_grad():
        for parameter, raw_aux, projected_aux in zip(
            parameters, raw_aux_grads, projected_aux_grads
        ):
            if parameter.grad is None or raw_aux is None or projected_aux is None:
                continue
            parameter.grad.add_(projected_aux - raw_aux, alpha=aux_weight)


def register_representation_gradient_correction(
    representation,
    raw_aux_grad,
    projected_aux_grad,
    aux_weight: float,
    grad_scale: float = 1.0,
):
    """Add the V2 correction to the normal total gradient arriving at ``E``.

    ``GradScaler`` scales the normal backward gradient before hooks run, so the
    externally computed correction must use the same scale.  The returned hook
    must be removed immediately after backward.
    """
    # The projection helper returns the original object for a non-conflicting
    # step.  Avoid allocating/scanning a representation-sized all-zero tensor.
    if projected_aux_grad is raw_aux_grad:
        return None
    correction = (
        (projected_aux_grad - raw_aux_grad).detach()
        * float(aux_weight)
        * float(grad_scale)
    )

    def add_correction(total_grad):
        return total_grad + correction.to(dtype=total_grad.dtype)

    return representation.register_hook(add_correction)


def _save_checkpoint(
    model, optimizer, scheduler, scaler, train_loader, config,
    experiment, seed, epoch, elapsed_train, history, gradients,
):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.as_dict(), "experiment": experiment, "seed": seed,
        "epoch": epoch, "elapsed_train": elapsed_train,
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
    path = config.checkpoint_path(experiment, seed)
    torch.save(state, path)
    if config.save_epoch_checkpoints:
        torch.save(state, path.with_name(path.stem + f"_epoch{epoch}.pt"))


def _load_resume_checkpoint(
    resume_from, model, optimizer, scheduler, scaler, train_loader,
    config, experiment, seed,
):
    path = Path(resume_from).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "scaler_state_dict", "epoch", "elapsed_train", "history", "gradients",
        "python_rng_state", "numpy_rng_state", "torch_rng_state",
        "loader_generator_state",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            "Checkpoint cannot resume training; missing state: " + ", ".join(missing)
        )
    if checkpoint.get("experiment") != experiment:
        raise ValueError(
            f"Checkpoint experiment={checkpoint.get('experiment')!r}, expected {experiment!r}"
        )
    if int(checkpoint.get("seed")) != int(seed):
        raise ValueError(f"Checkpoint seed={checkpoint.get('seed')}, expected {seed}")
    saved_config = checkpoint.get("config", {})
    current_config = config.as_dict()
    compatibility_keys = {
        "run_mode", "checkpoint", "data_seed", "train_images", "val_images",
        "epochs", "batch_size", "image_size", "lr", "backbone_lr",
        "weight_decay", "grad_clip", "base_aux_weight", "feature_level",
        "horizontal_flip_p", "use_amp",
        "deterministic",
    }
    mismatches = {
        key: (saved_config.get(key), current_config.get(key))
        for key in compatibility_keys
        if saved_config.get(key) != current_config.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved!r}, current={current!r}"
            for key, (saved, current) in sorted(mismatches.items())
        )
        raise ValueError(f"Resume config mismatch: {details}")

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


def train_one_experiment(
    config: ExperimentConfig,
    bundle: DataBundle,
    experiment: str,
    seed: int | None = None,
    save_checkpoint: bool = True,
    resume_from: str | Path | None = None,
):
    seed = config.seed if seed is None else seed
    if experiment not in GTDeformableDetr.VALID_MODES:
        raise ValueError(f"Unknown experiment: {experiment}")
    print(f"\n===== {experiment} / seed={seed} =====")
    seed_everything(seed, deterministic=config.deterministic)
    train_loader, val_loader = make_loaders(config, bundle, seed)
    model, fingerprint = make_model(config, experiment, seed)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
    history, gradient_history, elapsed_train, start_epoch = [], [], 0.0, 1

    if resume_from is not None:
        checkpoint, resume_path = _load_resume_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, train_loader,
            config, experiment, seed,
        )
        history = list(checkpoint["history"])
        gradient_history = list(checkpoint["gradients"])
        elapsed_train = float(checkpoint["elapsed_train"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[resume] loaded {resume_path}; continuing at epoch {start_epoch}")
    else:
        print(f"[phase] initial main-only validation: {len(val_loader)} batches")
        initial_metrics = evaluate_main(model, val_loader, bundle.processor, config)
        print(f"[phase] initial validation complete: mAP={initial_metrics['map']:.4f}, "
              f"AP@0.5={initial_metrics['map50']:.4f}")
        history.append({
            "experiment": experiment, "seed": seed, "data_seed": config.data_seed,
            "epoch": 0,
            "model_fingerprint": fingerprint, "train_seconds": 0.0,
            "total_loss": np.nan, "main_loss": np.nan,
            "main_cls_or_obj_loss": np.nan,
            "main_bbox_loss": np.nan, "main_giou_loss": np.nan,
            "aux_loss": np.nan, "aux_l1": np.nan, "aux_giou": np.nan,
            "aux_coverage": np.nan, "collision_rate": np.nan, "aux_weight_mean": 0.0,
            "projection_conflict_rate": np.nan,
            "enc_cls_aux_cosine_raw_mean": np.nan,
            "projection_removed_ratio_mean": np.nan,
            "epoch_train_seconds": np.nan,
            "iteration_seconds": np.nan,
            "peak_extra_cuda_mb": np.nan,
            "optimizer_step_skip_rate": np.nan,
            **initial_metrics,
        })

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"[phase] training epoch {epoch}/{config.epochs}: {len(train_loader)} batches")
        model.train()
        sums = Counter()
        total_objects = used_objects = collision_targets = 0
        epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(config.device)
            epoch_memory_start = torch.cuda.memory_allocated(config.device)
        else:
            epoch_memory_start = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"{experiment} e{epoch}", leave=False)):
            pixel_values = batch["pixel_values"].to(config.device, non_blocking=True)
            pixel_mask = batch["pixel_mask"].to(config.device, non_blocking=True)
            labels = move_labels_to_device(batch["labels"], config.device)
            progress = (epoch - 1) + step / max(len(train_loader), 1)
            weight = auxiliary_weight(experiment, progress, config.epochs, config.base_aux_weight)
            with torch.autocast(device_type=config.device.type, dtype=torch.float16, enabled=config.use_amp):
                result = model(pixel_values=pixel_values, pixel_mask=pixel_mask,
                               labels=labels, aux_weight=weight)

            separate_modes = {"separate", "separate_e2e"}
            parameter_projection_state = None
            representation_hook = None
            projection_row = None
            if experiment in PARAMETER_PROJECTED_MODES:
                parameters, raw_aux_grads, projected_aux_grads, projection_stats = (
                    parameter_projected_encoder_gradients(model, result)
                )
                parameter_projection_state = (
                    parameters, raw_aux_grads, projected_aux_grads
                )
                projection_scope = "transformer_encoder_parameters"
                projection_vector_numel = sum(
                    parameter.numel() for parameter in parameters
                )
            elif experiment == REPRESENTATION_PROJECTED_MODE:
                representation, raw_aux_grad, projected_aux_grad, projection_stats = (
                    representation_projected_gradients(model, result)
                )
                representation_hook = register_representation_gradient_correction(
                    representation,
                    raw_aux_grad,
                    projected_aux_grad,
                    aux_weight=weight,
                    grad_scale=scaler.get_scale(),
                )
                projection_scope = "encoder_output_representation"
                projection_vector_numel = representation.numel()
            else:
                projection_stats = None

            if projection_stats is not None:
                projection_row = {
                    "experiment": experiment, "seed": seed,
                    "data_seed": config.data_seed, "epoch": epoch, "step": step,
                    "projection_scope": projection_scope,
                    "projection_vector_numel": projection_vector_numel,
                    "projection_vector_mb": projection_vector_numel * 4 / 2**20,
                    "cosine": projection_stats["enc_cls_aux_cosine_raw"],
                    "main_grad_norm": projection_stats["enc_cls_grad_norm"],
                    "weighted_aux_grad_norm": (
                        weight * projection_stats["enc_aux_grad_norm"]
                    ),
                    "norm_ratio": (
                        weight * projection_stats["enc_aux_grad_norm"]
                        / max(projection_stats["enc_cls_grad_norm"], 1e-12)
                    ),
                    **projection_stats,
                }
                gradient_history.append(projection_row)
                sums["projection_steps"] += 1
                sums["projection_applied"] += int(projection_stats["projection_applied"])
                sums["enc_cls_aux_cosine_raw"] += projection_stats["enc_cls_aux_cosine_raw"]
                sums["projection_removed_ratio"] += projection_stats["projection_removed_ratio"]
            elif step == 0 and result["aux_executed"] and experiment not in separate_modes:
                parameters = unique_parameters(model.shared_bbox_head.parameters())
                cosine, main_norm, aux_norm = gradient_cosine(
                    result["main_loss"], weight * result["aux_loss"], parameters
                )
                gradient_history.append({
                    "experiment": experiment, "seed": seed,
                    "data_seed": config.data_seed, "epoch": epoch,
                    "cosine": cosine, "main_grad_norm": main_norm,
                    "weighted_aux_grad_norm": aux_norm,
                    "norm_ratio": aux_norm / max(main_norm, 1e-12),
                })

            grad_scale = scaler.get_scale()
            try:
                scaler.scale(result["loss"]).backward()
            finally:
                if representation_hook is not None:
                    representation_hook.remove()
            scaler.unscale_(optimizer)
            if parameter_projection_state is not None:
                replace_encoder_auxiliary_gradient(
                    *parameter_projection_state, aux_weight=weight
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_skipped = bool(
                config.use_amp and scaler.get_scale() < grad_scale
            )
            sums["optimizer_steps"] += 1
            sums["optimizer_steps_skipped"] += int(optimizer_step_skipped)
            if projection_row is not None:
                projection_row["grad_scale"] = grad_scale
                projection_row["optimizer_step_skipped"] = optimizer_step_skipped
            optimizer.zero_grad(set_to_none=True)

            loss_dict = result["outputs"].loss_dict or {}
            sums["batches"] += 1
            sums["total_loss"] += float(result["loss"].detach())
            sums["main_loss"] += float(result["main_loss"].detach())
            sums["main_cls_or_obj_loss"] += float(loss_dict.get("loss_ce", torch.tensor(float("nan"), device=config.device)).detach())
            sums["main_bbox_loss"] += float(loss_dict.get("loss_bbox", torch.tensor(float("nan"), device=config.device)).detach())
            sums["main_giou_loss"] += float(loss_dict.get("loss_giou", torch.tensor(float("nan"), device=config.device)).detach())
            sums["aux_weight"] += weight
            if result["aux_executed"]:
                sums["aux_loss"] += float(result["aux_loss"].detach())
                sums["aux_l1"] += float(result["aux_l1"].detach())
                sums["aux_giou"] += float(result["aux_giou"].detach())
                sums["aux_batches"] += 1
                for key, value in result["feature_stats"].items():
                    sums[key] += value
            total_objects += result["aux_total"]
            used_objects += result["aux_used"]
            collision_targets += result["aux_collisions"]

        scheduler.step()
        epoch_train_seconds = time.perf_counter() - epoch_start
        elapsed_train += epoch_train_seconds
        epoch_peak_extra_cuda_mb = (
            (torch.cuda.max_memory_allocated(config.device) - epoch_memory_start) / 2**20
            if torch.cuda.is_available() else np.nan
        )
        print(f"[phase] validating epoch {epoch}/{config.epochs}: {len(val_loader)} batches")
        metrics = evaluate_main(model, val_loader, bundle.processor, config)
        batches = max(sums["batches"], 1)
        aux_batches = max(sums["aux_batches"], 1)
        row = {
            "experiment": experiment, "seed": seed, "data_seed": config.data_seed,
            "epoch": epoch,
            "model_fingerprint": fingerprint, "train_seconds": elapsed_train,
            "total_loss": sums["total_loss"] / batches,
            "main_loss": sums["main_loss"] / batches,
            "main_cls_or_obj_loss": sums["main_cls_or_obj_loss"] / batches,
            "main_bbox_loss": sums["main_bbox_loss"] / batches,
            "main_giou_loss": sums["main_giou_loss"] / batches,
            "aux_loss": sums["aux_loss"] / aux_batches if sums["aux_batches"] else np.nan,
            "aux_l1": sums["aux_l1"] / aux_batches if sums["aux_batches"] else np.nan,
            "aux_giou": sums["aux_giou"] / aux_batches if sums["aux_batches"] else np.nan,
            "aux_coverage": used_objects / max(total_objects, 1) if total_objects else np.nan,
            "collision_rate": collision_targets / max(total_objects, 1) if total_objects else np.nan,
            "aux_weight_mean": sums["aux_weight"] / batches,
            "raw_feature_norm": sums["raw_feature_norm"] / aux_batches if sums["aux_batches"] else np.nan,
            "adapted_feature_norm": sums["adapted_feature_norm"] / aux_batches if sums["aux_batches"] else np.nan,
            "decoder_feature_norm": sums["decoder_feature_norm"] / aux_batches if sums["aux_batches"] else np.nan,
            "projection_conflict_rate": (
                sums["projection_applied"] / sums["projection_steps"]
                if sums["projection_steps"] else np.nan
            ),
            "enc_cls_aux_cosine_raw_mean": (
                sums["enc_cls_aux_cosine_raw"] / sums["projection_steps"]
                if sums["projection_steps"] else np.nan
            ),
            "projection_removed_ratio_mean": (
                sums["projection_removed_ratio"] / sums["projection_steps"]
                if sums["projection_steps"] else np.nan
            ),
            "epoch_train_seconds": epoch_train_seconds,
            "iteration_seconds": epoch_train_seconds / batches,
            "peak_extra_cuda_mb": epoch_peak_extra_cuda_mb,
            "optimizer_step_skip_rate": (
                sums["optimizer_steps_skipped"] / max(sums["optimizer_steps"], 1)
            ),
            **metrics,
        }
        history.append(row)
        print({key: round(value, 4) for key, value in row.items()
               if key in {"epoch", "total_loss", "main_loss", "aux_loss", "map", "map50", "map75",
                          "aux_coverage", "collision_rate"}})
        history_df = pd.DataFrame(history)
        # Preserve the CSV schema even when an experiment (for example,
        # baseline) does not produce auxiliary-gradient measurements.
        gradients_df = pd.DataFrame(gradient_history, columns=GRADIENT_COLUMNS)
        history_df.to_csv(config.history_path(experiment, seed), index=False)
        gradients_df.to_csv(config.gradients_path(experiment, seed), index=False)
        if save_checkpoint:
            _save_checkpoint(
                model, optimizer, scheduler, scaler, train_loader, config,
                experiment, seed, epoch, elapsed_train, history_df, gradients_df,
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
