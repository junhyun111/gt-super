from __future__ import annotations

import gc
import math
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import ExperimentConfig, seed_everything
from .data import DataBundle, make_loaders
from .eval import evaluate_main
from .model import GTDeformableDetr, make_model


def move_labels_to_device(labels, device):
    return [{key: value.to(device) for key, value in target.items()} for target in labels]


def auxiliary_weight(experiment: str, progress: float, total_epochs: int, base_weight: float) -> float:
    if experiment == "baseline":
        return 0.0
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


def _save_checkpoint(model, config, experiment, seed, epoch, history, gradients):
    state = {
        "model_state_dict": model.state_dict(),
        "config": config.as_dict(), "experiment": experiment, "seed": seed,
        "epoch": epoch, "history": history.to_dict("records"),
        "gradients": gradients.to_dict("records"),
    }
    path = config.checkpoint_path(experiment, seed)
    torch.save(state, path)
    torch.save(state, path.with_name(path.stem + f"_epoch{epoch}.pt"))


def train_one_experiment(
    config: ExperimentConfig,
    bundle: DataBundle,
    experiment: str,
    seed: int | None = None,
    save_checkpoint: bool = True,
):
    seed = config.seed if seed is None else seed
    if experiment not in GTDeformableDetr.VALID_MODES:
        raise ValueError(f"Unknown experiment: {experiment}")
    print(f"\n===== {experiment} / seed={seed} =====")
    seed_everything(seed)
    train_loader, val_loader = make_loaders(config, bundle, seed)
    model, fingerprint = make_model(config, experiment, seed)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
    history, gradient_history = [], []
    elapsed_train = 0.0

    initial_metrics = evaluate_main(model, val_loader, bundle.processor, config)
    history.append({
        "experiment": experiment, "seed": seed, "epoch": 0,
        "model_fingerprint": fingerprint, "train_seconds": 0.0,
        "main_loss": np.nan, "main_bbox_loss": np.nan, "main_giou_loss": np.nan,
        "aux_loss": np.nan, "aux_l1": np.nan, "aux_giou": np.nan,
        "aux_coverage": np.nan, "collision_rate": np.nan, "aux_weight_mean": 0.0,
        **initial_metrics,
    })

    for epoch in range(1, config.epochs + 1):
        model.train()
        sums = Counter()
        total_objects = used_objects = collision_targets = 0
        epoch_start = time.perf_counter()
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

            if step == 0 and result["aux_executed"] and experiment != "separate":
                parameters = unique_parameters(model.shared_bbox_head.parameters())
                cosine, main_norm, aux_norm = gradient_cosine(
                    result["main_loss"], weight * result["aux_loss"], parameters
                )
                gradient_history.append({
                    "experiment": experiment, "seed": seed, "epoch": epoch,
                    "cosine": cosine, "main_grad_norm": main_norm,
                    "weighted_aux_grad_norm": aux_norm,
                    "norm_ratio": aux_norm / max(main_norm, 1e-12),
                })

            scaler.scale(result["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss_dict = result["outputs"].loss_dict or {}
            sums["batches"] += 1
            sums["main_loss"] += float(result["main_loss"].detach())
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
        elapsed_train += time.perf_counter() - epoch_start
        metrics = evaluate_main(model, val_loader, bundle.processor, config)
        batches = max(sums["batches"], 1)
        aux_batches = max(sums["aux_batches"], 1)
        row = {
            "experiment": experiment, "seed": seed, "epoch": epoch,
            "model_fingerprint": fingerprint, "train_seconds": elapsed_train,
            "main_loss": sums["main_loss"] / batches,
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
            **metrics,
        }
        history.append(row)
        print({key: round(value, 4) for key, value in row.items()
               if key in {"epoch", "main_loss", "aux_loss", "map", "map50", "map75",
                          "aux_coverage", "collision_rate"}})
        history_df = pd.DataFrame(history)
        gradients_df = pd.DataFrame(gradient_history)
        history_df.to_csv(config.history_path(experiment, seed), index=False)
        gradients_df.to_csv(config.gradients_path(experiment, seed), index=False)
        if save_checkpoint:
            _save_checkpoint(model, config, experiment, seed, epoch, history_df, gradients_df)

    return model, pd.DataFrame(history), pd.DataFrame(gradient_history)


def release_model(model):
    if model is None:
        return
    model.close()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
