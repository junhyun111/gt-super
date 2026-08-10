from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, DeformableDetrForObjectDetection
from transformers.loss.loss_deformable_detr import DeformableDetrHungarianMatcher

from gt_aux.config import (
    ID2LABEL,
    LABEL2ID,
    ExperimentConfig,
    model_fingerprint,
    seed_everything,
)


EXPERIMENT_NAME = "moment_guided"


@dataclass(frozen=True)
class MomentLossWeights:
    """Weights inside the training-only spatial moment objective."""

    center: float = 1.0
    covariance: float = 1.0

    def __post_init__(self) -> None:
        if self.center < 0 or self.covariance < 0:
            raise ValueError("Moment loss weights must be non-negative")


def build_detector(config: ExperimentConfig, initialization_seed: int):
    """Build the same pretrained Deformable DETR used by the baseline."""

    seed_everything(initialization_seed)
    try:
        model_config = AutoConfig.from_pretrained(config.checkpoint, local_files_only=True)
        local_only = True
    except OSError:
        model_config = AutoConfig.from_pretrained(config.checkpoint)
        local_only = False

    if model_config.model_type != "deformable_detr":
        raise ValueError(f"Expected deformable_detr, got {model_config.model_type!r}")
    if model_config.two_stage or model_config.with_box_refine:
        raise ValueError("This experiment expects one-stage Deformable DETR without box refinement")
    if model_config.num_feature_levels != 4:
        raise ValueError("Moment implementation expects the baseline's four feature levels")

    model_config.num_labels = len(LABEL2ID)
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID
    model_config.auxiliary_loss = False
    model_config.disable_custom_kernels = config.disable_custom_kernels
    return DeformableDetrForObjectDetection.from_pretrained(
        config.checkpoint,
        config=model_config,
        ignore_mismatched_sizes=True,
        local_files_only=local_only,
    )


def _assert_bbox_heads_are_tied(detector: nn.Module) -> None:
    pointers = [
        tuple(parameter.data_ptr() for parameter in head.parameters())
        for head in detector.bbox_embed
    ]
    if len(set(pointers)) != 1:
        raise AssertionError("The baseline's shared DETR box heads are not tied")


class MomentGuidedDeformableDetr(nn.Module):
    """Deformable DETR with training-only query-attention moment supervision.

    Deformable DETR has no dense ``H x W`` decoder attention map.  Its exact
    analogue is a sparse distribution over the sampling locations used by each
    query/head/feature-level.  During a supervised forward pass this wrapper
    temporarily observes the final decoder cross-attention and computes the
    first and second spatial moments of that sparse distribution.

    No modules or parameters are added to the detector, and no hook is present
    when labels/moment diagnostics are not requested.  Inference therefore
    follows the original detector path.
    """

    def __init__(
        self,
        detector: DeformableDetrForObjectDetection,
        *,
        loss_weights: MomentLossWeights | None = None,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.loss_weights = loss_weights or MomentLossWeights()
        self.epsilon = float(epsilon)
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

        config = detector.config
        self.matcher = DeformableDetrHungarianMatcher(
            class_cost=config.class_cost,
            bbox_cost=config.bbox_cost,
            giou_cost=config.giou_cost,
        )
        self._attention_cache: dict[str, torch.Tensor] = {}
        self.moment_forward_calls = 0

    @property
    def final_cross_attention(self) -> nn.Module:
        return self.detector.model.decoder.layers[-1].encoder_attn

    @property
    def extra_parameter_count(self) -> int:
        detector_ids = {id(parameter) for parameter in self.detector.parameters()}
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in detector_ids
        )

    def close(self) -> None:
        """Compatibility no-op; training hooks are temporary per forward."""

        self._attention_cache = {}

    @staticmethod
    def _valid_ratios(
        attention_mask: torch.Tensor | None,
        spatial_shapes: torch.Tensor,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return ``(valid_width/full_width, valid_height/full_height)``."""

        if attention_mask is None:
            return torch.ones(
                batch_size,
                spatial_shapes.shape[0],
                2,
                dtype=dtype,
                device=device,
            )

        mask = attention_mask.to(torch.bool)
        ratios = []
        start = 0
        for height_tensor, width_tensor in spatial_shapes:
            height, width = int(height_tensor), int(width_tensor)
            level_mask = mask[:, start : start + height * width].reshape(
                batch_size, height, width
            )
            valid_height = level_mask.any(dim=2).sum(dim=1).to(dtype)
            valid_width = level_mask.any(dim=1).sum(dim=1).to(dtype)
            ratios.append(
                torch.stack((valid_width / width, valid_height / height), dim=-1)
            )
            start += height * width
        return torch.stack(ratios, dim=1).clamp_min(torch.finfo(dtype).eps)

    def _capture_final_cross_attention(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        required = {
            "hidden_states",
            "reference_points",
            "spatial_shapes",
            "spatial_shapes_list",
        }
        missing = required.difference(kwargs)
        if missing:
            raise RuntimeError(
                "Unsupported Transformers cross-attention API; missing "
                + ", ".join(sorted(missing))
            )

        query = kwargs["hidden_states"]
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            query = query + position_embeddings

        batch_size, num_queries, _ = query.shape
        offsets = module.sampling_offsets(query).view(
            batch_size,
            num_queries,
            module.n_heads,
            module.n_levels,
            module.n_points,
            2,
        )
        attention_weights = output[1]
        reference_points = kwargs["reference_points"]
        spatial_shapes = kwargs["spatial_shapes"]
        coordinate_count = reference_points.shape[-1]
        if coordinate_count == 2:
            offset_normalizer = torch.stack(
                (spatial_shapes[..., 1], spatial_shapes[..., 0]), dim=-1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif coordinate_count == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets
                / module.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise RuntimeError(
                f"Expected 2-D or 4-D reference points, got {coordinate_count}"
            )

        # Cross-attention samples in padded feature-map coordinates.  GT boxes
        # are normalized to each image's valid region, so undo padding before
        # comparing their geometry.
        work_dtype = torch.float32
        sampling_locations = sampling_locations.to(work_dtype)
        attention_weights = attention_weights.to(work_dtype)
        valid_ratios = self._valid_ratios(
            kwargs.get("attention_mask"),
            spatial_shapes,
            batch_size=batch_size,
            dtype=work_dtype,
            device=sampling_locations.device,
        )
        valid_locations = sampling_locations / valid_ratios[:, None, None, :, None, :]
        in_bounds = (
            (valid_locations[..., 0] >= 0.0)
            & (valid_locations[..., 0] <= 1.0)
            & (valid_locations[..., 1] >= 0.0)
            & (valid_locations[..., 1] <= 1.0)
        )

        # Each head has unit mass over levels/points.  Averaging heads makes one
        # query-level distribution with unit mass before invalid samples are
        # removed.
        weights = attention_weights / module.n_heads
        valid_weights = weights * in_bounds.to(weights.dtype)
        valid_mass = valid_weights.sum(dim=(2, 3, 4))
        normalized_weights = valid_weights / valid_mass.clamp_min(self.epsilon)[
            :, :, None, None, None
        ]

        mean = (
            normalized_weights[..., None] * valid_locations
        ).sum(dim=(2, 3, 4))
        centered = valid_locations - mean[:, :, None, None, None, :]
        covariance = torch.einsum(
            "bqhlp,bqhlpi,bqhlpj->bqij",
            normalized_weights,
            centered,
            centered,
        )
        self._attention_cache = {
            "mean": mean,
            "covariance": covariance,
            "valid_mass": valid_mass,
        }

    @staticmethod
    def _empty_details() -> list[dict[str, float | int]]:
        return []

    def _compute_moment_loss(
        self,
        outputs: Any,
        labels: list[dict[str, torch.Tensor]],
        *,
        return_details: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], list[dict]]:
        if not self._attention_cache:
            raise RuntimeError("Final cross-attention was not captured")

        predicted_mean = self._attention_cache["mean"]
        predicted_covariance = self._attention_cache["covariance"]
        valid_mass = self._attention_cache["valid_mass"]
        total_targets = sum(int(target["boxes"].shape[0]) for target in labels)
        if total_targets == 0:
            zero = predicted_mean.sum() * 0.0
            stats = {
                "matched_queries": 0.0,
                "valid_attention_mass": float("nan"),
                "off_diagonal_abs": float("nan"),
                "predicted_variance": float("nan"),
                "target_variance": float("nan"),
            }
            return zero, zero, zero, stats, self._empty_details()

        indices = self.matcher(
            {"logits": outputs.logits, "pred_boxes": outputs.pred_boxes}, labels
        )
        predicted_means, predicted_covariances = [], []
        target_boxes, matched_masses, batch_indices, query_indices, target_indices = [], [], [], [], []
        for batch_index, (query_index, target_index) in enumerate(indices):
            if len(query_index) == 0:
                continue
            query_index = query_index.to(predicted_mean.device)
            target_index = target_index.to(predicted_mean.device)
            predicted_means.append(predicted_mean[batch_index, query_index])
            predicted_covariances.append(predicted_covariance[batch_index, query_index])
            target_boxes.append(labels[batch_index]["boxes"][target_index].to(torch.float32))
            matched_masses.append(valid_mass[batch_index, query_index])
            batch_indices.extend([batch_index] * len(query_index))
            query_indices.extend(query_index.detach().cpu().tolist())
            target_indices.extend(target_index.detach().cpu().tolist())

        if not predicted_means:
            zero = predicted_mean.sum() * 0.0
            stats = {
                "matched_queries": 0.0,
                "valid_attention_mass": float("nan"),
                "off_diagonal_abs": float("nan"),
                "predicted_variance": float("nan"),
                "target_variance": float("nan"),
            }
            return zero, zero, zero, stats, self._empty_details()

        predicted_mean_matched = torch.cat(predicted_means)
        predicted_covariance_matched = torch.cat(predicted_covariances)
        target_boxes_matched = torch.cat(target_boxes)
        matched_mass = torch.cat(matched_masses)

        target_mean = target_boxes_matched[:, :2]
        target_covariance = predicted_covariance_matched.new_zeros(
            target_boxes_matched.shape[0], 2, 2
        )
        target_covariance[:, 0, 0] = target_boxes_matched[:, 2].square() / 12.0
        target_covariance[:, 1, 1] = target_boxes_matched[:, 3].square() / 12.0

        center_components = (predicted_mean_matched - target_mean).abs()
        covariance_components = (
            predicted_covariance_matched - target_covariance
        ).abs()
        center_per_match = center_components.sum(dim=-1)
        covariance_per_match = covariance_components.sum(dim=(-1, -2))
        center_loss = center_per_match.mean()
        covariance_loss = covariance_per_match.mean()
        moment_loss = (
            self.loss_weights.center * center_loss
            + self.loss_weights.covariance * covariance_loss
        )

        diagonal = predicted_covariance_matched.diagonal(dim1=-2, dim2=-1)
        target_diagonal = target_covariance.diagonal(dim1=-2, dim2=-1)
        stats = {
            "matched_queries": float(target_boxes_matched.shape[0]),
            "valid_attention_mass": float(matched_mass.detach().mean()),
            "off_diagonal_abs": float(
                predicted_covariance_matched[:, 0, 1].detach().abs().mean()
            ),
            "predicted_variance": float(diagonal.detach().mean()),
            "target_variance": float(target_diagonal.detach().mean()),
        }

        details: list[dict[str, float | int]] = []
        if return_details:
            center_cpu = center_components.detach().cpu()
            covariance_cpu = covariance_per_match.detach().cpu()
            diagonal_cpu = diagonal.detach().cpu()
            target_diagonal_cpu = target_diagonal.detach().cpu()
            off_diagonal_cpu = predicted_covariance_matched[:, 0, 1].detach().abs().cpu()
            mass_cpu = matched_mass.detach().cpu()
            boxes_cpu = target_boxes_matched.detach().cpu()
            for index in range(len(boxes_cpu)):
                batch_index = batch_indices[index]
                original_size = labels[batch_index].get("orig_size")
                if original_size is None:
                    area_pixels = float("nan")
                else:
                    height, width = original_size.detach().cpu().tolist()
                    area_pixels = float(
                        boxes_cpu[index, 2] * width * boxes_cpu[index, 3] * height
                    )
                details.append(
                    {
                        "batch_index": batch_index,
                        "query_index": int(query_indices[index]),
                        "target_index": int(target_indices[index]),
                        "target_area_pixels": area_pixels,
                        "center_error_x": float(center_cpu[index, 0]),
                        "center_error_y": float(center_cpu[index, 1]),
                        "center_error_l1": float(center_cpu[index].sum()),
                        "covariance_error_l1": float(covariance_cpu[index]),
                        "predicted_var_x": float(diagonal_cpu[index, 0]),
                        "predicted_var_y": float(diagonal_cpu[index, 1]),
                        "target_var_x": float(target_diagonal_cpu[index, 0]),
                        "target_var_y": float(target_diagonal_cpu[index, 1]),
                        "off_diagonal_abs": float(off_diagonal_cpu[index]),
                        "valid_attention_mass": float(mass_cpu[index]),
                    }
                )
        return moment_loss, center_loss, covariance_loss, stats, details

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
        labels: list[dict[str, torch.Tensor]] | None = None,
        *,
        moment_weight: float = 0.0,
        return_moment_details: bool = False,
    ) -> dict[str, Any]:
        if moment_weight < 0:
            raise ValueError("moment_weight must be non-negative")
        capture_moments = labels is not None and (
            (self.training and moment_weight > 0) or return_moment_details
        )
        self._attention_cache = {}
        hook_handle = None
        if capture_moments:
            hook_handle = self.final_cross_attention.register_forward_hook(
                self._capture_final_cross_attention, with_kwargs=True
            )
        try:
            outputs = self.detector(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        result: dict[str, Any] = {
            "outputs": outputs,
            "loss": outputs.loss,
            "main_loss": outputs.loss,
            "moment_loss": None,
            "center_loss": None,
            "covariance_loss": None,
            "moment_executed": False,
            "matched_queries": 0,
            "moment_stats": {},
            "moment_details": [],
        }
        if not capture_moments:
            return result

        moment_loss, center_loss, covariance_loss, stats, details = self._compute_moment_loss(
            outputs, labels, return_details=return_moment_details
        )
        self.moment_forward_calls += 1
        result.update(
            {
                "moment_loss": moment_loss,
                "center_loss": center_loss,
                "covariance_loss": covariance_loss,
                "moment_executed": True,
                "matched_queries": int(stats["matched_queries"]),
                "moment_stats": stats,
                "moment_details": details,
            }
        )
        if self.training and moment_weight > 0:
            result["loss"] = outputs.loss + moment_weight * moment_loss
        return result


def make_moment_model(
    config: ExperimentConfig,
    seed: int | None = None,
    *,
    center_weight: float = 1.0,
    covariance_weight: float = 1.0,
) -> tuple[MomentGuidedDeformableDetr, str]:
    seed = config.seed if seed is None else seed
    detector = build_detector(config, seed)
    _assert_bbox_heads_are_tied(detector)
    model = MomentGuidedDeformableDetr(
        detector,
        loss_weights=MomentLossWeights(center_weight, covariance_weight),
    ).to(config.device)
    if model.extra_parameter_count != 0:
        raise AssertionError("Moment supervision unexpectedly added inference parameters")
    return model, model_fingerprint(model.detector)
