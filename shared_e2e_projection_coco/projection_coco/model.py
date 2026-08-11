from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, DeformableDetrForObjectDetection

from .config import TrainConfig, model_fingerprint, seed_everything


def inverse_sigmoid(tensor: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    tensor = tensor.clamp(0, 1)
    return torch.log(tensor.clamp(min=eps) / (1 - tensor).clamp(min=eps))


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1, area2 = box_area(boxes1), box_area(boxes2)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    size = (right_bottom - left_top).clamp(min=0)
    intersection = size[..., 0] * size[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-7), union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou(boxes1, boxes2)
    left_top = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    size = (right_bottom - left_top).clamp(min=0)
    enclosing = (size[..., 0] * size[..., 1]).clamp(min=1e-7)
    return iou - (enclosing - union) / enclosing


class ResidualPatchAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.fc2(F.gelu(self.fc1(self.norm(features))))


def build_detector(config: TrainConfig):
    seed_everything(config.seed, config.deterministic)
    try:
        model_config = AutoConfig.from_pretrained(
            config.model_name, local_files_only=config.offline
        )
    except OSError as error:
        if config.offline:
            raise RuntimeError(
                f"Model config is not cached for offline use: {config.model_name}"
            ) from error
        raise
    if model_config.model_type != "deformable_detr":
        raise ValueError(f"Expected Deformable DETR, got {model_config.model_type}")
    if model_config.two_stage or model_config.with_box_refine:
        raise ValueError("This projection implementation expects one-stage tied bbox heads")
    if model_config.num_feature_levels != 4:
        raise ValueError("This projection implementation expects four feature levels")
    if model_config.num_labels < 91:
        raise ValueError(
            "The pretrained COCO classifier must expose the original 91 label slots"
        )
    model_config.auxiliary_loss = False
    model_config.disable_custom_kernels = config.disable_custom_kernels
    detector = DeformableDetrForObjectDetection.from_pretrained(
        config.model_name,
        config=model_config,
        local_files_only=config.offline,
    )
    pointers = [
        tuple(parameter.data_ptr() for parameter in head.parameters())
        for head in detector.bbox_embed
    ]
    if len(set(pointers)) != 1:
        raise RuntimeError("Deformable DETR bbox heads are not tied")
    return detector


class SharedE2ERepresentationProjected(nn.Module):
    """Shared-E2E V2 with projection at the shared encoder representation."""

    def __init__(self, detector, feature_level: int = 0) -> None:
        super().__init__()
        self.detector = detector
        self.feature_level = feature_level
        self.adapter = ResidualPatchAdapter(detector.config.d_model)
        self._encoder_cache: dict[str, torch.Tensor] = {}
        self.aux_forward_calls = 0
        self._hook_handle = self.detector.model.encoder.register_forward_hook(
            self._capture_encoder, with_kwargs=True
        )

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @property
    def shared_bbox_head(self):
        return self.detector.bbox_embed[-1]

    @property
    def encoder_representation(self) -> torch.Tensor:
        if not self._encoder_cache:
            raise RuntimeError("Encoder cache is empty; run a detector forward first")
        return self._encoder_cache["memory"]

    def _capture_encoder(self, module, args, kwargs, output) -> None:
        del module, args
        required = {"spatial_shapes", "level_start_index", "attention_mask"}
        missing = required.difference(kwargs)
        if missing:
            raise RuntimeError(f"Unsupported Transformers encoder API. Missing: {missing}")
        self._encoder_cache = {
            "memory": output.last_hidden_state,
            "spatial_shapes": kwargs["spatial_shapes"],
            "level_start_index": kwargs["level_start_index"],
            "attention_mask": kwargs["attention_mask"],
        }

    def _valid_level_layout(self, batch_index: int):
        cache = self._encoder_cache
        level = self.feature_level
        full_height, full_width = [
            int(value) for value in cache["spatial_shapes"][level].tolist()
        ]
        start = int(cache["level_start_index"][level].item())
        valid_mask = cache["attention_mask"][
            batch_index, start : start + full_height * full_width
        ].reshape(full_height, full_width)
        valid_height = int(valid_mask.any(dim=1).sum().item())
        valid_width = int(valid_mask.any(dim=0).sum().item())
        if valid_height <= 0 or valid_width <= 0:
            raise RuntimeError("Empty valid feature region")
        return start, full_width, valid_height, valid_width

    def select_aux_samples(self, labels: list[dict]) -> dict:
        memory = self.encoder_representation
        features = []
        references = []
        targets = []
        total = 0
        collisions = 0

        for batch_index, target in enumerate(labels):
            boxes = target["boxes"]
            object_count = len(boxes)
            total += object_count
            if object_count == 0:
                continue
            start, full_width, valid_height, valid_width = self._valid_level_layout(
                batch_index
            )
            centers = boxes[:, :2].clamp(min=0.0, max=1.0 - 1e-7)
            columns = torch.floor(centers[:, 0] * valid_width).long().clamp(
                0, valid_width - 1
            )
            rows = torch.floor(centers[:, 1] * valid_height).long().clamp(
                0, valid_height - 1
            )
            cells = rows * valid_width + columns
            keep = torch.zeros(object_count, dtype=torch.bool, device=boxes.device)
            for cell in cells.unique():
                indices = torch.where(cells == cell)[0]
                cell_row = torch.div(cell, valid_width, rounding_mode="floor")
                cell_column = cell % valid_width
                cell_center = boxes.new_tensor(
                    [
                        (float(cell_column.item()) + 0.5) / valid_width,
                        (float(cell_row.item()) + 0.5) / valid_height,
                    ]
                )
                distances = (centers[indices] - cell_center).square().sum(dim=-1)
                keep[indices[distances.argmin()]] = True

            kept_indices = torch.where(keep)[0]
            collisions += object_count - int(kept_indices.numel())
            for object_index in kept_indices.tolist():
                row = int(rows[object_index])
                column = int(columns[object_index])
                flat_index = start + row * full_width + column
                features.append(memory[batch_index, flat_index])
                references.append(
                    memory.new_tensor(
                        [
                            (column + 0.5) / valid_width,
                            (row + 0.5) / valid_height,
                        ]
                    )
                )
                targets.append(boxes[object_index])

        if not features:
            empty = memory.new_empty
            return {
                "features": empty((0, memory.shape[-1])),
                "references": empty((0, 2)),
                "targets": empty((0, 4)),
                "total": total,
                "collisions": collisions,
            }
        return {
            "features": torch.stack(features),
            "references": torch.stack(references),
            "targets": torch.stack(targets),
            "total": total,
            "collisions": collisions,
        }

    def decode_with_reference(
        self, features: torch.Tensor, references: torch.Tensor
    ) -> torch.Tensor:
        delta = self.shared_bbox_head(features)
        center_logits = delta[..., :2] + inverse_sigmoid(references)
        return torch.cat((center_logits, delta[..., 2:]), dim=-1).sigmoid()

    def auxiliary_loss(self, labels: list[dict], outputs):
        selected = self.select_aux_samples(labels)
        raw_features = selected["features"]
        used = len(raw_features)
        if used == 0:
            zero = self.encoder_representation.sum() * 0.0
            return zero, zero, zero, selected, {}
        adapted = self.adapter(raw_features)
        predicted = self.decode_with_reference(adapted, selected["references"])
        target_boxes = selected["targets"]
        loss_l1 = F.l1_loss(predicted, target_boxes, reduction="none").sum() / used
        giou = generalized_box_iou(
            cxcywh_to_xyxy(predicted), cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = (1.0 - torch.diag(giou)).sum() / used
        weighted = 5.0 * loss_l1 + 2.0 * loss_giou
        decoder_features = outputs.intermediate_hidden_states[:, -1]
        stats = {
            "raw_feature_norm": float(raw_features.detach().norm(dim=-1).mean()),
            "adapted_feature_norm": float(adapted.detach().norm(dim=-1).mean()),
            "decoder_feature_norm": float(
                decoder_features.detach().norm(dim=-1).mean()
            ),
        }
        return weighted, loss_l1, loss_giou, selected, stats

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
        labels: list[dict] | None = None,
        aux_weight: float = 0.0,
    ) -> dict:
        self._encoder_cache = {}
        outputs = self.detector(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )
        result = {
            "outputs": outputs,
            "loss": outputs.loss,
            "main_loss": outputs.loss,
            "aux_loss": None,
            "aux_l1": None,
            "aux_giou": None,
            "aux_executed": False,
            "aux_total": 0,
            "aux_used": 0,
            "aux_collisions": 0,
            "feature_stats": {},
        }
        if not self.training or labels is None or aux_weight <= 0:
            return result
        aux_loss, aux_l1, aux_giou, selected, stats = self.auxiliary_loss(
            labels, outputs
        )
        self.aux_forward_calls += 1
        result.update(
            {
                "loss": outputs.loss + aux_weight * aux_loss,
                "aux_loss": aux_loss,
                "aux_l1": aux_l1,
                "aux_giou": aux_giou,
                "aux_executed": True,
                "aux_total": selected["total"],
                "aux_used": len(selected["features"]),
                "aux_collisions": selected["collisions"],
                "feature_stats": stats,
            }
        )
        return result


def make_model(config: TrainConfig, device: torch.device):
    model = SharedE2ERepresentationProjected(
        build_detector(config), feature_level=config.feature_level
    ).to(device)
    return model, model_fingerprint(model.detector)
