from __future__ import annotations

import copy
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, DeformableDetrForObjectDetection

from .config import ID2LABEL, LABEL2ID, ExperimentConfig, model_fingerprint, seed_everything


def inverse_sigmoid(tensor, eps=1e-5):
    tensor = tensor.clamp(0, 1)
    return torch.log(tensor.clamp(min=eps) / (1 - tensor).clamp(min=eps))


def cxcywh_to_xyxy(boxes):
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack((
        cx - width / 2, cy - height / 2,
        cx + width / 2, cy + height / 2,
    ), dim=-1)


def box_area(boxes):
    return (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )


def box_iou(boxes1, boxes2):
    area1, area2 = box_area(boxes1), box_area(boxes2)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-7), union


def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    left_top = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    enclosing = (wh[..., 0] * wh[..., 1]).clamp(min=1e-7)
    return iou - (enclosing - union) / enclosing


class ResidualPatchAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features):
        return features + self.fc2(F.gelu(self.fc1(self.norm(features))))


def build_detector(config: ExperimentConfig, initialization_seed: int | None = None):
    seed_everything(config.seed if initialization_seed is None else initialization_seed)
    try:
        model_config = AutoConfig.from_pretrained(config.checkpoint, local_files_only=True)
        local_only = True
    except OSError:
        model_config = AutoConfig.from_pretrained(config.checkpoint)
        local_only = False
    assert model_config.model_type == "deformable_detr"
    assert model_config.two_stage is False
    assert model_config.with_box_refine is False
    assert model_config.num_feature_levels == 4
    model_config.num_labels = len(LABEL2ID)
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID
    model_config.auxiliary_loss = False
    model_config.disable_custom_kernels = config.disable_custom_kernels
    return DeformableDetrForObjectDetection.from_pretrained(
        config.checkpoint, config=model_config,
        ignore_mismatched_sizes=True, local_files_only=local_only,
    )


def assert_hf_bbox_heads_are_tied(detector):
    assert not detector.config.with_box_refine
    pointers = [tuple(parameter.data_ptr() for parameter in head.parameters())
                for head in detector.bbox_embed]
    assert len(set(pointers)) == 1, "DETR bbox heads are not shared"
    return pointers[0]


class GTDeformableDetr(nn.Module):
    VALID_MODES = {
        "baseline", "separate", "shared_detach", "shared_e2e",
        "shared_decay", "shared_late_decay", "random_patch", "no_adapter",
    }

    def __init__(self, detector, mode: str, feature_level: int = 0):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.detector = detector
        self.mode = mode
        self.feature_level = feature_level
        self.adapter = ResidualPatchAdapter(detector.config.d_model)
        self.separate_bbox_head = copy.deepcopy(detector.bbox_embed[-1])
        self._encoder_cache = {}
        self.aux_forward_calls = 0
        self._hook_handle = self.detector.model.encoder.register_forward_hook(
            self._capture_encoder, with_kwargs=True
        )

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @property
    def shared_bbox_head(self):
        return self.detector.bbox_embed[-1]

    def _capture_encoder(self, module, args, kwargs, output):
        required = {"spatial_shapes", "level_start_index", "attention_mask"}
        missing = required.difference(kwargs)
        if missing:
            raise RuntimeError(f"Unsupported Transformers encoder API. Missing: {missing}")
        self._encoder_cache = {
            "memory": output.last_hidden_state,
            "spatial_shapes": kwargs["spatial_shapes"],
            "level_start_index": kwargs["level_start_index"],
            "attention_mask": kwargs["attention_mask"],
            "valid_ratios": kwargs.get("valid_ratios"),
        }

    def decode_with_reference(self, features, reference_xy, head=None):
        head = self.shared_bbox_head if head is None else head
        delta = head(features)
        center_logits = delta[..., :2] + inverse_sigmoid(reference_xy)
        return torch.cat((center_logits, delta[..., 2:]), dim=-1).sigmoid()

    def _valid_level_layout(self, batch_index):
        cache = self._encoder_cache
        level = self.feature_level
        full_h, full_w = [int(value) for value in cache["spatial_shapes"][level].tolist()]
        start = int(cache["level_start_index"][level].item())
        valid_mask = cache["attention_mask"][
            batch_index, start:start + full_h * full_w
        ].reshape(full_h, full_w)
        valid_h = int(valid_mask.any(dim=1).sum().item())
        valid_w = int(valid_mask.any(dim=0).sum().item())
        if valid_h <= 0 or valid_w <= 0:
            raise RuntimeError("Empty valid feature region")
        return start, full_h, full_w, valid_h, valid_w

    def select_aux_samples(self, labels, selector="aligned"):
        if selector not in {"aligned", "random"}:
            raise ValueError("selector must be 'aligned' or 'random'")
        if not self._encoder_cache:
            raise RuntimeError("Encoder cache is empty; detector forward must run first")
        memory = self._encoder_cache["memory"]
        feature_list, reference_list, target_list, flat_indices = [], [], [], []
        total, collision_targets = 0, 0

        for batch_index, target in enumerate(labels):
            boxes = target["boxes"]
            number = len(boxes)
            total += number
            if number == 0:
                continue
            start, _, full_w, valid_h, valid_w = self._valid_level_layout(batch_index)

            # First determine one common target subset from aligned cells.
            centers = boxes[:, :2].clamp(min=0.0, max=1.0 - 1e-7)
            aligned_cols = torch.floor(centers[:, 0] * valid_w).long().clamp(0, valid_w - 1)
            aligned_rows = torch.floor(centers[:, 1] * valid_h).long().clamp(0, valid_h - 1)
            aligned_cells = aligned_rows * valid_w + aligned_cols
            keep = torch.zeros(number, dtype=torch.bool, device=boxes.device)
            for cell in aligned_cells.unique():
                indices = torch.where(aligned_cells == cell)[0]
                cell_row = torch.div(cell, valid_w, rounding_mode="floor")
                cell_col = cell % valid_w
                cell_center = boxes.new_tensor([
                    (float(cell_col.item()) + 0.5) / valid_w,
                    (float(cell_row.item()) + 0.5) / valid_h,
                ])
                distances = (centers[indices] - cell_center).square().sum(dim=-1)
                keep[indices[distances.argmin()]] = True

            kept_indices = torch.where(keep)[0]
            collision_targets += number - int(kept_indices.numel())
            rows, cols = aligned_rows.clone(), aligned_cols.clone()
            if selector == "random":
                cell_count = valid_h * valid_w
                random_cells = torch.randperm(cell_count, device=boxes.device)[:kept_indices.numel()]
                rows[kept_indices] = torch.div(random_cells, valid_w, rounding_mode="floor")
                cols[kept_indices] = random_cells % valid_w

            for object_index in kept_indices.tolist():
                row, col = int(rows[object_index]), int(cols[object_index])
                flat_index = start + row * full_w + col
                feature_list.append(memory[batch_index, flat_index])
                reference_list.append(memory.new_tensor([
                    (col + 0.5) / valid_w, (row + 0.5) / valid_h,
                ]))
                target_list.append(boxes[object_index])
                flat_indices.append((batch_index, flat_index))

        if not feature_list:
            empty = memory.new_empty
            return {
                "features": empty((0, memory.shape[-1])), "references": empty((0, 2)),
                "targets": empty((0, 4)), "flat_indices": flat_indices,
                "total": total, "collision_targets": collision_targets,
            }
        return {
            "features": torch.stack(feature_list), "references": torch.stack(reference_list),
            "targets": torch.stack(target_list), "flat_indices": flat_indices,
            "total": total, "collision_targets": collision_targets,
        }

    def auxiliary_loss(self, labels, outputs):
        selector = "random" if self.mode == "random_patch" else "aligned"
        selected = self.select_aux_samples(labels, selector=selector)
        raw_features = selected["features"]
        used = len(raw_features)
        if used == 0:
            zero = self._encoder_cache["memory"].sum() * 0.0
            return zero, zero, zero, selected, {}
        if self.mode == "shared_detach":
            raw_features = raw_features.detach()
        adapted = raw_features if self.mode == "no_adapter" else self.adapter(raw_features)
        head = self.separate_bbox_head if self.mode == "separate" else self.shared_bbox_head
        predicted = self.decode_with_reference(adapted, selected["references"], head=head)
        target_boxes = selected["targets"]
        loss_l1 = F.l1_loss(predicted, target_boxes, reduction="none").sum() / used
        giou = generalized_box_iou(cxcywh_to_xyxy(predicted), cxcywh_to_xyxy(target_boxes))
        loss_giou = (1.0 - torch.diag(giou)).sum() / used
        weighted = 5.0 * loss_l1 + 2.0 * loss_giou
        decoder_features = outputs.intermediate_hidden_states[:, -1]
        stats = {
            "raw_feature_norm": float(raw_features.detach().norm(dim=-1).mean()),
            "adapted_feature_norm": float(adapted.detach().norm(dim=-1).mean()),
            "decoder_feature_norm": float(decoder_features.detach().norm(dim=-1).mean()),
        }
        return weighted, loss_l1, loss_giou, selected, stats

    def forward(self, pixel_values, pixel_mask=None, labels=None, aux_weight=0.0):
        self._encoder_cache = {}
        outputs = self.detector(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        result = {
            "outputs": outputs, "loss": outputs.loss, "main_loss": outputs.loss,
            "aux_loss": None, "aux_l1": None, "aux_giou": None,
            "aux_executed": False, "aux_total": 0, "aux_used": 0,
            "aux_collisions": 0, "feature_stats": {},
        }
        should_run_aux = self.training and labels is not None and self.mode != "baseline" and aux_weight > 0
        if not should_run_aux:
            return result
        aux_loss, aux_l1, aux_giou, selected, stats = self.auxiliary_loss(labels, outputs)
        self.aux_forward_calls += 1
        result.update({
            "loss": outputs.loss + aux_weight * aux_loss,
            "aux_loss": aux_loss, "aux_l1": aux_l1, "aux_giou": aux_giou,
            "aux_executed": True, "aux_total": selected["total"],
            "aux_used": len(selected["features"]),
            "aux_collisions": selected["collision_targets"], "feature_stats": stats,
        })
        return result


def make_model(config: ExperimentConfig, experiment: str, seed: int | None = None):
    seed = config.seed if seed is None else seed
    detector = build_detector(config, seed)
    assert_hf_bbox_heads_are_tied(detector)
    model = GTDeformableDetr(detector, experiment, config.feature_level).to(config.device)
    return model, model_fingerprint(model.detector)
