from __future__ import annotations

import time

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm


@torch.inference_mode()
def evaluate_main(model, val_loader, processor, device: torch.device) -> dict[str, float]:
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    calls_before = model.aux_forward_calls
    start = time.perf_counter()
    for batch in tqdm(
        val_loader,
        desc="COCO validation",
        leave=False,
        mininterval=0.5,
    ):
        result = model(
            pixel_values=batch["pixel_values"].to(device, non_blocking=True),
            pixel_mask=batch["pixel_mask"].to(device, non_blocking=True),
            labels=None,
        )
        if result["aux_executed"]:
            raise RuntimeError("Auxiliary branch executed during main-only validation")
        target_sizes = torch.stack(
            [target["orig_size"] for target in batch["eval_targets"]]
        ).to(device)
        predictions = processor.post_process_object_detection(
            result["outputs"], threshold=0.0, target_sizes=target_sizes
        )
        predictions = [
            {key: value.detach().cpu() for key, value in prediction.items()}
            for prediction in predictions
        ]
        targets = [
            {
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            }
            for target in batch["eval_targets"]
        ]
        metric.update(predictions, targets)
    if model.aux_forward_calls != calls_before:
        raise RuntimeError("Validation changed the auxiliary forward counter")
    values = metric.compute()

    def clean(name: str) -> float:
        value = float(values[name])
        return value if value >= 0 else float("nan")

    return {
        "map": clean("map"),
        "map50": clean("map_50"),
        "map75": clean("map_75"),
        "map_small": clean("map_small"),
        "map_medium": clean("map_medium"),
        "map_large": clean("map_large"),
        "mar100": clean("mar_100"),
        "val_seconds": time.perf_counter() - start,
    }
