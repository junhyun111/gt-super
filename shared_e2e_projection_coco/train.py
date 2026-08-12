from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from projection_coco.config import TrainConfig, configure_huggingface_cache
from projection_coco.data import prepare_data
from projection_coco.distributed import cleanup_distributed, initialize_distributed
from projection_coco.engine import train


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent


def optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train only Shared-E2E V2 representation projection on COCO 2017. "
            "Use python for one GPU or torchrun for multi-GPU DDP."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("COCO_ROOT", REPOSITORY_ROOT / "data" / "coco")),
        help="Directory containing train2017, val2017, and annotations",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("OUTPUT_ROOT", PROJECT_DIR / "artifacts")),
        help="Persistent checkpoint and CSV output directory",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=os.environ.get("HF_HOME"),
        help="Persistent Hugging Face cache directory",
    )
    parser.add_argument("--model-name", default="SenseTime/deformable-detr")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-GPU batch size"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-min-size", type=int, default=800)
    parser.add_argument("--image-max-size", type=int, default=1333)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--feature-level", type=int, default=0)
    parser.add_argument("--horizontal-flip-p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-limit",
        type=optional_positive_int,
        default=None,
        help="Optional deterministic COCO subset size for smoke tests",
    )
    parser.add_argument(
        "--val-limit",
        type=optional_positive_int,
        default=None,
        help="Optional deterministic validation subset size for smoke tests",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--gradient-log-every", type=int, default=100)
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path, or 'auto' to use OUTPUT_ROOT/.../latest.pt",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA float16 automatic mixed precision",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer determinism over cuDNN throughput",
    )
    parser.add_argument(
        "--disable-custom-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the portable PyTorch deformable-attention implementation",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid Hugging Face downloads and require a populated cache",
    )
    parser.add_argument(
        "--skip-initial-eval",
        action="store_true",
        help="Skip epoch-0 pretrained COCO evaluation",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution only for small smoke tests",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_huggingface_cache(args.hf_cache)
    config = TrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_min_size=args.image_min_size,
        image_max_size=args.image_max_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        aux_weight=args.aux_weight,
        feature_level=args.feature_level,
        horizontal_flip_p=args.horizontal_flip_p,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        eval_every=args.eval_every,
        save_every=args.save_every,
        gradient_log_every=args.gradient_log_every,
        amp=args.amp,
        deterministic=args.deterministic,
        disable_custom_kernels=args.disable_custom_kernels,
        offline=args.offline,
        skip_initial_eval=args.skip_initial_eval,
    )
    context = initialize_distributed(allow_cpu=args.allow_cpu)
    try:
        if context.is_main:
            print(config.to_json())
        bundle = prepare_data(config)
        context.barrier()
        train(config, bundle, context, resume_from=args.resume)
        context.barrier()
    finally:
        cleanup_distributed(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
