"""Moment-Guided Deformable DETR experiment package.

The existing :mod:`gt_aux` package remains unchanged.  This package reuses its
dataset/configuration contract so the new experiment sees exactly the same VOC
split and image preprocessing as the stored baseline runs.
"""

from .model import MomentGuidedDeformableDetr, make_moment_model
from .train import MomentTrainingSettings, train_moment_experiment

__all__ = [
    "MomentGuidedDeformableDetr",
    "MomentTrainingSettings",
    "make_moment_model",
    "train_moment_experiment",
]
