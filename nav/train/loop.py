"""Backward-compatible exports for behavior-cloning training.

New code should import from :mod:`nav.baselines.bc.trainer`.
"""

from nav.baselines.bc.trainer import (
    BCTrainer,
    evaluate,
    get_class_weights,
    initialize_from_checkpoint,
    mask_untrained_stop_logits,
    set_seed,
    train,
    turn_direction_loss,
)

__all__ = [
    "BCTrainer",
    "evaluate",
    "get_class_weights",
    "initialize_from_checkpoint",
    "mask_untrained_stop_logits",
    "set_seed",
    "train",
    "turn_direction_loss",
]
