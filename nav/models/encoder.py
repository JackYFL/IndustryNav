"""Shared visual-encoder building blocks for the BC policies.

The cnn / resnet / dino "bases" the benchmark trains differ *only* in the
``timm`` backbone string handed to :class:`TimmEncoder` — there is no
per-backbone subclass. The policy heads in :mod:`nav.models.policy` build one
or two of these encoders (RGB and/or depth) via :func:`build_encoder_pair`,
which also centralizes the ``half_width`` / ViT-vs-ResNet kwarg handling that
was duplicated across the recurrent / transformer / diffusion policies.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_VIT_KEYWORDS = ("vit", "dino", "siglip", "swin", "beit", "eva", "deit", "clip")


def is_vit_like(model_name: str) -> bool:
    name = model_name.lower()
    return any(k in name for k in _VIT_KEYWORDS)


def validate_pretrained_half_width(
    model_name: str, pretrained: bool, half_width: bool, branch: str
) -> None:
    """Reject the one incompatible combo: pretrained ResNet + halved widths.

    ``half_width`` rewrites ResNet stem/base channel widths, so timm's
    pretrained weights no longer fit. ViT-like backbones ignore the width
    kwargs entirely, so the combo is only invalid for ResNet-style names.
    """
    if pretrained and half_width and not is_vit_like(model_name):
        raise ValueError(
            f"Incompatible settings for {branch}: pretrained=True with half_width=True for "
            f"backbone '{model_name}'. half_width changes ResNet channel widths, so timm "
            f"pretrained weights cannot be loaded. Set half_width=False or disable pretrained."
        )


class TimmEncoder(nn.Module):
    """A timm backbone with global average pooling and a projection to ``out_dim``."""

    def __init__(
        self,
        model_name: str,
        in_channels: int,
        out_dim: int,
        pretrained: bool = False,
        img_size: int = 256,
        **create_kwargs,
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "timm is required for TimmEncoder. Install with `pip install timm`."
            ) from exc

        # ViT-like models need img_size for positional embedding interpolation.
        if is_vit_like(model_name):
            create_kwargs.setdefault("img_size", img_size)

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="avg",
            **create_kwargs,
        )

        feat_dim = getattr(self.backbone, "num_features", out_dim)
        self.proj = nn.Linear(feat_dim, out_dim) if feat_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


def _half_width_kwargs(model_name: str, half_width: bool) -> Dict[str, int]:
    """ResNet stem/base width override; empty for ViT-like backbones."""
    if half_width and not is_vit_like(model_name):
        return {"stem_width": 32, "base_width": 32}
    return {}


def build_encoder_pair(
    *,
    use_rgb: bool,
    use_depth: bool,
    visual_dim: int,
    rgb_backbone: str,
    depth_backbone: str,
    pretrained_rgb: bool,
    pretrained_depth: bool,
    img_size: int,
    half_width: bool = False,
) -> Tuple[Optional[TimmEncoder], Optional[TimmEncoder]]:
    """Construct the (rgb_encoder, depth_encoder) pair shared by every policy.

    Either may be ``None`` when its modality is disabled. ``half_width`` is
    validated against pretrained weights per active branch before building.
    """
    rgb_encoder = None
    if use_rgb:
        validate_pretrained_half_width(rgb_backbone, pretrained_rgb, half_width, "rgb")
        rgb_encoder = TimmEncoder(
            model_name=rgb_backbone,
            in_channels=3,
            out_dim=visual_dim,
            pretrained=pretrained_rgb,
            img_size=img_size,
            **_half_width_kwargs(rgb_backbone, half_width),
        )

    depth_encoder = None
    if use_depth:
        validate_pretrained_half_width(depth_backbone, pretrained_depth, half_width, "depth")
        depth_encoder = TimmEncoder(
            model_name=depth_backbone,
            in_channels=1,
            out_dim=visual_dim,
            pretrained=pretrained_depth,
            img_size=img_size,
            **_half_width_kwargs(depth_backbone, half_width),
        )

    return rgb_encoder, depth_encoder
