"""Navigation BC policy heads.

Four interchangeable policies share the visual-encoder pair from
:mod:`nav.models.encoder`:

- :class:`NavPolicy` — feed-forward MLP fusion over a single frame + a numeric
  state vector (the ``mlp`` base).
- :class:`NavPolicyRNN` — LSTM over a window of (goal, prev-action[, vision]).
- :class:`NavPolicyTransformer` — causal transformer over the same window.
- :class:`NavPolicyDiffusion` — diffusion action head conditioned on a
  transformer context encoder.

:func:`build_policy` is the single constructor the training loop and inference
controller use; it dispatches on ``cfg.policy_type``.

The module attribute layout (``rgb_encoder`` / ``depth_encoder`` / ``goal_mlp``
/ ``action_embed`` / ``encoder`` …) is kept flat and stable so existing
checkpoints still load with ``strict=True``; only the duplicated encoder-pair
*construction* is factored into :func:`nav.models.encoder.build_encoder_pair`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nav.config import NAV_DEFAULT_BACKBONE, NAV_POLICY_ARCH as _ARCH
from nav.models.encoder import build_encoder_pair


class NavPolicy(nn.Module):
    """Single-frame feed-forward policy (RGB [+ depth] + numeric state)."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int = 4,
        use_depth: bool = True,
        visual_dim: int = _ARCH.visual_dim,
        state_hidden: int = _ARCH.state_hidden,
        fusion_hidden: int = _ARCH.fusion_hidden,
        dropout: float = _ARCH.dropout,
        rgb_backbone: str = NAV_DEFAULT_BACKBONE,
        depth_backbone: str = NAV_DEFAULT_BACKBONE,
        pretrained_rgb: bool = False,
        pretrained_depth: bool = False,
        img_size: int = 256,
        chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.use_depth = use_depth
        self.chunk_size = chunk_size
        self.num_actions = num_actions

        self.rgb_encoder, self.depth_encoder = build_encoder_pair(
            use_rgb=True,
            use_depth=use_depth,
            visual_dim=visual_dim,
            rgb_backbone=rgb_backbone,
            depth_backbone=depth_backbone,
            pretrained_rgb=pretrained_rgb,
            pretrained_depth=pretrained_depth,
            img_size=img_size,
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, state_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(state_hidden, state_hidden),
            nn.ReLU(inplace=True),
        )

        fusion_in = visual_dim + (visual_dim if use_depth else 0) + state_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_actions * chunk_size),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        feats = [self.rgb_encoder(rgb)]
        if self.use_depth:
            feats.append(self.depth_encoder(depth))
        feats.append(self.state_mlp(state))
        out = self.fusion(torch.cat(feats, dim=1))
        if self.chunk_size > 1:
            return out.reshape(out.shape[0], self.chunk_size, self.num_actions)
        return out


def _goal_mlp(goal_dim: int, goal_hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(goal_dim, goal_hidden),
        nn.ReLU(inplace=True),
        nn.Linear(goal_hidden, goal_hidden),
        nn.ReLU(inplace=True),
    )


def _encode_sequence_feats(policy, rgb, depth, goal, prev_action):
    """Build the per-step token features shared by the RNN/transformer heads.

    Returns the concatenated (B, T, token_dim) tensor of
    [rgb?, depth?, goal, prev_action] features. ``policy`` supplies the
    ``use_rgb`` / ``use_depth`` flags and the encoder / goal-MLP / action-embed
    submodules (kept as flat attributes for checkpoint stability).
    """
    B, T = goal.shape[0], goal.shape[1]
    feats = []
    if policy.use_rgb:
        feats.append(policy.rgb_encoder(rgb.reshape(B * T, *rgb.shape[2:])).reshape(B, T, -1))
    if policy.use_depth:
        feats.append(policy.depth_encoder(depth.reshape(B * T, *depth.shape[2:])).reshape(B, T, -1))
    feats.append(policy.goal_mlp(goal))
    feats.append(policy.action_embed(prev_action))
    return torch.cat(feats, dim=-1)


class NavPolicyRNN(nn.Module):
    def __init__(
        self,
        goal_dim: int,
        num_actions: int = 4,
        use_depth: bool = True,
        use_rgb: bool = False,
        visual_dim: int = _ARCH.visual_dim,
        goal_hidden: int = _ARCH.goal_hidden,
        action_embed: int = _ARCH.action_embed,
        lstm_hidden: int = _ARCH.lstm_hidden,
        num_layers: int = 2,
        dropout: float = _ARCH.dropout,
        rgb_backbone: str = NAV_DEFAULT_BACKBONE,
        depth_backbone: str = NAV_DEFAULT_BACKBONE,
        pretrained_rgb: bool = False,
        pretrained_depth: bool = False,
        half_width: bool = True,
        img_size: int = 256,
        chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.use_depth = use_depth
        self.use_rgb = use_rgb
        self.chunk_size = chunk_size
        self.num_actions = num_actions

        self.rgb_encoder, self.depth_encoder = build_encoder_pair(
            use_rgb=use_rgb, use_depth=use_depth, visual_dim=visual_dim,
            rgb_backbone=rgb_backbone, depth_backbone=depth_backbone,
            pretrained_rgb=pretrained_rgb, pretrained_depth=pretrained_depth,
            img_size=img_size, half_width=half_width,
        )
        self.goal_mlp = _goal_mlp(goal_dim, goal_hidden)
        self.action_embed = nn.Embedding(num_actions, action_embed)

        visual_in = (visual_dim if use_rgb else 0) + (visual_dim if use_depth else 0)
        lstm_in = visual_in + goal_hidden + action_embed
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, num_actions * chunk_size),
        )

    def forward(self, rgb, depth, goal, prev_action) -> torch.Tensor:
        B = goal.shape[0]
        x = _encode_sequence_feats(self, rgb, depth, goal, prev_action)
        out, _ = self.lstm(x)
        out = self.head(out[:, -1, :])
        if self.chunk_size > 1:
            return out.reshape(B, self.chunk_size, self.num_actions)
        return out


class NavPolicyTransformer(nn.Module):
    def __init__(
        self,
        goal_dim: int,
        num_actions: int = 4,
        use_depth: bool = True,
        use_rgb: bool = False,
        visual_dim: int = _ARCH.visual_dim,
        goal_hidden: int = _ARCH.goal_hidden,
        action_embed: int = _ARCH.action_embed,
        d_model: int = _ARCH.d_model,
        nhead: int = _ARCH.nhead,
        num_layers: int = 2,
        dropout: float = _ARCH.dropout,
        seq_len: int = 8,
        rgb_backbone: str = NAV_DEFAULT_BACKBONE,
        depth_backbone: str = NAV_DEFAULT_BACKBONE,
        pretrained_rgb: bool = False,
        pretrained_depth: bool = False,
        half_width: bool = True,
        img_size: int = 256,
        chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.use_depth = use_depth
        self.use_rgb = use_rgb
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.num_actions = num_actions

        self.rgb_encoder, self.depth_encoder = build_encoder_pair(
            use_rgb=use_rgb, use_depth=use_depth, visual_dim=visual_dim,
            rgb_backbone=rgb_backbone, depth_backbone=depth_backbone,
            pretrained_rgb=pretrained_rgb, pretrained_depth=pretrained_depth,
            img_size=img_size, half_width=half_width,
        )
        self.goal_mlp = _goal_mlp(goal_dim, goal_hidden)
        self.action_embed = nn.Embedding(num_actions, action_embed)

        visual_in = (visual_dim if use_rgb else 0) + (visual_dim if use_depth else 0)
        token_in = visual_in + goal_hidden + action_embed
        self.token_proj = nn.Linear(token_in, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions * chunk_size),
        )

    def _encode_tokens(self, rgb, depth, goal, prev_action) -> torch.Tensor:
        x = self.token_proj(_encode_sequence_feats(self, rgb, depth, goal, prev_action))
        T = x.shape[1]
        return x + self.pos_embed[:, :T, :]

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        return self.encoder(x, mask=causal_mask)

    def forward(self, rgb, depth, goal, prev_action) -> torch.Tensor:
        B = goal.shape[0]
        out = self._run_encoder(self._encode_tokens(rgb, depth, goal, prev_action))
        out = self.head(out[:, -1, :])
        if self.chunk_size > 1:
            return out.reshape(B, self.chunk_size, self.num_actions)
        return out


class NavPolicyDiffusion(nn.Module):
    def __init__(
        self,
        goal_dim: int,
        num_actions: int = 4,
        use_depth: bool = True,
        use_rgb: bool = False,
        visual_dim: int = _ARCH.visual_dim,
        goal_hidden: int = _ARCH.goal_hidden,
        action_embed: int = _ARCH.action_embed,
        d_model: int = _ARCH.d_model,
        nhead: int = _ARCH.nhead,
        num_layers: int = 2,
        dropout: float = _ARCH.dropout,
        seq_len: int = 8,
        rgb_backbone: str = NAV_DEFAULT_BACKBONE,
        depth_backbone: str = NAV_DEFAULT_BACKBONE,
        pretrained_rgb: bool = False,
        pretrained_depth: bool = False,
        half_width: bool = True,
        img_size: int = 256,
        diffusion_steps: int = _ARCH.diffusion_steps,
        beta_start: float = _ARCH.diffusion_beta_start,
        beta_end: float = _ARCH.diffusion_beta_end,
        chunk_size: int = 1,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.diffusion_steps = diffusion_steps
        self.chunk_size = chunk_size
        self._action_flat = num_actions * chunk_size

        # The context encoder always emits single-step features; the diffusion
        # head owns the chunk dimension via _action_flat.
        self.context_encoder = NavPolicyTransformer(
            goal_dim=goal_dim, num_actions=num_actions, use_depth=use_depth, use_rgb=use_rgb,
            visual_dim=visual_dim, goal_hidden=goal_hidden, action_embed=action_embed,
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout, seq_len=seq_len,
            rgb_backbone=rgb_backbone, depth_backbone=depth_backbone,
            pretrained_rgb=pretrained_rgb, pretrained_depth=pretrained_depth,
            half_width=half_width, img_size=img_size, chunk_size=1,
        )

        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True))
        self.denoiser = nn.Sequential(
            nn.Linear(d_model + self._action_flat + d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, self._action_flat),
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

    def _time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device)], dim=1)
        return emb

    def _context(self, rgb, depth, goal, prev_action) -> torch.Tensor:
        x = self.context_encoder._encode_tokens(rgb, depth, goal, prev_action)
        return self.context_encoder._run_encoder(x)[:, -1, :]

    def _predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self._time_embedding(t, context.shape[1]))
        return self.denoiser(torch.cat([context, x_t, t_emb], dim=1))

    def _action_onehot(self, action: torch.Tensor) -> torch.Tensor:
        x0 = torch.nn.functional.one_hot(action, num_classes=self.num_actions).float()
        return x0.reshape(action.shape[0], self._action_flat)

    def diffusion_loss(self, rgb, depth, goal, prev_action, action: torch.Tensor) -> torch.Tensor:
        context = self._context(rgb, depth, goal, prev_action)
        t = torch.randint(0, self.diffusion_steps, (action.shape[0],), device=action.device)
        x0 = self._action_onehot(action)
        noise = torch.randn_like(x0)
        a_bar = self.alpha_bar[t].unsqueeze(1)
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
        return torch.mean((self._predict_noise(x_t, t, context) - noise) ** 2)

    @torch.no_grad()
    def sample_logits(self, rgb, depth, goal, prev_action, steps: int = 50) -> torch.Tensor:
        B = goal.shape[0]
        context = self._context(rgb, depth, goal, prev_action)
        x_t = torch.randn(B, self._action_flat, device=goal.device)
        steps = min(steps, self.diffusion_steps)
        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=goal.device, dtype=torch.long)
            a_bar = self.alpha_bar[t].unsqueeze(1)
            eps = self._predict_noise(x_t, t, context)
            x0_pred = (x_t - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar)
            if i == 0:
                x_t = x0_pred
            else:
                a_bar_prev = self.alpha_bar[t - 1].unsqueeze(1)
                x_t = torch.sqrt(a_bar_prev) * x0_pred + torch.sqrt(1.0 - a_bar_prev) * torch.randn_like(x_t)
        if self.chunk_size > 1:
            return x_t.reshape(B, self.chunk_size, self.num_actions)
        return x_t


def build_policy(cfg, input_dim: int, num_actions: int = 4) -> nn.Module:
    """Construct a policy from a :class:`nav.config.BCTrainConfig`.

    ``input_dim`` is the dataset-derived feature width: the numeric state
    vector length for the ``mlp`` policy, or the goal-vector length for the
    sequence policies (``lstm`` / ``transformer`` / ``diffusion``).
    """
    pt = cfg.policy_type
    if pt == "mlp":
        return NavPolicy(
            state_dim=input_dim, num_actions=num_actions, use_depth=cfg.use_depth,
            rgb_backbone=cfg.rgb_backbone, depth_backbone=cfg.depth_backbone,
            pretrained_rgb=cfg.pretrained_rgb, pretrained_depth=cfg.pretrained_depth,
            img_size=cfg.img_size, chunk_size=cfg.chunk_size,
        )

    shared = dict(
        goal_dim=input_dim, num_actions=num_actions, use_depth=cfg.use_depth, use_rgb=cfg.use_rgb,
        num_layers=cfg.num_layers, rgb_backbone=cfg.rgb_backbone, depth_backbone=cfg.depth_backbone,
        pretrained_rgb=cfg.pretrained_rgb, pretrained_depth=cfg.pretrained_depth,
        half_width=cfg.half_width, img_size=cfg.img_size, chunk_size=cfg.chunk_size,
    )
    if pt == "lstm":
        return NavPolicyRNN(**shared)
    if pt == "transformer":
        return NavPolicyTransformer(seq_len=cfg.seq_len, **shared)
    if pt == "diffusion":
        return NavPolicyDiffusion(seq_len=cfg.seq_len, **shared)
    raise ValueError(f"Unknown policy_type: {pt!r}")
