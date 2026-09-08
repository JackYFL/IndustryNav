"""Train a safety-aware recurrent PointGoal policy with PPO or synchronous DDP-PPO."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Categorical

from nav.baselines.rl.pointgoal_ppo import (
    PPO_BOS_LABEL,
    PointGoalActorCritic,
    PointGoalPPOConfig,
    generalized_advantage_estimate,
    load_bc_depth_encoder,
)
from nav.baselines.rl.unity_pointgoal_env import UnityPointGoalEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--scene-limit", type=int, default=0)
    parser.add_argument("--episodes-per-scene", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=43000)
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--bptt-len", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--chunks-per-minibatch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-bc-checkpoint", type=Path, default=None)
    parser.add_argument("--depth-backbone", default="resnet50")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--visual-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--half-width", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pretrained-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--goal-distance-scale-m", type=float, default=50.0)
    parser.add_argument("--analytic-goal-prior-strength", type=float, default=3.0)
    parser.add_argument("--reach-m", type=float, default=2.0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--dynamic-objects", choices=("moving", "static"), default="moving")
    parser.add_argument("--ego-width", type=int, default=320)
    parser.add_argument("--ego-height", type=int, default=240)
    parser.add_argument("--minimap-width", type=int, default=862)
    parser.add_argument("--minimap-height", type=int, default=512)
    parser.add_argument("--progress-scale", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--timeout-penalty", type=float, default=1.0)
    parser.add_argument("--collision-penalty", type=float, default=0.5)
    parser.add_argument("--warning-penalty", type=float, default=0.05)
    return parser.parse_args()


def read_tasks(args: argparse.Namespace, rank: int, world_size: int) -> list[list[dict]]:
    records = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if record.get("split") == args.split:
                    records.append(record)
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for task in records:
        by_scene[str(task["scene_name"])].append(task)
    scene_names = sorted(
        by_scene, key=lambda name: int(name.removeprefix("scene"))
    )
    if args.scene_limit > 0:
        scene_names = scene_names[: args.scene_limit]
    scene_names = scene_names[rank::world_size]
    if not scene_names:
        raise ValueError(f"Rank {rank} received no scenes")
    env_shards: list[list[dict]] = [[] for _ in range(args.num_envs)]
    for scene_index, scene_name in enumerate(scene_names):
        tasks = by_scene[scene_name]
        if args.episodes_per_scene > 0:
            tasks = tasks[: args.episodes_per_scene]
        env_shards[scene_index % args.num_envs].extend(tasks)
    env_shards = [shard for shard in env_shards if shard]
    if len(env_shards) != args.num_envs:
        raise ValueError(
            f"Requested {args.num_envs} envs but only {len(env_shards)} received tasks"
        )
    return env_shards


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def normalized_advantages(
    advantages: torch.Tensor, world_size: int
) -> torch.Tensor:
    flat = advantages.double().reshape(-1)
    stats = torch.stack((flat.sum(), (flat * flat).sum(), flat.new_tensor(flat.numel())))
    if world_size > 1:
        # NCCL only accepts CUDA collectives. Rollout storage intentionally
        # stays on CPU, so move only these three scalars for global moments.
        stats = stats.to(torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(stats)
        stats = stats.cpu()
    mean = stats[0] / stats[2]
    variance = (stats[1] / stats[2] - mean * mean).clamp_min(1e-8)
    return (advantages - mean.float()) / variance.sqrt().float()


def serializable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: PointGoalPPOConfig,
    args: argparse.Namespace,
    update: int,
    global_steps: int,
) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": serializable_args(args),
        "update": int(update),
        "global_steps": int(global_steps),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.total_updates <= 0 or args.rollout_steps <= 0:
        raise SystemExit("num-envs, total-updates, and rollout-steps must be positive")
    if args.rollout_steps % args.bptt_len:
        raise SystemExit("rollout-steps must be divisible by bptt-len")
    if args.chunks_per_minibatch <= 0:
        raise SystemExit("chunks-per-minibatch must be positive")
    if not args.manifest.is_file() or not args.unity.is_file():
        raise SystemExit("manifest and Unity executable must exist")

    rank, world_size, local_rank = setup_distributed()
    seed = args.seed + rank * 100003
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise SystemExit("PointGoal PPO currently requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    unity_device_ids = [
        value.strip()
        for value in os.environ.get("INDUSTRYNAV_UNITY_DEVICE_IDS", "").split(",")
        if value.strip()
    ]
    if unity_device_ids:
        os.environ["INDUSTRYNAV_UNITY_DEVICE_INDEX"] = unity_device_ids[
            local_rank % len(unity_device_ids)
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / f"train_rank{rank}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("pointgoal_ppo")

    model_config = PointGoalPPOConfig(
        depth_backbone=args.depth_backbone,
        img_size=args.img_size,
        visual_dim=args.visual_dim,
        hidden_size=args.hidden_size,
        half_width=args.half_width,
        pretrained_depth=args.pretrained_depth,
        goal_distance_scale_m=args.goal_distance_scale_m,
        analytic_goal_prior_strength=args.analytic_goal_prior_strength,
    )
    model = PointGoalActorCritic(model_config).to(device)
    latest_path = args.output_dir / "latest.pt"
    start_update = 1
    global_steps = 0
    if args.resume and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        start_update = int(checkpoint.get("update", 0)) + 1
        global_steps = int(checkpoint.get("global_steps", 0))
        logger.info("Resumed model at update %d", start_update - 1)
    elif args.init_bc_checkpoint is not None:
        loaded = load_bc_depth_encoder(model, args.init_bc_checkpoint)
        logger.info("Loaded %d BC depth-encoder tensors", loaded)

    encoder_params = list(model.depth_encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_params}
    other_params = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * args.encoder_lr_scale},
            {"params": other_params, "lr": args.lr},
        ],
        eps=1e-5,
        weight_decay=args.weight_decay,
    )
    if args.resume and latest_path.is_file():
        optimizer.load_state_dict(checkpoint["optimizer"])
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # Recurrent PPO calls the wrapped model once per BPTT timestep
            # before a single backward pass. Re-broadcasting mutable ResNet
            # BatchNorm buffers between those forwards invalidates autograd's
            # saved tensor versions. Parameters/gradients still synchronize.
            broadcast_buffers=False,
        )

    task_shards = read_tasks(args, rank, world_size)
    reward_kwargs = {
        "progress_scale": args.progress_scale,
        "step_penalty": args.step_penalty,
        "success_bonus": args.success_bonus,
        "timeout_penalty": args.timeout_penalty,
        "collision_penalty": args.collision_penalty,
        "warning_penalty": args.warning_penalty,
    }
    envs: list[UnityPointGoalEnv] = []
    observations: list[dict] = []
    try:
        for env_index, tasks in enumerate(task_shards):
            env = UnityPointGoalEnv(
                tasks,
                unity_path=str(args.unity),
                output_dir=args.output_dir / "unity" / f"rank{rank}",
                worker_id=rank * args.num_envs + env_index,
                base_port=args.base_port,
                reach_m=args.reach_m,
                max_steps=args.max_episode_steps,
                ego_width=args.ego_width,
                ego_height=args.ego_height,
                minimap_width=args.minimap_width,
                minimap_height=args.minimap_height,
                dynamic_objects=args.dynamic_objects,
                goal_distance_scale_m=args.goal_distance_scale_m,
                reward_kwargs=reward_kwargs,
                logger=logger,
            )
            observations.append(env.reset())
            envs.append(env)

        num_envs = len(envs)
        hidden = (model.module if hasattr(model, "module") else model).initial_hidden(
            num_envs, device
        )
        previous_action = torch.full(
            (num_envs,), PPO_BOS_LABEL, device=device, dtype=torch.long
        )
        masks = torch.zeros(num_envs, 1, device=device)
        metrics_path = args.output_dir / "metrics.jsonl"

        for update in range(start_update, args.total_updates + 1):
            storage: dict[str, list[torch.Tensor]] = defaultdict(list)
            completed_episodes: list[dict] = []
            model.eval()
            for _ in range(args.rollout_steps):
                depth = torch.from_numpy(
                    np.stack([obs["depth"] for obs in observations])
                ).to(device)
                goal = torch.from_numpy(
                    np.stack([obs["goal"] for obs in observations])
                ).to(device)
                storage["depth"].append(depth.cpu())
                storage["goal"].append(goal.cpu())
                storage["previous_action"].append(previous_action.cpu())
                storage["hidden"].append(hidden.detach().cpu())
                storage["masks"].append(masks.cpu())
                with torch.no_grad():
                    action, log_prob, value, hidden = model.module.act(
                        depth, goal, previous_action, hidden, masks
                    ) if hasattr(model, "module") else model.act(
                        depth, goal, previous_action, hidden, masks
                    )
                next_observations = []
                rewards = []
                dones = []
                for env, selected_action in zip(envs, action.cpu().tolist()):
                    obs, reward, done, info = env.step(selected_action)
                    next_observations.append(obs)
                    rewards.append(reward)
                    dones.append(done)
                    if done:
                        completed_episodes.append(info)
                storage["actions"].append(action.cpu())
                storage["old_log_probs"].append(log_prob.cpu())
                storage["values"].append(value.cpu())
                storage["rewards"].append(torch.tensor(rewards, dtype=torch.float32))
                nonterminal = torch.tensor(
                    [not done for done in dones], dtype=torch.float32
                )
                storage["nonterminal"].append(nonterminal)
                observations = next_observations
                previous_action = action
                masks = nonterminal.to(device).unsqueeze(-1)

            with torch.no_grad():
                depth = torch.from_numpy(
                    np.stack([obs["depth"] for obs in observations])
                ).to(device)
                goal = torch.from_numpy(
                    np.stack([obs["goal"] for obs in observations])
                ).to(device)
                _, next_value, _ = model(
                    depth, goal, previous_action, hidden, masks
                )
            tensors = {key: torch.stack(value) for key, value in storage.items()}
            advantages, returns = generalized_advantage_estimate(
                tensors["rewards"],
                tensors["values"],
                next_value.cpu(),
                tensors["nonterminal"],
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            advantages = normalized_advantages(advantages, world_size)

            chunks = [
                (env_index, start)
                for env_index in range(num_envs)
                for start in range(0, args.rollout_steps, args.bptt_len)
            ]
            epoch_losses = []
            model.train()
            for epoch in range(args.ppo_epochs):
                random.Random(seed + update * 101 + epoch).shuffle(chunks)
                for offset in range(0, len(chunks), args.chunks_per_minibatch):
                    batch_chunks = chunks[offset : offset + args.chunks_per_minibatch]
                    batch_hidden = torch.stack(
                        [tensors["hidden"][start, env] for env, start in batch_chunks]
                    ).to(device)
                    new_log_probs = []
                    new_values = []
                    entropies = []
                    old_log_probs = []
                    old_values = []
                    batch_returns = []
                    batch_advantages = []
                    for relative_step in range(args.bptt_len):
                        indices = [
                            (start + relative_step, env) for env, start in batch_chunks
                        ]
                        step_depth = torch.stack(
                            [tensors["depth"][step, env] for step, env in indices]
                        ).to(device)
                        step_goal = torch.stack(
                            [tensors["goal"][step, env] for step, env in indices]
                        ).to(device)
                        step_previous = torch.stack(
                            [tensors["previous_action"][step, env] for step, env in indices]
                        ).to(device)
                        step_masks = torch.stack(
                            [tensors["masks"][step, env] for step, env in indices]
                        ).to(device)
                        logits, value, batch_hidden = model(
                            step_depth,
                            step_goal,
                            step_previous,
                            batch_hidden,
                            step_masks,
                        )
                        selected = torch.stack(
                            [tensors["actions"][step, env] for step, env in indices]
                        ).to(device)
                        distribution = Categorical(logits=logits)
                        new_log_probs.append(distribution.log_prob(selected))
                        new_values.append(value)
                        entropies.append(distribution.entropy())
                        old_log_probs.append(torch.stack([
                            tensors["old_log_probs"][step, env]
                            for step, env in indices
                        ]).to(device))
                        old_values.append(torch.stack([
                            tensors["values"][step, env] for step, env in indices
                        ]).to(device))
                        batch_returns.append(torch.stack([
                            returns[step, env] for step, env in indices
                        ]).to(device))
                        batch_advantages.append(torch.stack([
                            advantages[step, env] for step, env in indices
                        ]).to(device))

                    new_log_probs_t = torch.stack(new_log_probs)
                    new_values_t = torch.stack(new_values)
                    entropy = torch.stack(entropies).mean()
                    old_log_probs_t = torch.stack(old_log_probs)
                    old_values_t = torch.stack(old_values)
                    returns_t = torch.stack(batch_returns)
                    advantages_t = torch.stack(batch_advantages)
                    ratio = (new_log_probs_t - old_log_probs_t).exp()
                    policy_loss = -torch.minimum(
                        ratio * advantages_t,
                        ratio.clamp(1.0 - args.clip_param, 1.0 + args.clip_param)
                        * advantages_t,
                    ).mean()
                    clipped_value = old_values_t + (
                        new_values_t - old_values_t
                    ).clamp(-args.clip_param, args.clip_param)
                    value_loss = 0.5 * torch.maximum(
                        (new_values_t - returns_t).pow(2),
                        (clipped_value - returns_t).pow(2),
                    ).mean()
                    loss = (
                        policy_loss
                        + args.value_loss_coef * value_loss
                        - args.entropy_coef * entropy
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    approx_kl = (old_log_probs_t - new_log_probs_t).mean().detach()
                    epoch_losses.append(
                        torch.tensor(
                            [loss.item(), policy_loss.item(), value_loss.item(), entropy.item(), approx_kl.item()],
                            device=device,
                        )
                    )

            global_steps += args.rollout_steps * num_envs * world_size
            loss_stats = torch.stack(epoch_losses).mean(0)
            rollout_stats = torch.tensor(
                [
                    tensors["rewards"].mean().item(),
                    len(completed_episodes),
                    sum(int(ep["success"]) for ep in completed_episodes),
                    sum(float(ep["episode_return"]) for ep in completed_episodes),
                    sum(int(ep["episode_collisions"]) for ep in completed_episodes),
                    sum(int(ep["episode_warnings"]) for ep in completed_episodes),
                ],
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(loss_stats)
                loss_stats /= world_size
                dist.all_reduce(rollout_stats)
            episode_count = int(rollout_stats[1].item())
            record = {
                "update": update,
                "global_steps": global_steps,
                "mean_step_reward": float(rollout_stats[0].item() / world_size),
                "episodes": episode_count,
                "success_rate": (
                    float(rollout_stats[2].item() / episode_count)
                    if episode_count else None
                ),
                "mean_episode_return": (
                    float(rollout_stats[3].item() / episode_count)
                    if episode_count else None
                ),
                "collisions_per_episode": (
                    float(rollout_stats[4].item() / episode_count)
                    if episode_count else None
                ),
                "warnings_per_episode": (
                    float(rollout_stats[5].item() / episode_count)
                    if episode_count else None
                ),
                "loss": float(loss_stats[0].item()),
                "policy_loss": float(loss_stats[1].item()),
                "value_loss": float(loss_stats[2].item()),
                "entropy": float(loss_stats[3].item()),
                "approx_kl": float(loss_stats[4].item()),
            }
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                logger.info("PPO %s", json.dumps(record, sort_keys=True))
                save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    model_config,
                    args,
                    update,
                    global_steps,
                )
                if update % args.checkpoint_interval == 0:
                    save_checkpoint(
                        args.output_dir / f"update_{update:06d}.pt",
                        model,
                        optimizer,
                        model_config,
                        args,
                        update,
                        global_steps,
                    )
            if world_size > 1:
                dist.barrier(device_ids=[local_rank])
    finally:
        for env in envs:
            env.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
