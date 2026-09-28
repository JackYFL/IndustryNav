"""Behavior-cloning baseline: agent and algorithm-specific trainer."""

__all__ = ["BCTrainer", "BCNavAgent", "BCNavController", "train"]


def __getattr__(name: str):
    if name in {"BCNavAgent", "BCNavController"}:
        from nav.baselines.bc.agent import BCNavAgent, BCNavController

        aliases = {
            "BCNavAgent": BCNavAgent,
            "BCNavController": BCNavController,
        }
        return aliases[name]
    if name in {"BCTrainer", "train"}:
        from nav.baselines.bc.trainer import BCTrainer, train

        return {"BCTrainer": BCTrainer, "train": train}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
