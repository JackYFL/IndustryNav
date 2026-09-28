"""Datasets and data transformations shared by training workflows."""

__all__ = [
    "NavEpisodeDataset",
    "NavEpisodeSequenceDataset",
    "depth_observation_uint8",
]


def __getattr__(name: str):
    if name in {"NavEpisodeDataset", "NavEpisodeSequenceDataset"}:
        from nav.data.pointgoal import NavEpisodeDataset, NavEpisodeSequenceDataset

        return {
            "NavEpisodeDataset": NavEpisodeDataset,
            "NavEpisodeSequenceDataset": NavEpisodeSequenceDataset,
        }[name]
    if name == "depth_observation_uint8":
        from nav.data.preprocessing import depth_observation_uint8

        return depth_observation_uint8
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
