"""ml-agents side channels for the unified scene_all client.

Two channels carry data the standard observation/action protocol can't:

- :class:`BoundsSideChannel` — the client reports the minimap's world-space
  bounds (minX/maxX/minZ/maxZ) once after reset.
- :class:`TargetSideChannel` — bidirectional: Python queues spawn/target
  pixel coordinates; the client acks back the pixel→world mapping it
  resolved, which Python reads to learn the agent's world spawn + the
  target's world position.

UUIDs + opcodes live in :mod:`nav.config` (``SIDE_CHANNEL_*``) and must
match the C# registration in the client.
"""

from __future__ import annotations

import uuid
from typing import Optional, Tuple

from mlagents_envs.side_channel.side_channel import (
    IncomingMessage,
    OutgoingMessage,
    SideChannel,
)

from nav.config import (
    SIDE_CHANNEL_BOUNDS_UUID,
    SIDE_CHANNEL_MSG_SPAWN_MAPPING,
    SIDE_CHANNEL_MSG_TARGET_MAPPING,
    SIDE_CHANNEL_TARGET_UUID,
)


class BoundsSideChannel(SideChannel):
    """Receives the minimap's world-space bounds from the client."""

    def __init__(self) -> None:
        super().__init__(uuid.UUID(SIDE_CHANNEL_BOUNDS_UUID))
        self.bounds: Optional[dict] = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.bounds = {
            "minX": msg.read_float32(),
            "maxX": msg.read_float32(),
            "minZ": msg.read_float32(),
            "maxZ": msg.read_float32(),
        }


class TargetSideChannel(SideChannel):
    """Sends spawn/target pixels and receives the client's pixel→world acks.

    After Python primes ``spawn_*`` / ``target_*`` env params and resets, the
    client replies with the resolved world coordinates. ``last_spawn_world``
    and ``last_target_world`` hold those once received; ``last_ack`` always
    holds the most recent pixel ack (including the backward-compatible bare
    ``(x, y)`` form from older builds).
    """

    def __init__(self) -> None:
        super().__init__(uuid.UUID(SIDE_CHANNEL_TARGET_UUID))
        self.last_ack: Optional[Tuple[int, int]] = None
        self.last_spawn_pixel: Optional[Tuple[int, int]] = None
        self.last_spawn_world: Optional[Tuple[float, float]] = None
        self.last_target_pixel: Optional[Tuple[int, int]] = None
        self.last_target_world: Optional[Tuple[float, float]] = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        first = msg.read_int32()
        if first == SIDE_CHANNEL_MSG_SPAWN_MAPPING:
            px, py = msg.read_int32(), msg.read_int32()
            wx, wz = msg.read_float32(), msg.read_float32()
            self.last_spawn_pixel = (px, py)
            self.last_spawn_world = (float(wx), float(wz))
            self.last_ack = (px, py)
            return
        if first == SIDE_CHANNEL_MSG_TARGET_MAPPING:
            px, py = msg.read_int32(), msg.read_int32()
            wx, wz = msg.read_float32(), msg.read_float32()
            self.last_target_pixel = (px, py)
            self.last_target_world = (float(wx), float(wz))
            self.last_ack = (px, py)
            return
        # Backward-compatible bare (x, y) ack from older builds.
        x = first
        y = msg.read_int32()
        self.last_ack = (x, y)

    def send_target(self, x: int, y: int) -> None:
        msg = OutgoingMessage()
        msg.write_int32(int(x))
        msg.write_int32(int(y))
        super().queue_message_to_send(msg)
