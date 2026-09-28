"""Opt-in subprocess teardown timeout, separate from Unity connection timeout."""

import logging
import math
import os
import subprocess
from collections.abc import Mapping


CLOSE_TIMEOUT_ENV = "INDUSTRYNAV_UNITY_CLOSE_TIMEOUT_SECONDS"


def configured_close_timeout(environ: Mapping[str, str] | None = None) -> float | None:
    environ = os.environ if environ is None else environ
    raw = environ.get(CLOSE_TIMEOUT_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{CLOSE_TIMEOUT_ENV} must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{CLOSE_TIMEOUT_ENV} must be finite and positive")
    return value


def close_unity_environment(env, *, timeout_seconds: float | None = None, logger=None) -> None:
    """Close only this environment's owned player; preserve default behavior.

    The installed ML-Agents API's public close() reuses its connection timeout.
    Its _close(timeout=...) already sends the shutdown request and kills only
    its own subprocess when necessary. We retain the process handle to reap it
    afterward, since that API drops it immediately after kill(). Never change
    _timeout_wait: it also controls active RPC/initialization waits.
    """
    if timeout_seconds is None:
        env.close()
        return
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Unity close timeout must be finite and positive")
    log = logger or logging.getLogger(__name__)
    closer = getattr(env, "_close", None)
    if getattr(env, "_loaded", None) is not True or not callable(closer):
        # Preserve public errors for unloaded players and compatibility for
        # adapters that do not implement ML-Agents' private timeout API.
        env.close()
        return
    process = getattr(env, "_process", None)
    log.info("Closing owned Unity player pid=%s with %.3fs exit timeout",
             getattr(process, "pid", None), timeout_seconds)
    closer(timeout=timeout_seconds)
    if process is not None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            log.warning("Owned Unity player pid=%s was not reaped within 1s after close",
                        getattr(process, "pid", None))


__all__ = ["CLOSE_TIMEOUT_ENV", "configured_close_timeout", "close_unity_environment"]
