"""The W&B key a report producer should use for an entity (studies#157).

Producers read W&B over GraphQL. For a ``NeuralDynamics`` project that must be the ``han-hou``
key, not the personal one in ``~/.netrc``, so a session no longer exports it by hand. The
launchers pick keys the same way (dispatcher#225): ``~/.config/wandb/api_key.<entity>``.

Order: that file, then ``$WANDB_API_KEY``, then ``~/.netrc`` (``api.wandb.ai``). The file wins
over the env var because a stale personal key can sit in the env. A study on ``AIND-disRNN`` has
no key file, so it behaves as before.
"""

from __future__ import annotations

import os
from pathlib import Path

KEY_DIR_ENV = "BFM_WANDB_KEY_DIR"


def key_file(entity: str) -> Path:
    base = Path(os.environ.get(KEY_DIR_ENV) or Path.home() / ".config" / "wandb")
    return base / f"api_key.{entity}"


def _netrc_key() -> str | None:
    try:
        import netrc
        auth = netrc.netrc(str(Path.home() / ".netrc")).authenticators("api.wandb.ai")
    except (OSError, ImportError, netrc.NetrcParseError):
        return None
    return auth[2] if auth else None


def key_for(entity: str) -> str:
    f = key_file(entity)
    if f.is_file() and f.read_text().strip():
        return f.read_text().strip()
    key = os.environ.get("WANDB_API_KEY") or _netrc_key()
    if not key:
        raise RuntimeError(f"no W&B key for {entity!r}: put it in {f}, or export WANDB_API_KEY")
    return key
