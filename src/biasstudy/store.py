"""Append-only JSONL store keyed by a stable record key, so runs can resume after a crash."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A crash mid-write leaves a truncated last line; that job just reruns.
                    continue

    def keys(self) -> set[str]:
        return {r["key"] for r in self}

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.path.open("a") as f:
            # Start on a fresh line in case the previous write was cut off.
            if f.tell() > 0 and not self._ends_with_newline():
                f.write("\n")
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _ends_with_newline(self) -> bool:
        with self.path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) == b"\n"
