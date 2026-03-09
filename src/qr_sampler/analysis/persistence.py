"""Result persistence for entropick experimental sessions.

Saves and loads TokenSamplingRecord sequences as JSONL files with
optional metadata on the first line.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qr_sampler.logging.types import TokenSamplingRecord

logger = logging.getLogger("qr_sampler")


def save_records(
    records: list[TokenSamplingRecord],
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save sampling records to a JSONL file.

    If *metadata* is provided it is written as the first line with a
    ``_meta`` sentinel key.  Each subsequent line is one JSON-serialised
    ``TokenSamplingRecord``.

    Args:
        records: List of frozen TokenSamplingRecord instances.
        path: Destination file path (created or overwritten).
        metadata: Optional session metadata dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        if metadata is not None:
            meta_line: dict[str, Any] = {
                "_meta": True,
                "timestamp": time.time(),
                "session_id": str(uuid.uuid4()),
            }
            meta_line.update(metadata)
            fh.write(json.dumps(meta_line, default=str) + "\n")

        for record in records:
            fh.write(json.dumps(asdict(record), default=str) + "\n")

    logger.info(
        "Saved %d records to %s",
        len(records),
        path,
    )


def load_records(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load sampling records from a JSONL file.

    Returns a ``(metadata, records)`` tuple.  If the file has no
    metadata line the metadata dict is empty.

    Args:
        path: Source JSONL file path.

    Returns:
        Tuple of (metadata dict, list of record dicts).
    """
    path = Path(path)
    metadata: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if line_no == 1 and obj.get("_meta"):
                metadata = obj
            else:
                records.append(obj)

    logger.info(
        "Loaded %d records from %s",
        len(records),
        path,
    )
    return metadata, records
