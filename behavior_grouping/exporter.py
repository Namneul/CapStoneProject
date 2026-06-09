"""
JSON export helpers for behavior analysis results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def export_analysis_result(
    result: dict[str, Any],
    output_dir: str | Path = "result",
    filename: str = "analysis_result.json",
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        },
        **result,
    }

    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)
