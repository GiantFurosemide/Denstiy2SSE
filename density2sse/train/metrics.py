"""Simple metric logging."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List


def append_csv_row(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)
