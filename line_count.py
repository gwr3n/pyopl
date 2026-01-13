# Counts non-empty, non-comment lines in .py files (simple heuristic).
from __future__ import annotations

from pathlib import Path

EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "tmp",
    "pyopl_rhetor.egg-info",
    "rhetor.egg-info",
}
total = 0
files = 0

for p in Path(".").rglob("*.py"):
    if any(part in EXCLUDE_DIRS for part in p.parts):
        continue
    files += 1
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        total += 1

print({"python_files": files, "loc_no_blank_no_hash_comment": total})
