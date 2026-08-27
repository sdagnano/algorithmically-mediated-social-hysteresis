"""Create the SHA-256 inventory for a frozen repository release."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", "__pycache__", "tmp", "submission"}
EXCLUDED_NAMES = {"release_manifest.json", "archive_verification.json"}
EXCLUDED_SUFFIXES = {".aux", ".log", ".out", ".synctex.gz", ".xdv"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    if any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return False
    return path.is_file()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files = []
    for path in sorted((path for path in ROOT.rglob("*") if included(path)), key=lambda item: item.as_posix().lower()):
        files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    manifest = {
        "schema": "algorithmic-hysteresis-release-manifest/v1",
        "release": "v1.0.0",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "hash_algorithm": "SHA-256",
        "files": files,
    }
    (ROOT / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"release": manifest["release"], "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
