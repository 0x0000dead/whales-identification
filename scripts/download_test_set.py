#!/usr/bin/env python3
"""Fetch the extended EcoMarineAI test set from the project DVC remote.

Usage:
    python scripts/download_test_set.py --target data/test_split

This is intentionally a thin wrapper around DVC: the actual storage backend
(S3, Yandex Object Storage, etc.) is configured in `.dvc/config`. If DVC is
not yet pointed at a remote, this script falls back to a no-op + warning so
local dev still works.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess  # nosec B404 - we control the args
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = REPO_ROOT / "data" / "test_split"


def _run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)  # nosec B603


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Directory to populate (default: data/test_split/)",
    )
    parser.add_argument(
        "--remote",
        default=None,
        help="DVC remote name (default: configured in .dvc/config)",
    )
    args = parser.parse_args()

    if not shutil.which("dvc"):
        print(
            "WARN: dvc not found in PATH. Install with `pip install dvc[s3]` or similar.",
            file=sys.stderr,
        )
        print(
            "Falling back to manifest-only mode — drop your own files into "
            f"{args.target}/positives and {args.target}/negatives.",
            file=sys.stderr,
        )
        return 0

    args.target.mkdir(parents=True, exist_ok=True)
    cmd = ["dvc", "pull", str(args.target)]
    if args.remote:
        cmd.extend(["--remote", args.remote])

    rc = _run(cmd)
    if rc != 0:
        print(
            "WARN: `dvc pull` failed. The repo may not have a remote configured "
            "yet. You can populate test_split/ manually or skip the integration "
            "metrics tests until a DVC remote is set up.",
            file=sys.stderr,
        )
        return 0  # non-fatal — the manifest still works for whatever IS local

    print(f"OK: extended test set fetched into {args.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
