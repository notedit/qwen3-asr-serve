"""Download Qwen3-ASR + ForcedAligner weights from HuggingFace.

Usage:
    python scripts/download_models.py --mode {asr|aligner|both}

Tries the default HF endpoint first, falls back to https://hf-mirror.com when
HF is unreachable from this network.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError


ASR_REPO = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_MODEL_DIR = Path("./models")
MIRROR = "https://hf-mirror.com"


def _download_with_fallback(repo_id: str, local_dir: Path) -> None:
    """Try HF first; on connection failure switch endpoint to hf-mirror."""
    local_dir.mkdir(parents=True, exist_ok=True)
    # Resume capability is built into snapshot_download — it skips already-downloaded files.
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return
    except (HfHubHTTPError, OSError, ConnectionError) as e:
        print(f"[download] HF failed for {repo_id}: {e!r}", file=sys.stderr)

    print(f"[download] retrying via mirror {MIRROR}", file=sys.stderr)
    os.environ["HF_ENDPOINT"] = MIRROR
    # huggingface_hub reads the env var on next call, but to be safe we also pass
    # endpoint via kwargs if supported.
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            endpoint=MIRROR,
        )
    except TypeError:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["asr", "aligner", "both"], default="both")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--asr-repo", default=ASR_REPO)
    ap.add_argument("--aligner-repo", default=ALIGNER_REPO)
    args = ap.parse_args()

    if args.mode in ("asr", "both"):
        target = args.model_dir / args.asr_repo.split("/")[-1]
        print(f"[download] {args.asr_repo} → {target}")
        _download_with_fallback(args.asr_repo, target)

    if args.mode in ("aligner", "both"):
        target = args.model_dir / args.aligner_repo.split("/")[-1]
        print(f"[download] {args.aligner_repo} → {target}")
        _download_with_fallback(args.aligner_repo, target)

    print("[download] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
