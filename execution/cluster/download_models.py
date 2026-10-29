#!/usr/bin/env python3
"""Download model weights to $MODELS_DIR/hf (HF_HOME) for the B200 audit.

Run once on the login node (needs internet). Everything lands under
$HF_HOME/hub so later jobs can run with HF_HUB_OFFLINE=1.

  python download_models.py                # all 12 repos (~130 GB)
  python download_models.py --only tev1-4b # one slug
  python download_models.py --list
"""
import argparse
import os
import sys
from pathlib import Path
def _env_path(var, default):
    """Env var treated as unset when empty (exported-empty shells are common)."""
    import os
    v = os.environ.get(var, "").strip()
    return Path(v) if v else Path(default)



REPOS = {
    # slug: (repo_id, allow_patterns or None)
    "tev1-4b":      ("togethercomputer/Tev1-4B-experimental", None),
    "tev1-0.8b":    ("togethercomputer/Tev1-0.8B-experimental", None),
    "nimble-9b":    ("bespokelabs/Bespoke-Nimble-9B", None),          # LoRA adapter + configs
    "qwen3.5-9b-base": ("Qwen/Qwen3.5-9B", None),                     # Nimble base
    "qwen3.5-4b":   ("Qwen/Qwen3.5-4B", None),
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", None),                      # Tev1-0.8B base
    "qwen3.5-9b":   ("Qwen/Qwen3.5-9B", None),                        # same repo as base; shared cache
    "gpt-oss-20b":  ("openai/gpt-oss-20b", ["*.json", "*.safetensors", "tokenizer*", "original/*", "chat_template*"]),
    "clef":         ("Cloudflare/clef", None),
    "clef-flash":   ("Cloudflare/clef-flash", None),
    "laya":         ("convaiinnovations/laya", None),
    "von":          ("wfzyx/von", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated slugs")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    hf_home = _env_path("HF_HOME", "")
    if str(hf_home) in ("", "."):
        hf_home = _env_path("MODELS_DIR", "models") / "hf"
    hf_home.mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME={hf_home}")

    slugs = [s.strip() for s in args.only.split(",") if s.strip()] or list(REPOS)
    if args.list:
        for s in slugs:
            print(f"{s:<16} {REPOS[s][0]}")
        return

    from huggingface_hub import snapshot_download
    failed = []
    for slug in slugs:
        repo, patterns = REPOS[slug]
        kwargs = dict(repo_id=repo, max_workers=8)
        if patterns:
            kwargs["allow_patterns"] = patterns
        print(f"[{slug}] -> {repo}")
        try:
            path = snapshot_download(**kwargs)
            print(f"  OK {path}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {e}", file=sys.stderr)
            failed.append(slug)
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
