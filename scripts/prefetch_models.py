#!/usr/bin/env python
"""Download checkpoints into the Hugging Face cache ($HF_HOME).

Used either at image build time (``--build-arg PREFETCH_MODELS=...``) or
afterwards against the mounted volume:

    docker compose run --rm api python scripts/prefetch_models.py matxa_v2_multiaccent
"""
from __future__ import annotations

import sys

from app.engine import download_model
from app.registry import BUILTIN_MODELS, default_model


def main(argv: list[str]) -> int:
    names = argv or [default_model()]
    if names == ["all"]:
        names = list(BUILTIN_MODELS)

    unknown = [n for n in names if n not in BUILTIN_MODELS]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(BUILTIN_MODELS)} (or 'all')", file=sys.stderr)
        return 2

    for name in names:
        print(f"[*] fetching {name} ...", flush=True)
        download_model(BUILTIN_MODELS[name])
    print("[+] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
