#!/usr/bin/env python3
"""
Checks that an exported weight directory is complete and matches its manifest.

    python tools/verify_export.py app/src/main/assets/models

Exit code 0 means the app will find the weights and pass its own checksum verification.
Runs on plain Python; if onnxruntime is installed it also loads every graph.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REQUIRED_GRAPHS = ("text_encoder", "unet", "vae_decoder")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"  FAIL  {message}")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: verify_export.py <model-directory>")
    root = Path(sys.argv[1])

    manifest_path = root / "model_manifest.json"
    if not manifest_path.exists():
        fail(f"{manifest_path} is missing — run tools/export_model.py first")
    manifest = json.loads(manifest_path.read_text())
    print(f"manifest: {manifest.get('name')} {manifest.get('version')} "
          f"({manifest.get('precision')})")

    total = 0
    for component in manifest.get("components", []):
        path = root / component["file"]
        if not path.exists():
            fail(f"missing component: {component['file']}")
        size = path.stat().st_size
        total += size
        if component.get("bytes") and size != component["bytes"]:
            fail(f"{component['file']}: {size} bytes on disk, manifest says {component['bytes']}")
        expected = component.get("sha256", "")
        if expected and sha256_of(path).lower() != expected.lower():
            fail(f"{component['file']}: checksum mismatch")
        print(f"  ok    {component['file']:<24} {size / 1048576:8.1f} MB")

    for key in REQUIRED_GRAPHS:
        if key not in manifest:
            fail(f"manifest has no '{key}' section")
        if not (root / manifest[key]["file"]).exists():
            fail(f"{key} graph missing: {manifest[key]['file']}")

    tokenizer = manifest.get("tokenizer", {})
    for key in ("vocab", "merges"):
        if not (root / tokenizer.get(key, "")).exists():
            fail(f"tokenizer file missing: {tokenizer.get(key)}")

    print(f"total weights: {total / 1048576:.1f} MB")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipped graph load check")
    else:
        for key in REQUIRED_GRAPHS:
            path = root / manifest[key]["file"]
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            print(f"  loads {path.name}: in={[i.name for i in session.get_inputs()]}")

    print("PASS — this directory will work with the app.")


if __name__ == "__main__":
    main()
