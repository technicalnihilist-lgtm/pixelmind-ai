#!/usr/bin/env python3
"""
Export Stable Diffusion 1.5 + LCM-LoRA into the ONNX graphs PixelMind AI loads on the phone.

Run this once, on a desktop with plenty of RAM (16 GB is comfortable, 8 GB may swap):

    pip install "torch>=2.2" "diffusers>=0.27" "transformers" "onnx" "onnxruntime" "safetensors"
    python tools/export_model.py --out modelpack/src/main/assets/models

It writes:

    text_encoder.onnx      CLIP text tower, int8 dynamic
    unet.onnx              LCM-distilled UNet, int8 dynamic
    vae_decoder.onnx       VAE decoder, fp32
    clip/vocab.json        CLIP tokenizer vocabulary
    clip/merges.txt        CLIP BPE merges
    model_manifest.json    file names, sizes, SHA-256, tensor names, tokenizer config

Nothing here runs on the phone, and the phone never downloads anything: the output of this
script is committed into the asset pack (or copied there) and installed with the app.

Licensing: the resulting weights inherit CreativeML OpenRAIL-M from SD 1.5. Read docs/MODEL.md
before redistributing them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"
SAMPLE_SIZE = 512
LATENT_CHANNELS = 4


def log(message: str) -> None:
    print(f"[export] {message}", flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human(size: int) -> str:
    return f"{size / 1048576:.1f} MB"


def load_pipeline():
    try:
        import torch
        from diffusers import AutoPipelineForText2Image, LCMScheduler
    except ImportError as exc:  # pragma: no cover - environment guard
        sys.exit(
            "Missing dependencies. Install with:\n"
            '  pip install "torch>=2.2" "diffusers>=0.27" "transformers" onnx onnxruntime\n'
            f"Original error: {exc}"
        )

    log(f"loading {BASE_MODEL}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, safety_checker=None
    )
    log(f"fusing {LCM_LORA}")
    pipe.load_lora_weights(LCM_LORA)
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    # Four steps is what the distillation was trained for.
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def export_text_encoder(pipe, out_dir: Path, opset: int) -> Path:
    import torch

    encoder = pipe.text_encoder.eval()
    target = out_dir / "text_encoder.onnx"
    dummy = torch.ones(1, 77, dtype=torch.long)
    log("exporting text encoder")
    torch.onnx.export(
        encoder,
        (dummy,),
        str(target),
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "batch"}, "last_hidden_state": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    return target


def export_unet(pipe, out_dir: Path, opset: int) -> Path:
    import torch

    unet = pipe.unet.eval()
    target = out_dir / "unet.onnx"
    sample = torch.randn(1, LATENT_CHANNELS, SAMPLE_SIZE // 8, SAMPLE_SIZE // 8)
    timestep = torch.tensor([1], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 77, 768)
    log("exporting UNet")
    torch.onnx.export(
        unet,
        (sample, timestep, encoder_hidden_states),
        str(target),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"],
        dynamic_axes={
            "sample": {0: "batch", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch"},
            "out_sample": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return target


def export_vae(pipe, out_dir: Path, opset: int) -> Path:
    import torch

    vae = pipe.vae.eval()
    target = out_dir / "vae_decoder.onnx"
    latent = torch.randn(1, LATENT_CHANNELS, SAMPLE_SIZE // 8, SAMPLE_SIZE // 8)
    log("exporting VAE decoder")
    torch.onnx.export(
        vae.decode,
        (latent,),
        str(target),
        input_names=["latent_sample"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 2: "height", 3: "width"},
            "sample": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return target


def quantize_int8(source: Path, destination: Path) -> None:
    """Dynamic int8 quantisation: ~4x smaller weights, no calibration data needed."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    log(f"quantising {source.name} -> {destination.name}")
    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        weight_type=QuantType.QInt8,
        # Keep the graph's matmuls in int8 but leave ops that hurt quality alone.
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        per_channel=True,
        reduce_range=False,
    )


def copy_tokenizer(pipe, out_dir: Path) -> tuple[Path, Path]:
    from transformers import CLIPTokenizer

    clip_dir = out_dir / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    log("saving CLIP tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    tokenizer.save_pretrained(str(clip_dir))

    vocab = clip_dir / "vocab.json"
    merges = clip_dir / "merges.txt"
    if not vocab.exists() or not merges.exists():
        sys.exit("Tokenizer export did not produce vocab.json / merges.txt")
    return vocab, merges


def verify(out_dir: Path) -> None:
    """Loads every graph with ONNX Runtime and prints its I/O. Fails loudly if broken."""
    try:
        import onnxruntime as ort
    except ImportError:
        log("onnxruntime not installed; skipping graph verification")
        return

    for name in ("text_encoder.onnx", "unet.onnx", "vae_decoder.onnx"):
        path = out_dir / name
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = [(i.name, i.shape) for i in session.get_inputs()]
        outputs = [(o.name, o.shape) for o in session.get_outputs()]
        log(f"verified {name}: in={inputs} out={outputs}")


def build_manifest(out_dir: Path, vae_file: Path) -> Path:
    components = []
    for name in ("text_encoder.onnx", "unet.onnx", vae_file.name, "clip/vocab.json", "clip/merges.txt"):
        path = out_dir / name
        components.append(
            {
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )

    manifest = {
        "name": "Stable Diffusion 1.5 · LCM",
        "version": "1.0.0",
        "architecture": "sd15-lcm",
        "precision": "int8",
        "license": "CreativeML OpenRAIL-M",
        "latent_channels": LATENT_CHANNELS,
        "sample_size": SAMPLE_SIZE // 8,
        "components": components,
        "text_encoder": {
            "file": "text_encoder.onnx",
            "inputs": ["input_ids"],
            "output": "last_hidden_state",
        },
        "unet": {
            "file": "unet.onnx",
            "inputs": ["sample", "timestep", "encoder_hidden_states"],
            "output": "out_sample",
        },
        "vae_decoder": {
            "file": vae_file.name,
            "inputs": ["latent_sample"],
            "output": "sample",
        },
        "tokenizer": {
            "vocab": "clip/vocab.json",
            "merges": "clip/merges.txt",
            "max_length": 77,
            "bos": "<|startoftext|>",
            "eos": "<|endoftext|>",
            "pad": "<|endoftext|>",
        },
    }

    target = out_dir / "model_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(c["bytes"] for c in components)
    log(f"manifest written: {target}  (total weights {human(total)})")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SD1.5+LCM to ONNX for PixelMind AI")
    parser.add_argument("--out", required=True, type=Path, help="output directory (asset pack)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--keep-fp32", action="store_true", help="skip int8 quantisation")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline()

    raw_encoder = export_text_encoder(pipe, out_dir, args.opset)
    raw_unet = export_unet(pipe, out_dir, args.opset)
    vae_file = export_vae(pipe, out_dir, args.opset)

    if not args.keep_fp32:
        for raw in (raw_encoder, raw_unet):
            quantized = raw.with_name(raw.stem + ".int8.onnx")
            quantize_int8(raw, quantized)
            raw.unlink()  # the quantised graph replaces the fp32 export
            quantized.rename(raw)

    copy_tokenizer(pipe, out_dir)
    verify(out_dir)
    build_manifest(out_dir, vae_file)

    # Keep the tokenizer files, drop intermediate artefacts.
    for stray in out_dir.glob("*.onnx_data"):
        shutil.move(str(stray), str(out_dir / stray.name))

    log("done. Rebuild the app to bundle the new weights.")


if __name__ == "__main__":
    main()
