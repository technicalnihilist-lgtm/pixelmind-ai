#!/usr/bin/env python3
"""
Builds the **verification model**: a small, genuinely-runnable diffusion pipeline that matches
PixelMind's tensor contract exactly, so the whole on-device path (tokenizer -> conditioning ->
LCM loop -> VAE decode -> PNG) can be exercised without the 600 MB Stable Diffusion export.

    python tools/make_tiny_model.py --out app/src/main/assets/models

What this is:
  * three real ONNX graphs (text encoder, UNet, VAE decoder) executed by ONNX Runtime on the phone
  * the real CLIP tokenizer (vocab.json + merges.txt, 49k tokens), so tokenisation is not faked
  * ~6 MB of weights, so it fits in an APK you can install in seconds

What this is NOT:
  * not Stable Diffusion. The graphs are untrained random-weight networks with the same shapes as
    the real pipeline. Output is abstract, prompt-dependent gradient art, not photoreal imagery.
  * the point is to prove the plumbing runs on hardware, not to produce portfolio images.

Ship quality still requires tools/export_model.py (SD 1.5 + LCM). This model exists so that
"the app generates nothing" is never a mystery: if this produces an image, the app works and the
only missing piece is better weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
# Tuned so the LCM trajectory neither collapses to flat grey nor diverges to saturation.
OUTPUT_GAIN = 0.35
TEXT_GAIN = 6.0
OUT_TEXT_GAIN = 3.0
EOS_ID = 49407
VOCAB_SIZE = 49408
SEQ_LEN = 77
EMBED_DIM = 32
HIDDEN = 768
LATENT_CHANNELS = 4
TOKENIZER_BASE = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main"

rng = np.random.default_rng(20261028)


# ---------------------------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------------------------

class Graph:
    """Tiny builder: collects nodes and initialisers, then emits a ModelProto."""

    def __init__(self, name: str):
        self.name = name
        self.nodes: list = []
        self.inits: list = []
        self.inputs: list = []
        self.outputs: list = []
        self._counter = 0

    def uid(self, hint: str) -> str:
        self._counter += 1
        return f"{hint}_{self._counter}"

    def add_input(self, name: str, elem_type, shape: list) -> None:
        self.inputs.append(helper.make_tensor_value_info(name, elem_type, shape))

    def add_output(self, name: str, elem_type, shape: list) -> None:
        self.outputs.append(helper.make_tensor_value_info(name, elem_type, shape))

    def const(self, array: np.ndarray, hint: str = "c") -> str:
        name = self.uid(hint)
        self.inits.append(numpy_helper.from_array(np.asarray(array, dtype=np.float32), name))
        return name

    def const_i64(self, array: np.ndarray, hint: str = "i") -> str:
        name = self.uid(hint)
        self.inits.append(numpy_helper.from_array(np.asarray(array, dtype=np.int64), name))
        return name

    def node(self, op: str, inputs: list, hint: str, out_name: str | None = None, **attrs) -> str:
        out = out_name if out_name else self.uid(hint)
        self.nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    # --- compound ops -------------------------------------------------------------------
    def conv(self, x: str, cin: int, cout: int, hint: str, kernel: int = 3, pad: int = 1) -> str:
        w = rng.normal(0, np.sqrt(2.0 / (cin * kernel * kernel)), (cout, cin, kernel, kernel))
        b = np.zeros(cout, dtype=np.float32)
        wn = self.uid(f"{hint}_w")
        bn = self.uid(f"{hint}_b")
        self.inits.append(numpy_helper.from_array(w.astype(np.float32), wn))
        self.inits.append(numpy_helper.from_array(b, bn))
        return self.node(
            "Conv", [x, wn, bn], hint,
            kernel_shape=[kernel, kernel], pads=[pad, pad, pad, pad], strides=[1, 1],
        )

    def silu(self, x: str, hint: str = "silu") -> str:
        s = self.node("Sigmoid", [x], f"{hint}_sig")
        return self.node("Mul", [x, s], hint)

    def upscale(self, x: str, factor: int, hint: str) -> str:
        scales = self.const([1.0, 1.0, float(factor), float(factor)], "scales")
        return self.node(
            "Resize", [x, "", scales], hint,
            mode="linear", coordinate_transformation_mode="align_corners",
        )

    def model(self) -> onnx.ModelProto:
        graph = helper.make_graph(self.nodes, self.name, self.inputs, self.outputs, self.inits)
        model = helper.make_model(
            graph,
            producer_name="pixelmind-tiny",
            opset_imports=[helper.make_opsetid("", OPSET)],
        )
        # The Android runtime bundled with the app is ONNX Runtime 1.18, which reads IR <= 9.
        # Recent `onnx` releases default to IR 14, so pin it down explicitly.
        model.ir_version = 8
        return model


# ---------------------------------------------------------------------------------------------
# text encoder: token ids -> [1, 77, 768] conditioning
# ---------------------------------------------------------------------------------------------

def build_text_encoder() -> onnx.ModelProto:
    g = Graph("text_encoder")
    g.add_input("input_ids", TensorProto.INT64, [1, SEQ_LEN])
    g.add_output("last_hidden_state", TensorProto.FLOAT, [1, SEQ_LEN, HIDDEN])

    table = rng.normal(0, 0.06, (VOCAB_SIZE, EMBED_DIM)).astype(np.float32)
    table_name = g.uid("embedding")
    g.inits.append(numpy_helper.from_array(table, table_name))

    w = rng.normal(0, 1.0 / np.sqrt(EMBED_DIM), (EMBED_DIM, HIDDEN)).astype(np.float32)
    b = np.zeros(HIDDEN, dtype=np.float32)
    wn, bn = g.uid("proj_w"), g.uid("proj_b")
    g.inits.append(numpy_helper.from_array(w, wn))
    g.inits.append(numpy_helper.from_array(b, bn))

    gathered = g.node("Gather", [table_name, "input_ids"], "gather", axis=0)   # [1,77,32]

    # Mask out padding. Without this, 64 of 77 positions are the same EOS token and the pooled
    # text vector is dominated by padding, so every prompt conditions the image identically.
    eos = g.const_i64(np.array([EOS_ID]), "eos")
    is_pad = g.node("Equal", ["input_ids", eos], "is_pad")                      # [1,77] bool
    keep = g.node("Not", [is_pad], "keep")
    mask = g.node("Cast", [keep], "mask", to=TensorProto.FLOAT)                 # [1,77]
    mask3 = g.node("Reshape", [mask, g.const_i64(np.array([1, SEQ_LEN, 1]), "mask_shape")], "mask3")
    gathered = g.node("Mul", [gathered, mask3], "masked")                       # pads -> 0

    projected = g.node("MatMul", [gathered, wn], "matmul")                      # [1,77,768]
    g.node("Add", [projected, bn], "last_hidden_state", out_name="last_hidden_state")
    return g.model()


# ---------------------------------------------------------------------------------------------
# UNet: epsilon prediction conditioned on timestep + text
# ---------------------------------------------------------------------------------------------

def build_unet() -> onnx.ModelProto:
    g = Graph("unet")
    g.add_input("sample", TensorProto.FLOAT, [1, LATENT_CHANNELS, "H", "W"])
    g.add_input("timestep", TensorProto.INT64, [1])
    g.add_input("encoder_hidden_states", TensorProto.FLOAT, [1, SEQ_LEN, HIDDEN])
    g.add_output("out_sample", TensorProto.FLOAT, [1, LATENT_CHANNELS, "H", "W"])

    MID = 64
    h = g.conv("sample", LATENT_CHANNELS, MID, "conv_in")
    h = g.silu(h, "act_in")

    # --- text conditioning: mean over tokens -> per-channel scale and shift ----------------
    # Two separate projections instead of a Split node: Split's signature changed in opset 13
    # and this is clearer anyway.
    # Sum, because padding is already zeroed: this is the prompt's content, not an average
    # diluted by 77 positions.
    # ReduceSum takes `axes` as an input tensor since opset 13, not as an attribute.
    axes = g.const_i64(np.array([1]), "pool_axes")
    pooled = g.node("ReduceSum", ["encoder_hidden_states", axes], "text_pool", keepdims=0)
    # L2-normalise, then amplify: the raw sum has a magnitude that depends on prompt length, and
    # a small text vector produces modulation too weak to see in the output.
    pooled_sq = g.node("Mul", [pooled, pooled], "pooled_sq")
    paxes = g.const_i64(np.array([1]), "pooled_axes")
    pooled_ss = g.node("ReduceSum", [pooled_sq, paxes], "pooled_ss", keepdims=1)
    pooled_norm = g.node("Sqrt", [pooled_ss], "pooled_norm")
    pooled_norm = g.node("Add", [pooled_norm, g.const(np.array([1e-6]), "pooled_eps")], "pooled_safe")
    pooled = g.node("Div", [pooled, pooled_norm], "pooled_unit")
    pooled = g.node("Mul", [pooled, g.const(np.array([TEXT_GAIN]), "text_gain")], "text_pooled")

    def text_proj(hint: str, width: int = MID) -> str:
        w = rng.normal(0, 1.0 / np.sqrt(HIDDEN), (HIDDEN, width)).astype(np.float32)
        wn = g.uid(f"{hint}_w")
        g.inits.append(numpy_helper.from_array(w, wn))
        return g.node("MatMul", [pooled, wn], hint)          # [1, 64]

    def to_chw(name: str, hint: str, channels: int = MID) -> str:
        shape = g.const_i64(np.array([1, channels, 1, 1]), f"shape{channels}")
        return g.node("Reshape", [name, shape], hint)        # [1, C, 1, 1]

    shift = to_chw(text_proj("text_shift"), "text_shift_r")
    scale = to_chw(text_proj("text_scale"), "text_scale_r")
    # Scale stays in a sane band: 0.5x .. 2.5x of the feature map.
    scale = g.node("Tanh", [scale], "text_scale_t")
    scale = g.node("Mul", [scale, g.const(np.array([1.0]), "one")], "text_scale_k")
    scale = g.node("Add", [scale, g.const(np.array([1.5]), "bias")], "text_scale_b")

    h = g.node("Mul", [h, scale], "mod_scale")
    h = g.node("Add", [h, shift], "mod_shift")

    # --- timestep conditioning --------------------------------------------------------------
    t_f = g.node("Cast", ["timestep"], "t_cast", to=TensorProto.FLOAT)     # [1]
    t_n = g.node("Div", [t_f, g.const(np.array([999.0]), "t_scale")], "t_norm")
    wt = rng.normal(0, 1.0, (1, MID)).astype(np.float32)
    wtn = g.uid("time_w")
    g.inits.append(numpy_helper.from_array(wt, wtn))
    t_proj = g.node("MatMul", [t_n, wtn], "t_proj")                        # [1, 64]
    t_gate = g.node("Sigmoid", [t_proj], "t_gate")
    t_gate = to_chw(t_gate, "t_gate_r")
    h = g.node("Mul", [h, t_gate], "mod_time")

    # --- body --------------------------------------------------------------------------------
    body = g.conv(h, MID, MID, "conv_mid")
    body = g.silu(body, "act_mid")
    out = g.conv(body, MID, LATENT_CHANNELS, "conv_out")
    # A tanh here would clamp the noise prediction to +/-1 while the latent lives around +/-7,
    # so the sampler could never cancel it and the trajectory would diverge. A plain linear
    # output with a tuned gain keeps the step stable.
    out = g.node("Mul", [out, g.const(np.array([OUTPUT_GAIN]), "out_gain")], "out_scaled")

    # A direct prompt term on the noise prediction, so different prompts steer the trajectory
    # apart instead of only nudging an internal feature map.
    text_out = to_chw(text_proj("text_out", LATENT_CHANNELS), "text_out_r", LATENT_CHANNELS)
    text_out = g.node("Mul", [text_out, g.const(np.array([OUT_TEXT_GAIN]), "text_out_gain")], "text_out_k")
    out = g.node("Add", [out, text_out], "out_with_text")
    g.node("Identity", [out], "out_sample", out_name="out_sample")
    return g.model()


# ---------------------------------------------------------------------------------------------
# VAE decoder: latents -> RGB, x8 upsampling
# ---------------------------------------------------------------------------------------------

def build_vae() -> onnx.ModelProto:
    g = Graph("vae_decoder")
    g.add_input("latent_sample", TensorProto.FLOAT, [1, LATENT_CHANNELS, "H", "W"])
    g.add_output("sample", TensorProto.FLOAT, [1, 3, "8H", "8W"])

    # Normalise the incoming latents by their own RMS. A random-weight UNet cannot cancel its
    # own noise prediction the way a trained one does, so the trajectory's scale drifts; making
    # the decoder scale-invariant keeps the output well exposed instead of blown out to pure
    # black and white. Structure (and therefore the prompt) lives in the pattern, not the
    # magnitude.
    sq = g.node("Mul", ["latent_sample", "latent_sample"], "rms_sq")
    # ReduceMean still takes `axes` as an attribute at opset 17 (it only became an input in 18).
    mean_sq = g.node("ReduceMean", [sq], "rms_mean", axes=[0, 1, 2, 3], keepdims=1)
    rms = g.node("Sqrt", [mean_sq], "rms")
    rms = g.node("Add", [rms, g.const(np.array([1e-3]), "rms_eps")], "rms_safe")
    normed = g.node("Div", ["latent_sample", rms], "rms_normed")

    # Collapse the latent grid 8x before doing anything else. Untrained weights have no reason to
    # prefer low spatial frequencies, so a per-pixel decoder renders the latent as RGB static.
    # Pooling first turns that static into smooth colour fields — abstract art rather than noise.
    coarse = g.node("AveragePool", [normed], "coarse", kernel_shape=[8, 8], strides=[8, 8])

    h = g.conv(coarse, LATENT_CHANNELS, 64, "dec_in")
    h = g.silu(h, "dec_act0")

    # Three x4 stages take 8 -> 32 -> 128 -> 512, matching the 8x latent-to-pixel ratio.
    for stage, (cin, cout) in enumerate([(64, 32), (32, 16), (16, 16)]):
        h = g.upscale(h, 4, f"up{stage}")
        h = g.conv(h, cin, cout, f"dec_conv{stage}")
        h = g.silu(h, f"dec_act{stage + 1}")

    h = g.conv(h, 16, 3, "dec_out")

    # Centre each colour channel before the tanh: random weights otherwise land on an arbitrary
    # colour cast (an early build came out uniformly green).
    ch_mean = g.node("ReduceMean", [h], "ch_mean", axes=[2, 3], keepdims=1)
    h = g.node("Sub", [h, ch_mean], "ch_centred")

    # Then normalise the field by its own standard deviation so contrast is independent of how
    # large the random weights happen to make it. Without this the output is a flat grey sheet.
    c_sq = g.node("Mul", [h, h], "c_sq")
    c_mean = g.node("ReduceMean", [c_sq], "c_mean", axes=[0, 1, 2, 3], keepdims=1)
    c_std = g.node("Sqrt", [c_mean], "c_std")
    c_std = g.node("Add", [c_std, g.const(np.array([1e-4]), "c_eps")], "c_std_safe")
    h = g.node("Div", [h, c_std], "c_normed")
    h = g.node("Mul", [h, g.const(np.array([0.95]), "ch_gain")], "ch_scaled")
    g.node("Tanh", [h], "sample", out_name="sample")
    return g.model()


# ---------------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_tokenizer(out_dir: Path) -> tuple[Path, Path]:
    clip_dir = out_dir / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for name in ("vocab.json", "merges.txt"):
        target = clip_dir / name
        if target.exists() and target.stat().st_size > 1000:
            continue
        url = f"{TOKENIZER_BASE}/{name}"
        print(f"[tiny] fetching {name} from {url}")
        with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        print(f"[tiny]   {name}: {target.stat().st_size} bytes")
    return clip_dir / "vocab.json", clip_dir / "merges.txt"


def save(model: onnx.ModelProto, path: Path) -> None:
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print(f"[tiny] wrote {path.name}: {path.stat().st_size / 1048576:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PixelMind's tiny verification model")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    save(build_text_encoder(), out_dir / "text_encoder.onnx")
    save(build_unet(), out_dir / "unet.onnx")
    save(build_vae(), out_dir / "vae_decoder.onnx")
    fetch_tokenizer(out_dir)

    components = []
    for name in ("text_encoder.onnx", "unet.onnx", "vae_decoder.onnx",
                 "clip/vocab.json", "clip/merges.txt"):
        path = out_dir / name
        components.append({
            "file": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        })

    manifest = {
        "name": "PixelMind Tiny (verification)",
        "version": "0.1.0",
        "architecture": "tiny-unet-lcm-compatible",
        "precision": "fp32",
        "license": "MIT (synthetic weights, not a diffusion checkpoint)",
        "latent_channels": LATENT_CHANNELS,
        "sample_size": 64,
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
            "file": "vae_decoder.onnx",
            "inputs": ["latent_sample"],
            "output": "sample",
        },
        "tokenizer": {
            "vocab": "clip/vocab.json",
            "merges": "clip/merges.txt",
            "max_length": SEQ_LEN,
            "bos": "<|startoftext|>",
            "eos": "<|endoftext|>",
            "pad": "<|endoftext|>",
        },
    }
    (out_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(c["bytes"] for c in components)
    print(f"[tiny] manifest written, total {total / 1048576:.2f} MB")
    print("[tiny] this is a plumbing test, not Stable Diffusion — see the docstring")


if __name__ == "__main__":
    main()
