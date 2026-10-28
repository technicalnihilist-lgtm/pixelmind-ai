#!/usr/bin/env python3
"""Runs the full PixelMind pipeline against a model directory, mirroring the Kotlin engine.

    python tools/smoke_test_model.py <model-dir> --prompt "..." --out preview.png

It reproduces, step for step, what OnnxDiffusionEngine does on the phone: CLIP BPE tokenisation,
text conditioning, the LCM consistency loop (including classifier-free guidance), and the VAE
decode. If this produces an image, the on-device path will too — the graphs and the maths are
identical; only the runtime differs.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

LATENT_SCALE = 0.18215
SEQ_LEN = 77


# --- CLIP byte-pair tokenizer, mirroring app/src/main/java/ai/pixelmind/engine/ClipTokenizer.kt ---

def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) \
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


PATTERN = re.compile(
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[\d]|[^\s\w]+", re.UNICODE
)


class ClipTokenizer:
    def __init__(self, vocab_path: Path, merges_path: Path):
        self.encoder = json.loads(vocab_path.read_text())
        self.byte_encoder = bytes_to_unicode()
        self.bpe_ranks = {}
        with merges_path.open(encoding="utf-8") as handle:
            lines = [ln.strip() for ln in handle if ln.strip() and not ln.startswith("#")]
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = i
        self.cache: dict[str, list[str]] = {}

    def _bpe(self, token: str) -> list[str]:
        if token in self.cache:
            return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        pairs = set(zip(word[:-1], word[1:]))
        while pairs:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            pairs = set(zip(word[:-1], word[1:]))
        self.cache[token] = word
        return word

    def encode(self, text: str) -> np.ndarray:
        ids = [self.encoder["<|startoftext|>"]]
        for match in PATTERN.findall(text.lower()):
            mapped = "".join(self.byte_encoder[b] for b in match.encode("utf-8"))
            for token in self._bpe(mapped):
                if token in self.encoder and len(ids) < SEQ_LEN - 1:
                    ids.append(self.encoder[token])
        ids.append(self.encoder["<|endoftext|>"])
        ids = ids + [self.encoder["<|endoftext|>"]] * (SEQ_LEN - len(ids))
        return np.array([ids[:SEQ_LEN]], dtype=np.int64)


# --- LCM scheduler, mirroring LcmScheduler.kt ------------------------------------------------

class LcmScheduler:
    def __init__(self, num_train=1000, beta_start=0.00085, beta_end=0.012, sigma_data=0.5):
        self.n = num_train
        self.sigma_data = sigma_data
        t = np.arange(num_train, dtype=np.float64) / (num_train - 1)
        betas = (np.sqrt(beta_start) + t * (np.sqrt(beta_end) - np.sqrt(beta_start))) ** 2
        self.alphas_cumprod = np.cumprod(1.0 - betas)

    def timesteps(self, steps: int) -> list[int]:
        ratio = self.n // steps
        return [self.n - 1 - i * ratio for i in range(steps)]

    def sigma(self, t: int) -> float:
        alpha = self.alphas_cumprod[max(0, min(self.n - 1, t))]
        return float(np.sqrt((1 - alpha) / alpha))

    def initial_scale(self, t: int) -> float:
        s = self.sigma(t)
        return float(np.sqrt(s * s + self.sigma_data ** 2))

    def step(self, latent: np.ndarray, model_output: np.ndarray, t: int, prev: int) -> None:
        sigma = self.sigma(t)
        alpha_prev = 1.0 if prev < 0 else self.alphas_cumprod[prev]
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / np.sqrt(sigma ** 2 + self.sigma_data ** 2)
        eps = c_out * model_output + c_skip * latent
        latent[...] = (latent - np.sqrt(1 - alpha_prev) * eps) / np.sqrt(alpha_prev)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="a brass astrolabe on mossy granite, morning light")
    parser.add_argument("--negative", default="")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("preview.png"))
    args = parser.parse_args()

    manifest = json.loads((args.model_dir / "model_manifest.json").read_text())
    root = args.model_dir
    providers = ["CPUExecutionProvider"]

    print(f"model: {manifest['name']} {manifest['version']}")

    tokenizer = ClipTokenizer(root / manifest["tokenizer"]["vocab"],
                              root / manifest["tokenizer"]["merges"])
    encoder = ort.InferenceSession(str(root / manifest["text_encoder"]["file"]), providers=providers)
    unet = ort.InferenceSession(str(root / manifest["unet"]["file"]), providers=providers)
    vae = ort.InferenceSession(str(root / manifest["vae_decoder"]["file"]), providers=providers)

    started = time.time()
    ids = tokenizer.encode(args.prompt)
    print(f"tokens: {int((ids != tokenizer.encoder['<|endoftext|>']).sum())} non-pad")

    cond = encoder.run([manifest["text_encoder"]["output"]], {"input_ids": ids})[0]
    print(f"conditioning: {cond.shape}")

    uncond = None
    if args.guidance > 1.05:
        uncond_ids = tokenizer.encode(args.negative)
        uncond = encoder.run([manifest["text_encoder"]["output"]], {"input_ids": uncond_ids})[0]

    scheduler = LcmScheduler()
    timesteps = scheduler.timesteps(args.steps)
    latent_h = latent_w = args.size // 8
    channels = manifest["latent_channels"]

    rng = np.random.default_rng(args.seed)
    scale = scheduler.initial_scale(timesteps[0])
    latent = (rng.standard_normal((1, channels, latent_h, latent_w)) * scale).astype(np.float32)

    for i, t in enumerate(timesteps):
        prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        sample = latent * LATENT_SCALE
        eps = unet.run([manifest["unet"]["output"]], {
            "sample": sample,
            "timestep": np.array([t], dtype=np.int64),
            "encoder_hidden_states": cond,
        })[0]
        if uncond is not None:
            eps_u = unet.run([manifest["unet"]["output"]], {
                "sample": sample,
                "timestep": np.array([t], dtype=np.int64),
                "encoder_hidden_states": uncond,
            })[0]
            eps = eps_u + args.guidance * (eps - eps_u)
        scheduler.step(latent, eps, t, prev)
        print(f"  step {i + 1}/{len(timesteps)}  t={t}  latent std={float(latent.std()):.3f}")

    rgb = vae.run([manifest["vae_decoder"]["output"]], {"latent_sample": latent / LATENT_SCALE})[0]
    print(f"decoded: {rgb.shape}")

    pixels = np.clip((rgb[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(args.out)
    print(f"wrote {args.out}  ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
