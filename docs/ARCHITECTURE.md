# Architecture

## Layers

1. **UI (`ui/`)** — Compose only. Screens read `StateFlow`s from one `PixelMindViewModel` and
   render. No inference, no file I/O, no tensor maths.
2. **Orchestration (`manager/GenerationManager`)** — owns a single worker coroutine, a job
   queue, progress reporting, cancellation and engine lifetime. It is the only class allowed to
   call `LocalAIEngine`.
3. **Inference (`engine/`)** — `OnnxDiffusionEngine` behind the `LocalAIEngine` interface.
   Contains the CLIP tokenizer, the LCM scheduler, tensor marshalling and the VAE decode.
4. **Storage (`data/`, `util/ImageStore`)** — SQLite for history, app-private files for PNGs.

## Why one generation at a time

A single 512×512 pass peaks around 700–900 MB. Two concurrent passes would exceed what most
phones will give a foreground app before the low-memory killer intervenes. The manager therefore
serialises: extra requests queue, and the UI shows them as `QUEUED` rather than silently
dropping them.

## Pipeline

```
prompt ──▶ ClipTokenizer ──▶ text_encoder.onnx ──▶ [1,77,768] conditioning
                                                          │
noise (seeded) ──▶ LCM loop (N steps) ◀───────────────────┘
                     │  UNet int8 per step (+ CFG when guidance > 1.05)
                     ▼
              latents ──▶ vae_decoder.onnx ──▶ float RGB ──▶ ARGB_8888 ──▶ PNG
```

* **Tokenizer** — CLIP byte-pair encoding ported in `ClipTokenizer.kt`, including the
  byte→unicode table and the `</w>` end-of-word marker. Getting either wrong turns conditioning
  into noise.
* **Scheduler** — `LcmScheduler.kt` implements the latent-consistency update with the
  `scaled_linear` beta schedule. Four steps is the default because that is what the distillation
  was trained for; more steps cost time without adding detail.
* **Classifier-free guidance** — when guidance > 1.05 the engine runs the UNet twice per step
  (conditional + negative prompt) and mixes. That doubles step cost, which is why the default
  LCM configuration sits at guidance 1.0–1.4.
* **Determinism** — the seed drives `java.util.Random`, which is specified by the JDK, so the
  same seed produces the same image on every device.

## Memory strategy

`MemoryGuard` estimates peak usage before the first step:

```
weights (≈640 MB resident) + activations (∝ pixels) + VAE decode buffer + overhead
```

If the estimate exceeds 75% of the device's heap class it degrades the request — first steps,
then resolution — and returns `null` (a clear error) rather than attempting a run that would be
killed halfway. Sessions are closed when the app leaves the foreground
(`ProcessLifecycleOwner` in `PixelMindApp`), because holding ~640 MB of weights in a background
process is how you get killed.

## Threading

* UI: main thread, Compose only.
* Generation: `Dispatchers.Default`; ONNX Runtime is given `min(4, cores)` intra-op threads and
  one inter-op thread.
* Disk and SQLite: `Dispatchers.IO`.
* Cancellation is cooperative — `ensureActive()` runs between diffusion steps, so a cancel
  releases the loop within one step.

## Native layer (optional)

`app/src/main/cpp/pixelmind_native.cpp` implements the scheduler step as a NEON elementwise
kernel, called through `NativeScheduler`. The Kotlin implementation runs when the `.so` is
absent, and both are fed identical scalars from `LcmScheduler`, so results agree. It is an
optimisation, not a dependency — the default build does not compile it. See
[BUILD.md](BUILD.md).
