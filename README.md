# PixelMind AI

An Android text-to-image app that runs its diffusion model entirely on the device: no API keys,
no server, no analytics, and no `INTERNET` permission in the manifest.

Kotlin + Jetpack Compose UI, ONNX Runtime as the local inference layer, and a Liquid Glass
design system.

**Download the APK:** [pixelmind-ai-1.1.0-arm64.apk](https://github.com/technicalnihilist-lgtm/pixelmind-ai/releases/download/v1.1.0/pixelmind-ai-1.1.0-arm64.apk)
· 29 MB · arm64-v8a · Android 8.0+ · SHA-256 `902907b542cd8d705817d6fe51d3e0ad4c969d354ae7d51936bc1b1a8191075c`

Screens: [Home](design/01-home.png) · [Studio](design/02-studio.png) · [Gallery](design/03-gallery.png) · [Settings](design/04-settings.png) · [sample output](design/05-sample-output.png)

---

## Status — read this first

This repository is an honest snapshot, not a launch announcement. Here is exactly what is
verified and what is not.

| Claim | State |
| --- | --- |
| Project compiles, APKs build | **Verified.** `assembleDebug` and `assembleRelease` both succeed. |
| Unit tests pass | **Verified.** 13/13 pass (`./gradlew :app:testDebugUnitTest`). |
| Manifest requests no platform permissions | **Verified** on the built APK with `aapt2 dump permissions`. |
| Real ONNX inference code (no mocks, no stock images) | **Implemented** — CLIP BPE tokenizer, LCM scheduler, UNet/VAE graph execution. |
| Weights bundled in the APK | **Yes** — the *verification* model (8.2 MB): real ONNX graphs with the same shapes as the SD pipeline, untrained weights. |
| Image generation works in the shipped APK | **Yes, end to end** — tokenizer → conditioning → 4 LCM steps → VAE → PNG, on device, offline. Output is smooth abstract colour fields, **not** photoreal imagery. |
| Photoreal / SD-quality output | **No.** Needs the 600 MB Stable Diffusion export: `tools/export_model.py`. The app code does not change. |
| Tested on a physical device | **Not done.** No Android hardware was available to this build. |
| Airplane-mode generation verified | **Not done** (depends on the two rows above). |
| Screenshots | The images in `design/` are **UI renders produced from the Compose layout**, not device screenshots. |

Anything below marked "measured" was measured on the build machine described in
[docs/MEASUREMENTS.md](docs/MEASUREMENTS.md). Nothing is estimated without saying so.

---

## Two models, one pipeline

| | Verification model (shipped) | Production model (opt-in) |
| --- | --- | --- |
| Weights | 8.2 MB, synthetic | ~600 MB, SD 1.5 + LCM-LoRA |
| Build | `tools/make_tiny_model.py` | `tools/export_model.py` |
| Output | smooth abstract colour fields | photoreal / artistic images |
| Purpose | proves the on-device path runs | the actual product |

The shipped APK generates images today. It does not pretend they are photographs: the weights are
untrained, so what comes out is prompt-dependent abstract colour — honest output from a real
pipeline rather than a stock image presented as AI.

Swapping in the real model is a build step, not a code change:

```bash
python tools/export_model.py --out app/src/main/assets/models   # ~20-40 min, needs ~16 GB RAM
python tools/verify_export.py app/src/main/assets/models
./gradlew :app:assembleRelease && adb install -r app/build/outputs/apk/release/app-release.apk
```

Full walkthrough, including the asset-pack trap that makes a correct export look broken:
**[docs/GETTING_IT_RUNNING.md](docs/GETTING_IT_RUNNING.md)**.

---

## Architecture

```
 ┌───────────────┐   prompt    ┌──────────────────────┐
 │  Compose UI   │────────────▶│  GenerationManager   │  queue · cancel · retry
 │  (screens)    │◀────────────│  (StateFlow)         │
 └───────────────┘  progress   └──────────┬───────────┘
                                          │ LocalAIEngine (interface)
                                          ▼
                               ┌──────────────────────┐
                               │  OnnxDiffusionEngine │
                               │  ┌────────────────┐  │
                               │  │ CLIP encoder   │  │
                               │  │ LCM UNet int8  │  │  ONNX Runtime
                               │  │ VAE decoder    │  │
                               │  └────────────────┘  │
                               └──────────┬───────────┘
                                          ▼
                               ┌──────────────────────┐
                               │  ImageStore → PNG    │
                               │  SQLite history      │
                               └──────────────────────┘
```

The UI never touches ONNX Runtime. Everything the interface can see is `LocalAIEngine`
(`engine/EngineContract.kt`); the runtime, tensor layouts and quantisation stay behind it.

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Modules

| Path | What lives there |
| --- | --- |
| `app/src/main/java/ai/pixelmind/engine/` | Inference: tokenizer, scheduler, ONNX pipeline, memory guard, model provisioning |
| `app/src/main/java/ai/pixelmind/manager/` | `GenerationManager` — queue, progress, cancellation, engine lifecycle |
| `app/src/main/java/ai/pixelmind/data/` | SQLite history, favourites, prompt history, settings |
| `app/src/main/java/ai/pixelmind/ui/` | Compose screens, Liquid Glass design system, navigation |
| `app/src/main/cpp/` | Optional NEON JNI scheduler (see [docs/BUILD.md](docs/BUILD.md)) |
| `modelpack/` | Play Asset Delivery **install-time** pack that carries the weights |
| `tools/export_model.py` | Exports + quantises Stable Diffusion 1.5 + LCM into the ONNX graphs this app loads |
| `tools/make_tiny_model.py` | Builds the 8 MB verification model that ships in the APK |
| `tools/smoke_test_model.py` | Runs the whole pipeline in Python against any model directory |

## Build

```bash
# 1. produce weights — pick one:
python tools/make_tiny_model.py --out app/src/main/assets/models   # 8 MB verification model, ~10s
python tools/export_model.py   --out app/src/main/assets/models   # 600 MB production model, ~30min

# 2. build
./gradlew :app:assembleRelease
```

Weights are **not** committed: they are generated artifacts, and the 600 MB production export has
no business in git history. The verification model is reproducible from a fixed RNG seed in about
ten seconds, which is why a fresh clone builds to a working APK in two commands.

Without step 1 the app still builds and installs; it simply reports that the weights are
missing instead of pretending to generate. See [docs/BUILD.md](docs/BUILD.md).

## Measured artifacts

| Artifact | Size |
| --- | --- |
| `app-release.apk` (arm64-v8a, with the 8.2 MB verification model) | 29.0 MB (30,387,428 bytes) |
| `app-release.apk` (earlier build, no weights) | 22.2 MB (23,254,529 bytes) |
| ↳ native (ONNX Runtime) | 16.3 MB uncompressed |
| ↳ dex | 16.2 MB uncompressed |
| ↳ resources + manifest | 0.44 MB uncompressed |
| ↳ bundled verification weights | 8.2 MB uncompressed |

Swapping in the production model grows the install by roughly 0.6–1.2 GB depending on the
precision you export at. That is the honest cost of offline generation — no amount of app-code
optimisation changes it.

## Privacy

No `INTERNET` permission. No networking library. No analytics SDK. No account. Cloud backup and
device-transfer are disabled for every data domain. Details and how to verify:
[docs/PRIVACY.md](docs/PRIVACY.md).

## Model licensing

The pipeline targets **Stable Diffusion 1.5** (CreativeML OpenRAIL-M) with the **LCM-LoRA**
consistency distillation. Both carry use restrictions you must honour if you redistribute the
weights. Read [docs/MODEL.md](docs/MODEL.md) before shipping anything built from this code.

## License

Application code: MIT (see `LICENSE`). Model weights: not included, and not covered by that
license.
