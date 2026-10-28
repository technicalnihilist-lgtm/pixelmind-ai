# Making the model work

## What the shipped APK actually runs

The APK ships the **verification model**: 8.2 MB of real ONNX graphs with the same tensor
contract as Stable Diffusion, but untrained weights. Confirm it yourself:

```bash
unzip -l pixelmind-ai-1.0.0-arm64.apk | grep assets/models
#   assets/models/text_encoder.onnx     6.1 MB
#   assets/models/unet.onnx             0.5 MB
#   assets/models/vae_decoder.onnx      0.1 MB
#   assets/models/clip/vocab.json
#   assets/models/clip/merges.txt
#   assets/models/model_manifest.json
```

It generates images on device, offline, today. They are smooth abstract colour fields — the
network is untrained, so there is nothing for it to have learned about leopards or astrolabes.
What it *does* prove is the whole path: real CLIP BPE tokenisation, real text conditioning, a real
LCM sampling loop, real VAE decode, a real PNG in app-private storage.

Real photographic quality needs the production weights (0.6–1.2 GB), which cannot live in this
repository — GitHub rejects files that size. That is a build step, not a code change.

## What you need

| | |
| --- | --- |
| RAM | 16 GB comfortable, 8 GB will swap |
| Disk | ~20 GB free (model download + ONNX exports) |
| Time | 20–40 minutes, mostly downloading |
| Python | 3.10–3.12 |
| GPU | optional, only speeds up the export |

## Step 1 — export the production weights

```bash
git clone <your-repo> && cd pixelmind-ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install "torch>=2.2" "diffusers>=0.27" transformers onnx onnxruntime safetensors

# Put the weights where a plain APK build will pick them up (see the trap below).
python tools/export_model.py --out app/src/main/assets/models
```

This replaces the verification model in `app/src/main/assets/models` (delete the directory first
if you want a clean swap). That script downloads SD 1.5 + LCM-LoRA, fuses them, exports `text_encoder.onnx`,
`unet.onnx`, `vae_decoder.onnx`, int8-quantises the first two, copies the CLIP tokenizer, and
writes `model_manifest.json` with sizes and SHA-256 checksums.

Check the result before building:

```bash
python tools/verify_export.py app/src/main/assets/models
```

## Step 2 — build and install

```bash
./gradlew :app:assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
```

First launch copies the graphs from `assets/` into app-private storage (a local file copy —
ONNX Runtime needs a real path), then the engine loads them. Home should show
`Local AI · XNNPACK` and Settings → Local model should show the weights size.

## The trap: asset packs and plain APKs

`modelpack/` is a Play Asset Delivery **install-time** pack. It is only delivered as part of an
**App Bundle**:

```bash
./gradlew :app:bundleRelease
bundletool build-apks --bundle=app/build/outputs/bundle/release/app-release.aab \
                      --output=pixelmind.apks --mode=universal --local-testing
bundletool install-apks --apks=pixelmind.apks
```

`adb install` of a plain APK **will not** include the pack — the app will still say "Weights
missing" even though you just exported them. That is the single most common way to think the
export failed when it did not.

For local testing, skip the pack entirely: export into `app/src/main/assets/models` as shown
above. `ModelLocator` checks that location first, and a normal APK build bundles it.

## Expected performance

Untested on hardware for this build (see [VERIFICATION.md](VERIFICATION.md)), so treat these as
planning figures, not measurements:

| Device tier | 512×512, 4 LCM steps |
| --- | --- |
| Flagship (SD 8 Gen 2+, NNAPI accepted) | ~10–20 s |
| Mid-range (2019–2022, XNNPACK) | ~35–70 s |
| Older / 4 GB RAM | 90 s+, possibly reduced to 384×384 by `MemoryGuard` |

The app shows the real step count and elapsed time while it runs, and Settings shows which
accelerator was actually selected.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Still "Weights missing" after export | You built an APK while the weights sat in `modelpack/`. See the trap above. |
| `Checksum mismatch for unet.onnx` | The file was truncated in copy, or you edited a graph after the manifest was written. Re-run the export. |
| Killed during generation | Peak RAM exceeded the device budget. Lower the quality tier or aspect ratio; `MemoryGuard` already degrades, but a 4 GB device has a hard ceiling. |
| "Engine error: …" on Home | Read the actual exception with `adb logcat -s OnnxDiffusionEngine ModelLocator`. |
| Very slow first run | ONNX Runtime builds its optimisation plan on first session creation. Subsequent runs in the same process are faster. |
