# Model options — what actually exists, and what it costs

This is a survey of every pre-exported ONNX text-to-image model that could plausibly drop into
PixelMind's tensor contract, with the sizes and validity findings measured on 2026-10-28.

The short version: **there is no usable model under ~950 MB**, and the one public export that is
small enough to be comfortable (618 MB) is structurally broken.

## The arithmetic nobody escapes

Stable Diffusion 1.5's UNet is 860 M parameters. That is not compressible by app code:

| Precision | UNet alone | Full pipeline (UNet + text encoder + VAE) |
| --- | --- | --- |
| fp32 | 3.3 GB | ~3.9 GB |
| fp16 | 1.7 GB | ~1.9 GB |
| int8 | 860 MB | ~1.17 GB |

A request for a "300 MB image model" is asking for a model roughly a quarter of SD 1.5. Distilled
variants (`tiny-sd`, `bk-sdm`) get to about 600 MB at int8 — that is the floor for anything that
produces recognisable images.

## Surveyed exports

| Repository | UNet | Text enc | VAE | Total | Verdict |
| --- | --- | --- | --- | --- | --- |
| `szFangshi/tiny-sd-int8` | 311 MB | 119 MB | 189 MB | **618 MB** | **Broken** — see below |
| `RanaLLC/tiny-sd-onnx-fp16` | 617 MB | 235 MB | 94 MB | 946 MB | Loads only with >2 GB free |
| `akameswa/lcm-tiny-sd-onnx-fp16` | 617 MB | 235 MB | 94 MB | 946 MB | LCM-distilled; same size |
| `danyalmalik/bk-sdm-small-2m-ONNX-WebGPU` | 921 MB | 235 MB | 189 MB | 1.34 GB | Too large for 4 GB phones |
| `tlwu/stable-diffusion-v1-5-onnxruntime` | 1.64 GB | 235 MB | 94 MB | 1.97 GB | Full SD 1.5, fp16 |
| `onnx-community/stable-diffusion-v1-5-ONNX` | 3.28 GB | 470 MB | 189 MB | 3.9 GB | fp32, impractical on mobile |
| `endman100/stable-diffusion-v1-5-onnx` | 1.64 GB (split) | 235 MB | 94 MB | 1.97 GB | Full SD 1.5, fp16 |

Everything in that table that loads is at least 946 MB. The only 618 MB option does not load.

## Why `tiny-sd-int8` fails

Two independent defects, found by loading it and then inspecting the graph:

1. **Duplicate tensor definitions (50 of them).** For every quantised conv and matmul, the
   producer node was given its downstream `Cast` node's output name:

   ```
   QLinearConv -> /conv_in/Conv_output_0_output_quantized_cast_output   <- wrong
   Cast        -> /conv_in/Conv_output_0_output_quantized_cast_output   <- correct
   ```

   ONNX Runtime reports this as `Duplicate definition of name`. `tools/repair_onnx.py` fixes this
   layer by restoring the producer's name, which resolves 50 of 50.

2. **Misaligned QLinearConv inputs (unfixable without re-quantising).** With the names restored,
   the graph is cyclic, because a conv reads its own output as a quantisation parameter:

   ```
   QLinearConv  in[5] = .../Conv_output_0_output_quantized_cast_output
                out[0] = .../Conv_output_0_output_quantized_cast_output
   ```

   `QLinearConv` takes 3 or 8 inputs. These nodes have 6, with the weight tensor where the
   zero-point belongs and the output name where the scale belongs. The quantisation parameters
   were never written, so there is nothing to restore — the graph must be re-quantised from fp32
   weights, which needs roughly 4 GB of RAM.

`tools/repair_onnx.py` is kept because defect 1 is a recurring pattern in hand-rolled quantised
exports and the tool diagnoses it quickly. It cannot fix defect 2.

## What to do instead

Export your own. It is one command and it produces a model that is verified as it is built:

```bash
pip install "torch>=2.2" "diffusers>=0.27" transformers onnx onnxruntime
python tools/export_model.py --out app/src/main/assets/models   # ~600 MB int8, ~30 min, 16 GB RAM
python tools/verify_export.py app/src/main/assets/models
./gradlew :app:assembleRelease
```

`export_model.py` fuses LCM-LoRA into SD 1.5, exports the three graphs, applies per-channel int8
quantisation to the UNet and text encoder, and writes a manifest with SHA-256 checksums. Unlike
every model in the table above, it is built for this app's exact tensor contract and checked
before you spend time on a build.

## Sizes to expect in the APK

| Configuration | Weights | APK |
| --- | --- | --- |
| Verification model (shipped) | 8.2 MB | 29 MB |
| tiny-sd / distilled, int8 | ~600 MB | ~630 MB |
| SD 1.5 + LCM, int8 | ~1.2 GB | ~1.2 GB |
| SD 1.5 + LCM, fp16 | ~1.9 GB | ~1.9 GB |

Play Asset Delivery supports packs up to 4 GB, delivered at install time — so the 1.2 GB
configuration is distributable through Play, just not as a single sideloaded APK.
