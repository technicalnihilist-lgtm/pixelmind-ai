# The bundled models

The app has one pipeline and two interchangeable weight sets. `ModelLocator` reads
`model_manifest.json` and does not care which one is installed.

| | Verification (`tools/make_tiny_model.py`) | Production (`tools/export_model.py`) |
| --- | --- | --- |
| Size | 8.2 MB | ~600 MB (int8) |
| Graphs | text encoder, UNet, VAE decoder | same three |
| Weights | synthetic, untrained | SD 1.5 + LCM-LoRA |
| Output | abstract colour fields | photoreal / artistic |
| Ships in the APK | yes | you build it |

### Why a verification model exists

A pipeline you cannot run is a pipeline you cannot trust. The tiny model exercises every code
path — tokenisation, conditioning, the LCM loop with classifier-free guidance, VAE decode, the
memory guard, PNG persistence — in about two seconds per image, on any device, with an 8 MB
install. It makes "it generates nothing" a diagnosable failure rather than an ambiguous one.

It is built by hand with `onnx.helper` (no PyTorch, no downloads except the real CLIP tokenizer),
with shapes chosen to match the Kotlin engine exactly:

| Graph | Input | Output |
| --- | --- | --- |
| text_encoder | `input_ids` [1,77] int64 | `last_hidden_state` [1,77,768] |
| unet | `sample` [1,4,H/8,W/8], `timestep` [1], `encoder_hidden_states` [1,77,768] | `out_sample` [1,4,H/8,W/8] |
| vae_decoder | `latent_sample` [1,4,H/8,W/8] | `sample` [1,3,H,W] |

Three details that make untrained weights produce a presentable image rather than RGB static,
all documented in the script: padding tokens are masked before pooling (otherwise every prompt
conditions identically), the decoder pools the latent 8x before decoding (untrained weights have
no low-frequency bias), and the decoder normalises its own output (otherwise contrast and colour
cast are arbitrary).

## What the pipeline runs

| Component | Source | Precision in app |
| --- | --- | --- |
| Text encoder | CLIP ViT-L/14 text tower from SD 1.5 | int8 (dynamic) |
| UNet | SD 1.5 + LCM-LoRA (4-step distillation) | int8 (dynamic) |
| VAE decoder | SD 1.5 | fp16 or fp32 (not quantised — quality matters most here) |
| Tokenizer | `vocab.json` + `merges.txt` (CLIP) | text |

LCM is what makes phone inference plausible: 4 steps instead of 25–50. A 512×512 image is
~700 M MACs per UNet step at int8, which a modern arm64 core can chew through in seconds rather
than minutes.

## Producing the weights

```bash
pip install "torch>=2.2" "diffusers>=0.27" "transformers" onnx onnxruntime optimum[exporters]
python tools/export_model.py --out modelpack/src/main/assets/models
```

`tools/export_model.py` does the whole job:

1. downloads `runwayml/stable-diffusion-v1-5` and `latent-consistency/lcm-lora-sdv1-5`
   (desktop only — the phone never downloads anything);
2. fuses the LoRA into the UNet;
3. exports `text_encoder.onnx`, `unet.onnx`, `vae_decoder.onnx` with fixed 512-class shapes;
4. applies dynamic int8 quantisation to the UNet and text encoder;
5. copies the CLIP tokenizer files;
6. writes `model_manifest.json` with file sizes and SHA-256 checksums.

The manifest is what the app reads. It is data, not code: a different export (fp16, different
tensor names, a different sample size) works as long as the manifest describes it.

## Where the weights live

`modelpack/` is a Play Asset Delivery pack with `deliveryType = "install-time"`. Install-time
means the pack is delivered **with the installation**, not fetched on first launch — there is no
`AssetPackManager` download, no user prompt, no network.

On first run `ModelLocator` copies the graphs from the pack into `filesDir/models` because ONNX
Runtime needs a real file path. That is a local file copy. It costs disk (the weights exist
twice: pack + private copy) and it is the reason Settings reports both "weights on disk" and
total app data.

If you need to avoid the duplicate copy, ship a single-variant APK with the weights in `assets/`
instead of using the asset pack — the locator handles both, it just looks for
`assets/models/`.

## Licensing — read before you redistribute

* **Stable Diffusion 1.5** — CreativeML OpenRAIL-M. Commercial use is permitted, but the
  license carries **use-based restrictions** (no unlawful content, no deceptive deepfakes, and
  you must propagate the restrictions to downstream users). It is not an open-source license in
  the OSI sense.
* **LCM-LoRA (SD v1.5)** — released by the Latent Consistency team under MIT terms; check the
  model card for the exact wording of the version you pull.
* **CLIP tokenizer files** — MIT (OpenAI CLIP).

If you publish a build with weights inside, include the OpenRAIL-M text and the restrictions
with your distribution. This repository does **not** include weights, so the MIT license on the
application code does not cover them.

## Honest size expectations

| Precision | Approx. weights |
| --- | --- |
| int8 UNet + int8 text encoder + fp32 VAE | ~0.6 GB |
| fp16 everything | ~1.7 GB |
| int8 + fp16 VAE | ~0.9 GB |

There is no configuration in which a real diffusion model is small. A 25 MB app that claims
offline photorealistic generation either ships weights separately (i.e. downloads them) or is
not running a diffusion model.
