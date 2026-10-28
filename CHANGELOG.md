# Changelog

## 1.1.5 — import a model from device storage

* **Settings → Import from storage.** Pick a `.zip` or individual files with the system file
  picker and use them immediately. The app still declares **no permissions**: the Storage Access
  Framework grants access to one document at a time.
* A `model_manifest.json` is optional. If the archive has `unet`, `text_encoder` and `vae_decoder`
  graphs plus CLIP `vocab.json`/`merges.txt`, a manifest is written for you.
* Import is transactional: files are staged, validated, then swapped in. **If the imported model
  fails to load, the previous model is restored automatically** — the app never ends up holding
  weights it cannot run.
* Zip entries are sanitised against path traversal (`../`, absolute paths, drive letters) and the
  archive is capped at 4 GB.
* "Restore bundled model" removes the import and re-provisions the shipped one.

Also restored two files that an earlier push had reverted: `MemoryGuard` (device-RAM budget) and
`ModelLocator` (nested-directory provisioning). Both fixes are in the shipped 1.1.2+ APKs but had
drifted on `main`, which would have broken a fresh clone.


## 1.1.4 — timings on screen, clearer model versioning

* **Generation timing is now shown in the app.** Studio reports where the wall clock went —
  prompt encoding, per-step diffusion, VAE decode, and which accelerator ran it. No `adb` needed.
* Settings distinguishes **"Model build"** (the manifest's own version, 0.1.0 for the verification
  model) from **"App version"**. Labelling the model's version simply as "Version" made it look
  like an out-of-date app.
* Settings also names the model type: *Verification (untrained weights)* or *Production*.


## 1.1.3 — generation no longer stalls

Reported from a device: progress froze at 5% for over a minute. 5% is the text-encoding stage, so
the stall was before the diffusion loop even started.

* **XNNPACK is now the first delegate, not NNAPI.** NNAPI recompiles its model for every new input
  shape and pays a per-node dispatch cost; on a graph containing Gather, ReduceMean and Resize
  that turned a sub-second pass into minutes. NNAPI remains as a fallback.
* **Warm-up at load time.** Each graph runs once with tiny inputs right after session creation, so
  first-use compilation is charged to "Loading engine" instead of to the user's first Generate.
* **Per-stage timing in logcat** (`OnnxDiffusionEngine`): text encoding, every diffusion step, and
  VAE decode now log their durations, so a slow device can be diagnosed from the log alone.


## 1.1.2 — provisioning fix

Fixes the failure seen on a real device:

    Could not provision weights: /data/user/0/ai.pixelmind/files/models/clip/vocab.json:
    open failed: ENOENT (No such file or directory)

* `ModelLocator` now creates the containing directory before writing each asset. Manifest entries
  are relative paths (`clip/vocab.json`) and the `clip/` directory was never created, so the copy
  aborted before the manifest was written and every launch retried the same failure.
* Writes go through a `.part` file and are renamed into place, so an interrupted copy cannot leave
  a truncated graph behind.
* `MemoryGuard` no longer budgets against `ActivityManager.memoryClass` — that is the Java heap
  cap, and ONNX Runtime allocates natively, outside it. The budget now comes from the device's
  actual free RAM.
* `MemoryGuard` no longer assumes 640 MB of weights. The real figure is passed in from the model
  manifest, so the 8 MB verification model is not treated like a 600 MB UNet.
* Regression tests for nested-path provisioning (`ModelProvisioningTest`): 16 unit tests, all pass.


## 1.1.1 — model survey

* `docs/MODEL_OPTIONS.md`: every pre-exported ONNX text-to-image model that fits this app's tensor
  contract, with measured sizes and load-test results. Documents why a 300 MB diffusion model does
  not exist and why the one 618 MB export is unusable.
* `tools/repair_onnx.py`: repairs duplicate tensor definitions in quantised ONNX exports (50 of 50
  found in `szFangshi/tiny-sd-int8`), and reports the deeper defect that makes that particular
  model unrecoverable.
* No app code changes; the shipped APK is unchanged.

## 1.1.0

* Ships a runnable **verification model** (8.2 MB) so the APK generates images out of the box:
  `tools/make_tiny_model.py` builds the three ONNX graphs by hand with `onnx.helper`.
* `tools/smoke_test_model.py` runs the entire pipeline in Python against any model directory,
  mirroring the Kotlin engine step for step.
* Engine now falls back NNAPI -> XNNPACK -> CPU on *session creation*, not just on option
  creation, so a driver that accepts the options but rejects the graph no longer fails init.
* README/docs updated: the APK no longer reports "weights missing"; production SD weights remain
  a build step.

## 1.0.0 — initial

* Local ONNX inference pipeline: CLIP BPE tokenizer, LCM scheduler, UNet + VAE graph execution.
* `LocalAIEngine` abstraction with `OnnxDiffusionEngine` implementation; UI never touches runtime.
* `GenerationManager`: serialised queue, progress reporting, cancellation, engine release on
  background.
* Liquid Glass design system and four screens (Home, Studio, Gallery, Settings).
* Play Asset Delivery install-time pack for bundled weights.
* SQLite history, favourites, prompt history, MediaStore export.
* 13 unit tests, 2 instrumentation suites.
* No platform permissions; no networking code.
