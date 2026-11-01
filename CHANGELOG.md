# Changelog

## 1.1.6 — stability pass: engine lifecycle, model repair, import integrity

A maintenance release. No UI, feature, model-format or permission changes — the app still
declares no platform permissions and cannot open a socket.

**Engine lifecycle (crashes and stalls)**

* **Loading the model is now serialised against releasing it.** A background transition that
  arrived while the sessions were being built could leave one set of ONNX sessions open and
  unreachable, with the status flipping to *Ready* afterwards over closed handles. The next
  Generate then failed with a confusing message.
* **Releasing the engine no longer runs on the main thread.** `engine.release()` waits for an
  in-flight generation to reach its next cancellation check — a whole diffusion step — and the
  caller is `onStop()`. A long step could therefore freeze the UI on the way to the background.
* **A cancelled generation no longer wedges the queue.** The active job was only cleared inside
  the job's own coroutine, so a task cancelled before it started left `activeJob` set for the rest
  of the process lifetime — and `reloadEngine()` silently refuses to run while a job is active,
  which made "Restore bundled model" and post-import reloads do nothing after a cancel.
* **A full result buffer no longer stalls the queue.** Results went out through a suspending
  `send()`; with no collector attached the 64-deep buffer could fill and block the worker forever.
* `retry` no longer enqueues a job that is already queued, which ran the same request twice.

**Model files (startup, I/O, recovery)**

* **A damaged weight file is now repaired instead of being fatal.** A truncated or half-copied
  component made `resolve()` report *Corrupt* on every launch until the user cleared app data,
  even though the asset pack still held a good copy. The bundled set is now re-provisioned once
  and re-verified. An *imported* model that fails its own checksums is left untouched — silently
  swapping a user's weights for the bundled ones would be worse than saying the file is broken.
* **Components are not re-hashed on every load.** Verifying a 1 GB weight set costs seconds and
  used to happen on every engine load, including every return from the background (sessions are
  released on stop). Digests are cached against (length, last-modified); any change forces a
  re-read. The cache is invalidated whenever weights are provisioned or imported.
* A failed `.part` copy no longer leaves the staging file behind in `filesDir`.

**Model import**

* **Checksums are verified before the swap, not after.** A truncated graph used to install, fail
  to load, and roll back with a vaguer message. It is now rejected up front, naming the file.
* **The 4 GB cap is enforced while copying.** The old check ran after an entry had been fully
  written, so one oversized zip entry could fill the device before being refused.
* Selecting individual files that share a name (`unet/model.onnx` and `text_encoder/model.onnx`
  both flatten to `model.onnx`) now fails with a message that says to import a zip, instead of
  silently keeping only the last one.

**Memory and correctness**

* A PNG that failed to encode no longer strands a full-resolution bitmap.
* A VAE whose output does not match the 8× latent grid now reports that, instead of crashing with
  an index-out-of-bounds deep inside the decode loop.
* A failed export to Pictures no longer leaves an invisible `IS_PENDING` row in the gallery.
* Two generations with the same seed in the same millisecond no longer overwrite one image file.
* The UNet's VAE-scaled copy of the latent is a reused buffer rather than a fresh array per call.
* The CLIP pre-tokeniser regex is compiled once instead of on every encode, and the BPE cache is
  bounded and thread-safe.
* `LcmScheduler.timesteps(0)` no longer divides by zero.

**Tests** — 25 unit tests, up from 13. New coverage for zip entry sanitising (traversal, absolute
paths, drive letters), component detection, generated-manifest round-trip and digests, schedule
validity across every step count the app offers, and tokenizer truncation.

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
