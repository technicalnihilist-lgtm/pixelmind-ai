# Changelog

## 1.2.0 — concurrency (Day 2 of the optimisation backlog)

Maintenance release. No UI, feature, model-format or permission changes; the app still declares no
platform permissions and no `INTERNET`.

* **Concurrent enqueue no longer corrupts the queue.** `nextId++` on a plain `Long` could hand two
  callers the same id, and the job-list append was a lost update, which could leave an id in the work
  channel with no job record behind it — the drain loop then skipped that job silently. Id
  assignment, the list append and the channel send now happen under one lock, and the ids are
  allocated atomically.
* **A retried job cannot run twice.** A cancelled queued job could be re-sent to the work channel
  while its original id was still buffered there. A pending set now keeps at most one copy of an id
  in the channel.
* **Cancel actually cancels.** The job was marked RUNNING and the worker handle was stored in two
  separate steps, so a cancel landing in between found nothing to cancel and the generation ran to
  completion under a "cancelled" label. The worker is created lazily and published with the state
  change in one critical section.
* **Terminal states are final.** A cancelled job is no longer relabelled by the result that arrives
  while the engine unwinds.
* **Progress cannot reopen after a job finished.** Late callbacks are dropped.
* **Results are no longer dropped when nobody is collecting.** The result channel was buffered at 64
  and written with `trySend`, so a burst of queued jobs lost the tail of its events.
* **Releasing the engine no longer blocks a thread.** The engine's monitor became a `Mutex` and
  `LocalAIEngine.release()` is now `suspend`; a release during a generation suspends instead of
  parking the caller (which was the main thread, from `ProcessLifecycleOwner.onStop`).
* **I/O moved off the compute pool.** The PNG encode and write run on `Dispatchers.IO`, and
  `ModelImporter.discardPrevious()` no longer deletes a model directory on the main thread.
* **The status pill cannot flicker back to Loading** over a model that is already in memory.

Tests: 48 (34 existing + `GenerationQueueTest`, `EngineStatusReducerTest`). See
`docs/CONCURRENCY.md` for what each item turned out to be and for what is still unmeasured.

## 1.1.7 — crash and lifecycle sweep (Day 1 of the optimisation backlog)

Maintenance release. No UI, feature, model-format or permission changes; the app still declares no
platform permissions and no `INTERNET`.

* **The engine no longer throws out of its own initialiser.** Seven `requireNotNull` call sites in
  `OnnxDiffusionEngine` became explicit `EngineStatus.Failed` states, so a half-built session set is
  reported as a failure with a sentence instead of an `IllegalArgumentException` from inside
  `initialize()`.
* **Failures are described, not echoed.** A new `describeFailure` maps an `OutOfMemoryError`, a
  `SecurityException` and a missing model file to actionable text; an out-of-memory kill used to
  reach the failure card as the bare string "OutOfMemoryError".
* **Out of memory is no longer treated as an accelerator fallback.** It is logged at error level and
  stops the XNNPACK → NNAPI → CPU chain instead of being retried twice more.
* **`System.gc()` after every generation is gone.** The reasoning (and the fact that it is
  unmeasured on a device) is recorded in `docs/MEASUREMENTS.md`.
* **A process death cannot leave a half-written PNG in the gallery.** `ImageStore.save` encodes into
  a `*.part` staging file and renames it into place; abandoned staging files are swept on the next
  save and excluded from the storage totals.
* **Releasing the engine is idempotent.** A repeated `onStop` no longer queues a second
  `engine.release()` behind the engine lock, and a generation that was in flight when the app went
  to the background is recorded as cancelled rather than left running or blamed on the prompt.
* **A queued job no longer fails with a false message.** It reloads the model first, instead of
  reporting "open the app again" while the app is open.

Tests: 34 (25 existing + `ProvisioningTest`, `FailureDescriptionTest`).

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
