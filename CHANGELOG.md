# Changelog

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
