# Offline verification protocol

> **Status for this build: NOT EXECUTED.**
> No Android device or emulator with arm64 support was available to the environment that produced
> this repository, and the shipped APK contains no model weights. The protocol below is what must
> be run — and reported — before anyone claims offline generation works. Do not mark these boxes
> without doing them.

## Preconditions

- A physical arm64 Android device, API 29+, ≥ 6 GB RAM.
- A build with weights bundled (`python tools/export_model.py` then `bundleRelease`, installed via
  `bundletool` or Play internal testing — install-time packs are not delivered by a plain
  `adb install` of a single APK).

## Procedure

1. **Install with no network.** Enable airplane mode on the phone *before* installing. Install
   the bundle via `bundletool install-apks --apks=...`.
2. **Confirm the pack arrived.** Launch the app. Home must show `Local AI · <accelerator>`, and
   Settings → Local model must show the model name, precision and a non-zero weights size.
   If it shows *Weights missing*, the pack did not install — that is a delivery failure, not a
   pass.
3. **Confirm no connectivity.** In airplane mode, `adb shell dumpsys package ai.pixelmind | grep -i
   permission` must list no `INTERNET`. `adb shell "run-as ai.pixelmind ping -c1 8.8.8.8"` must
   fail.
4. **Generate.** Studio → prompt "a brass astrolabe on a mossy rock, morning light" → Generate.
   Expect: progress ring advancing through *Reading your prompt → Diffusing → Decoding image*,
   then a PNG. Record wall-clock time and step count.
5. **Confirm the image is new.** Re-run with the same seed — the output must be identical.
   Re-run with a different seed — it must differ. A stock image or a cached file fails this test.
6. **Kill and relaunch in airplane mode.** Generate again. It must still work: nothing was
   downloaded in step 4 that the app depends on.
7. **Measure.** `adb shell dumpsys meminfo ai.pixelmind` peak PSS during generation; app install
   size from `adb shell pm path ai.pixelmind` + `du`; generation time per resolution.
8. **Record** all of the above in `docs/MEASUREMENTS.md`, including device model and OS build.

## What has actually been verified for this build

| Check | Result |
| --- | --- |
| APK builds (debug + release) | pass |
| Unit tests | 13/13 pass |
| No platform permissions in built APK | pass (`aapt2 dump permissions`) |
| Engine reports `WeightsMissing` when weights are absent, without faking output | pass (code path + instrumentation test written) |
| Generation on device | **not run** |
| Airplane-mode generation | **not run** |
| Peak RAM / install size on device | **not measured** |
