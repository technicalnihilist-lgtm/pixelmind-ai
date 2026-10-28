# Supported devices

## Minimum

| Requirement | Value |
| --- | --- |
| Android | 8.0 (API 26) |
| ABI | arm64-v8a |
| Free storage | ~1.5 GB (weights + private copy + generated images) |
| RAM | 6 GB recommended, 4 GB workable at reduced resolution |

`minSdk 26` is a real floor, not a formality: `Bitmap.Config` handling, scoped-storage-free file
layout assumptions and the ONNX Runtime Android build all target it.

Only `arm64-v8a` is packaged. Shipping `armeabi-v7a` would add ~16 MB of native libraries for
devices that cannot realistically run the model anyway, and 32-bit address spaces cannot map
int8 weights of this size.

## Memory behaviour

`MemoryGuard` reads the ActivityManager heap class and degrades the request before inference:

| Heap class | Behaviour |
| --- | --- |
| ≥ 512 MB | Full resolution (up to 640×640), requested steps |
| 256–511 MB | Steps reduced first, then resolution, until the estimate fits |
| < 256 MB | Likely refuses with a clear message rather than being killed mid-run |

The estimate assumes ~640 MB of resident weights plus activations proportional to pixel count.
It is deliberately conservative; a device that *could* squeeze a run in may still be downgraded.

## Acceleration

At startup the engine tries, in order:

1. **NNAPI** — used when the device's driver accepts the graphs.
2. **XNNPACK** — the CPU fallback with optimised kernels.
3. **Plain CPU** — always works.

The accelerator actually in use is shown on Home and in Settings, read from the live session
rather than assumed.

## Devices not supported

* x86/x86_64 emulators (no arm64 native libraries; the emulator can translate, but expect
  minutes per step and unreliable results).
* Anything without 64-bit support.
* Devices where the vendor's NNAPI driver crashes on large graphs — the engine catches the
  failure and falls back to XNNPACK, but this path has not been exercised on real hardware in
  this build (see the status table in the README).
