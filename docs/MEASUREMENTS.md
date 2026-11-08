# Measurements

## Build host

| | |
| --- | --- |
| CPU | 2 vCPU x86_64 |
| RAM | 2 GB |
| OS | Debian 12 |
| JDK | Temurin 17.0.21 |
| Gradle | 8.7 |
| AGP | 8.5.2 |
| Compile SDK | 35, build-tools 35.0.0 |

This is a build machine, not a phone. It is only used to measure *artifacts* (APK size,
composition, build times). Nothing about runtime performance on Android can be concluded from it.

## APK sizes — measured

| Artifact | Compressed | Uncompressed |
| --- | --- | --- |
| `app-release.apk` (with 8.2 MB verification model, minify off) | 29.0 MB (30,387,428 bytes) | 41.1 MB |
| `app-release.apk` (no weights, earlier build) | 22.2 MB | 32.9 MB |
| `app-debug.apk` (no weights) | 25.0 MB | — |

Composition of the release APK (uncompressed):

| Component | Size |
| --- | --- |
| Native libraries (`libonnxruntime.so`, `libonnxruntime4j_jni.so`) | 16.3 MB |
| DEX | 16.2 MB |
| Resources + manifest | 0.44 MB |
| Assets | 0.01 MB |

ONNX Runtime's native library is the single largest object in the app and is not something app
code can shrink — it *is* the inference engine.

Expected release size with R8 enabled (not built here, see below): the DEX portion typically
drops 45–60% for a Compose app of this shape, putting the APK around 12–14 MB before weights.

## Build times — measured

| Task | Time |
| --- | --- |
| `:app:compileDebugKotlin` (cold) | 4m 32s |
| `:app:assembleDebug` (after compile) | 2m 50s |
| `:app:testDebugUnitTest` | 4m 02s |
| `:app:mergeExtDexRelease` | 3m 43s |
| `:app:assembleRelease` (no R8) | 2m 22s |

## Deliberate non-measurements

| Decision | What was and was not established |
| --- | --- |
| `System.gc()` removed from the end of `OnnxDiffusionEngine.generate` (Day 1, item 7) | Not measured: no device and no weights to run, so there is no before/after number for peak RSS or time-to-next-generation. It was removed because the call forced a full stop-the-world collection after every generation on the same thread pool that runs the next one, and because every large buffer it was reaching for (latents, text embedding, decoded bitmap) is already unreachable at that point. If a device measurement later shows a genuine memory plateau, this entry is the place to record the counter-evidence. |
| Release APK built with R8 disabled | Measured: 23,287,301 bytes (22.2 MB) with `isMinifyEnabled = false` via a machine-local init script. Not measured: the R8-minified size on this host — R8 needs more than five minutes here (two cores, 2 GB), and a single command on this build pod is capped at 300 s. `BUILD.md` documents the same constraint for the artifact in `release/`. |

## Memory — what is measured and what is only estimated

Day 3 of the backlog (items 21–30) is the memory sweep. Nothing on this page about a phone's memory
is measured: there is no device here and no SD weights to run on one. What follows is what the app
*plans* against, and what changed about the plan.

### Per-pass working set — computed, not measured

One generation allocates five float arrays of `latentChannels x latentH x latentW` and reuses them
for every step (`DiffusionBuffers`). At 512x512 the latent is 4 x 64 x 64:

| Resolution | Buffers | Per buffer | Working set |
| --- | --- | --- | --- |
| 512x512 | 5 | 64 KB | 320 KB |
| 640x384 | 5 | 48 KB | 240 KB |
| 384x640 | 5 | 48 KB | 240 KB |

Before this sweep two of those five (the conditional and unconditional UNet outputs) were reallocated
on every step, so a four-step generation with guidance on churned about 512 KB of short-lived arrays
instead of holding 320 KB. Both the size and the reuse are asserted in `DiffusionBuffersTest`.

### Peak resident memory estimate, 512x512, by quality tier

These are the numbers `MemoryGuard` budgets against, computed by `MemoryEstimate.peakMb` and pinned by
`MemoryEstimateTest`. They are **estimates from a formula, not measurements**:

| Quality tier | Steps | Peak estimate, 640 MB weights | Peak estimate, 8 MB verification model |
| --- | --- | --- | --- |
| Fast | 4 | 1008 MB | 121 MB |
| Balanced | 6 | 1012 MB | 125 MB |
| Quality | 8 | 1016 MB | 129 MB |

The formula is `weights + megapixels x 180 x complexity + megapixels x 120 x complexity + 60 + 2 x
steps`, where `megapixels = w x h / 262144` and `complexity = clamp(weightsMb / 640, 0.15, 1.0)`.
Non-square aspects are cheaper at the same step count: 640x384 Balanced estimates 993 MB and 448x576
Balanced 1007 MB.

`MemoryGuard.fitToDevice` plans against at most 60% of free RAM and at most half of physical RAM, so
a device whose *free* memory is below the estimate is downgraded (steps first, then resolution) rather
than being allowed to try. Whether the estimate is close to reality on a phone is **unverified**.

### Decode cache budget

`ImageStore`'s LRU cache holds decoded previews. It used to be sized as `Runtime.maxMemory() / 8` —
a fraction of the *Java* heap, which has not contained bitmap pixels since API 26, with no absolute
cap. On a 512 MB-heap-class device with 1 GB of RAM that allowed 64 MB of native bitmap memory.

| Device | Budget |
| --- | --- |
| 256 MB class, 2 GB RAM | 21 MB (min of heap-class/8 and RAM/96) |
| 512 MB class, 1 GB RAM | 10 MB |
| 128 MB class, 1 GB RAM, low-RAM | 5 MB |
| any | clamped to 4–24 MB |

The rule is `min(memoryClassKb / 8, totalRamKb / 96)`, halved on a low-RAM device, clamped to
[4 MB, 24 MB]; see `PreviewCachePolicy` and `PreviewCachePolicyTest`. Cache keys record the decoded
sample size, so a thumbnail and a full-size preview of the same file no longer evict or impersonate
one another. None of this was exercised on a device: the cache is not unit-testable on the JVM
(`android.util.LruCache` is a stub there), so the policy is tested and the wiring is inspection-only.

### Still not measured

Peak RSS for a real generation at any tier, and whether 20 cancelled generations leave the process
where it started. `DiffusionBuffersTest` proves that 20 create/release cycles leave no working set
alive in the JVM's sense — it cannot observe a device's RSS, the collector's timing, or the native
allocator.

## Not measured

| Metric | Why |
| --- | --- |
| Installed size on device | no device |
| Peak RAM during generation | no device, and no weights to run |
| Generation time per step / per image | same |
| Battery impact | same |
| APK size with production (SD) weights | not exportable here (needs ~16 GB RAM and a multi-GB download) |
| On-device generation time for the verification model | no device; measured 0.7–0.9 s per 512×512 image on a 2-core x86 CPU, which says nothing about arm64 phones |

Anyone quoting these numbers for this build would be inventing them.
