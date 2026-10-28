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
