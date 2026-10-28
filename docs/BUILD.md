# Building PixelMind AI

## Requirements

| Tool | Version |
| --- | --- |
| JDK | 17 |
| Android SDK platform | 35 |
| Build tools | 35.0.0 |
| Gradle | 8.7 (wrapper included) |
| NDK | only for the optional native scheduler (see below) |

## Quick build

```bash
./gradlew :app:assembleDebug      # debug APK
./gradlew :app:testDebugUnitTest  # 13 unit tests
./gradlew :app:assembleRelease    # release APK (R8 enabled)
./gradlew :app:bundleRelease      # AAB — required for Play Asset Delivery
```

`bundleRelease` is the one that matters for distribution: an install-time asset pack is only
delivered as part of an App Bundle. A plain APK build leaves the pack out, which is exactly what
you want for a weightless test build.

## About the release APK in `release/`

The APK attached to this repository was built with `isMinifyEnabled = false`, because R8 needs
more than the six-minute-per-command ceiling of the build machine used here (see
[MEASUREMENTS.md](MEASUREMENTS.md)). The project's `app/build.gradle.kts` has minification
**enabled**; building it on a normal workstation produces a smaller APK. Nothing else differs —
same code, same resources, same signature config.

The release build is signed with the debug keystore so that it installs without extra setup.
For anything real, create your own signing config:

```kotlin
signingConfigs {
    create("release") {
        storeFile = file(System.getenv("PIXELMIND_KEYSTORE") ?: "keystore.jks")
        storePassword = System.getenv("PIXELMIND_STORE_PASSWORD")
        keyAlias = System.getenv("PIXELMIND_KEY_ALIAS")
        keyPassword = System.getenv("PIXELMIND_KEY_PASSWORD")
    }
}
```

Never commit a keystore or its passwords.

## Optional native scheduler

The C++ kernel in `app/src/main/cpp` is a NEON version of the LCM update step. It is not part of
the default build — the Kotlin implementation produces identical results — so the project builds
with no NDK installed.

To enable it:

1. Install the NDK: `sdkmanager "ndk;26.1.10909125"`.
2. In `app/build.gradle.kts`, inside `android { }`, add:

   ```kotlin
   ndkVersion = "26.1.10909125"
   externalNativeBuild {
       cmake {
           path = file("src/main/cpp/CMakeLists.txt")
           version = "3.22.1"
       }
   }
   ```

3. Rebuild. `NativeScheduler` will find `libpixelmind.so` and use it; if the library is missing
   for any reason it falls back to Kotlin at runtime.

## Troubleshooting

* **"Weights missing" on the home screen** — expected on a build without
  `modelpack/src/main/assets/models/*.onnx`. Run `tools/export_model.py`.
* **Build fails on `minSdk` / manifest merger** — you are probably on an old AGP; the project
  pins 8.5.2.
* **Gradle can't resolve dependencies behind a corporate proxy** — set
  `systemProp.https.proxyHost` etc. in `~/.gradle/gradle.properties`, and add the proxy's CA to a
  truststore passed via `-Djavax.net.ssl.trustStore` if it re-signs TLS.
