# Prebuilt artifact

`pixelmind-ai-1.1.0-arm64.apk` — release variant, arm64-v8a, signed with the debug keystore so it
installs without setup.

**This build generates images out of the box.** It bundles the 8.2 MB *verification* model: real
ONNX graphs with the same tensor contract as Stable Diffusion, but untrained weights. Output is
smooth abstract colour fields — proof that the whole on-device path runs, not a photo generator.

For photoreal output, export the production weights and rebuild:

```bash
python tools/export_model.py --out app/src/main/assets/models
./gradlew :app:assembleRelease
```

Built with `isMinifyEnabled = false` — R8 needs more time than the build machine's per-command
ceiling allowed. The project defaults to minification on.

Verify the download:

```bash
sha256sum -c SHA256SUMS.txt
```
