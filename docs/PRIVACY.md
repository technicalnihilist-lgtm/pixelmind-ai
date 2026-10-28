# Privacy

## The one-line version

The app cannot reach the network, because Android does not give it permission to.

## What is declared

`app/src/main/AndroidManifest.xml` contains no `<uses-permission>` elements. The merged manifest
of the built APK contains exactly one permission entry, added by AndroidX:

```
ai.pixelmind.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION
```

That is an app-scoped signature permission used to mark dynamically registered broadcast
receivers as not-exported. It grants no platform capability and is not requestable by any other
app. Everything else — storage, location, network — is absent.

Verify it yourself on the shipped APK:

```bash
aapt2 dump permissions app-release.apk
```

## What that means in practice

* No `INTERNET` permission → the process cannot open a socket. This is enforced by the kernel
  (the `inet` group), not by good intentions in code. Even a compromised library inside the app
  cannot exfiltrate a prompt.
* No analytics SDK. No crash reporter. No advertising ID. No account.
* Prompts are stored locally in SQLite; images are PNG files in app-private storage.
* `android:allowBackup="false"` plus `data_extraction_rules.xml` excluding every domain from
  both cloud backup and device transfer — so prompts and images are not copied to Google's
  servers by the OS on the user's behalf.

## What leaves the device

Nothing, by the app. Two deliberate exceptions, both initiated by the user:

1. **Share** — hands a file URI to whichever app the user picks in the system chooser. What that
   app does with it is outside PixelMind's control, and the UI says so.
2. **Save to Pictures** — copies a PNG into the public `Pictures/PixelMind` collection via
   MediaStore (API 29+, no permission needed). From there other apps can see it.

On API 26–28 the app cannot write to shared storage without `WRITE_EXTERNAL_STORAGE`, which it
does not request; the gallery offers Share instead of Save on those versions.

## Verifying offline operation

See [VERIFICATION.md](VERIFICATION.md) for the exact procedure, and note the honest status
recorded there: it has **not** been run on physical hardware for this build.
