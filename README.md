# PixelMind AI

An Android text-to-image app that runs its diffusion model entirely on the device: no API keys,
no server, no analytics, and no `INTERNET` permission in the manifest.

Kotlin + Jetpack Compose UI, ONNX Runtime as the local inference layer, and a Liquid Glass
design system.

**Download the APK:** [pixelmind-ai-1.1.6-arm64.apk](https://github.com/technicalnihilist-lgtm/pixelmind-ai/releases/download/v1.1.6/pixelmind-ai-1.1.6-arm64.apk)
· 29 MB · arm64-v8a · Android 8.0+ · SHA-256 `755a20edd6727f7413f7cd6f8d29cfa5210d193f8a400dda5029726beb00178c`

Screens: [Home](design/01-home.png) · [Studio](design/02-studio.png) · [Gallery](design/03-gallery.png) · [Settings](design/04-settings.png) · [sample output](design/05-sample-output.png)

---

## Status — read this first

This repository is an honest snapshot, not a launch announcement. Here is exactly what is
verified and what is not.

| Claim | State |
| --- | --- |
| Project compiles, APKs build | **Verified.** `assembleDebug` and `assembleRelease` both succeed. |
| Unit tests pass | **Verified.** 13/13 pass (`./gradlew :app:testDebugUnitTest`). |
| Manifest requests no platform permissions | **Verified** on the built APK with `aapt2 dump permissions`. |
| Real ONNX inference code (no mocks, no stock images) | **Implemented** — CLIP BPE tokenizer, LCM scheduler, UNet/VAE graph execution. |
| Weights bundled in the APK | **Yes** — the *verification* model (8.2 MB): real ONNX graphs with the same shapes as the SD pipeline, untrained weights. |
| Image generation works in the shipped APK | **Yes, end to end** — tokenizer → conditioning → 4 LCM steps → VAE → PNG, on device, offline. Output is smooth abstract colour fields, **not** photoreal imagery. |
| Photoreal / SD-quality output | **No.** Needs the Stable Diffusion export: `tools/export_model.py`. No public pre-exported model below ~950 MB is usable — see [docs/MODEL_OPTIONS.md](docs/MODEL_OPTIONS.md). |
| Tested on a physical device | **Not done.** No Android hardware was available to this build. |
| Airplane-mode generation verified | **Not done** (depends on the two rows above). |
| Screenshots | The images in `design/` are **UI renders produced from the Compose layout**, not device screenshots. |

Anything below marked "measured" was measured on the build machine described in
[docs/MEASUREMENTS.md](docs/MEASUREMENTS.md). Nothing is estimated without saying so.

---

## Two models, one pipeline

| | Verification model (shipped) | Production model (opt-in) |
| --- | --- | --- |
| Weights | 8.2 MB, synthetic | ~600 MB, SD 1.5 + LCM-LoRA |
| Build | `tools/make_tiny_model.py` | `tools/export_model.py` |
| Output | smooth abstract colour fields | photoreal / artistic images |
| Purpose | proves the on-device path runs | the actual product |

The shipped APK generates images today. It does not pretend they are photographs: the weights are
untrained, so what comes out is prompt-dependent abstract colour — honest output from a real
pipeline rather than a stock image presented as AI.

Every pre-exported ONNX alternative was surveyed, measured and load-tested; the results (including
one widely-mirrored "int8" export that is structurally broken) are in
**[docs/MODEL_OPTIONS.md](docs/MODEL_OPTIONS.md)**. Short version: nothing usable exists below
~950 MB, and a 300 MB diffusion model does not exist at all.

Swapping in the real model is a build step, not a code change:

```bash
python tools/export_model.py --out app/src/main/assets/models   # ~20-40 min, needs ~16 GB RAM
python tools/verify_export.py app/src/main/assets/models
./gradlew :app:assembleRelease && adb install -r app/build/outputs/apk/release/app-release.apk
```

Full walkthrough, including the asset-pack trap that makes a correct export look broken:
**[docs/GETTING_IT_RUNNING.md](docs/GETTING_IT_RUNNING.md)**.

---

## Architecture

```
 ┌───────────────┐   prompt    ┌──────────────────────┐
 │  Compose UI   │────────────▶│  GenerationManager   │  queue · cancel · retry
 │  (screens)    │◀────────────│  (StateFlow)         │
 └───────────────┘  progress   └──────────┬───────────┘
                                          │ LocalAIEngine (interface)
                                          ▼
                               ┌──────────────────────┐
                               │  OnnxDiffusionEngine │
                               │  ┌────────────────┐  │
                               │  │ CLIP encoder   │  │
                               │  │ LCM UNet int8  │  │  ONNX Runtime
                               │  │ VAE decoder    │  │
                               │  └────────────────┘  │
                               └──────────┬───────────┘
                                          ▼
                               ┌──────────────────────┐
                               │  ImageStore → PNG    │
                               │  SQLite history      │
                               └──────────────────────┘
```

The UI never touches ONNX Runtime. Everything the interface can see is `LocalAIEngine`
(`engine/EngineContract.kt`); the runtime, tensor layouts and quantisation stay behind it.

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Modules

| Path | What lives there |
| --- | --- |
| `app/src/main/java/ai/pixelmind/engine/` | Inference: tokenizer, scheduler, ONNX pipeline, memory guard, model provisioning |
| `app/src/main/java/ai/pixelmind/manager/` | `GenerationManager` — queue, progress, cancellation, engine lifecycle |
| `app/src/main/java/ai/pixelmind/data/` | SQLite history, favourites, prompt history, settings |
| `app/src/main/java/ai/pixelmind/ui/` | Compose screens, Liquid Glass design system, navigation |
| `app/src/main/cpp/` | Optional NEON JNI scheduler (see [docs/BUILD.md](docs/BUILD.md)) |
| `modelpack/` | Play Asset Delivery **install-time** pack that carries the weights |
| `tools/export_model.py` | Exports + quantises Stable Diffusion 1.5 + LCM into the ONNX graphs this app loads |
| `tools/make_tiny_model.py` | Builds the 8 MB verification model that ships in the APK |
| `tools/smoke_test_model.py` | Runs the whole pipeline in Python against any model directory |
| `tools/repair_onnx.py` | Diagnoses and repairs duplicate-tensor-definition bugs in third-party quantised exports |

## Build

```bash
# 1. produce weights — pick one:
python tools/make_tiny_model.py --out app/src/main/assets/models   # 8 MB verification model, ~10s
python tools/export_model.py   --out app/src/main/assets/models   # 600 MB production model, ~30min

# 2. build
./gradlew :app:assembleRelease
```

Weights are **not** committed: they are generated artifacts, and the 600 MB production export has
no business in git history. The verification model is reproducible from a fixed RNG seed in about
ten seconds, which is why a fresh clone builds to a working APK in two commands.

Without step 1 the app still builds and installs; it simply reports that the weights are
missing instead of pretending to generate. See [docs/BUILD.md](docs/BUILD.md).

## Measured artifacts

| Artifact | Size |
| --- | --- |
| `app-release.apk` (arm64-v8a, with the 8.2 MB verification model) | 29.0 MB (30,387,428 bytes) |
| `app-release.apk` (earlier build, no weights) | 22.2 MB (23,254,529 bytes) |
| ↳ native (ONNX Runtime) | 16.3 MB uncompressed |
| ↳ dex | 16.2 MB uncompressed |
| ↳ resources + manifest | 0.44 MB uncompressed |
| ↳ bundled verification weights | 8.2 MB uncompressed |

Swapping in the production model grows the install by roughly 0.6–1.2 GB depending on the
precision you export at. That is the honest cost of offline generation — no amount of app-code
optimisation changes it.

## Privacy

No `INTERNET` permission. No networking library. No analytics SDK. No account. Cloud backup and
device-transfer are disabled for every data domain. Details and how to verify:
[docs/PRIVACY.md](docs/PRIVACY.md).

## Model licensing

The pipeline targets **Stable Diffusion 1.5** (CreativeML OpenRAIL-M) with the **LCM-LoRA**
consistency distillation. Both carry use restrictions you must honour if you redistribute the
weights. Read [docs/MODEL.md](docs/MODEL.md) before shipping anything built from this code.

## License

Application code: MIT (see `LICENSE`). Model weights: not included, and not covered by that
license.

---

## ⭐ Like this work? Give it a star

If you like this app and our work, please **star this repository** ⭐ and **[follow technicalnihilist-lgtm on GitHub](https://github.com/technicalnihilist-lgtm)**. If our work isn't for you, that's completely fine — the choice is yours. Thank you for stopping by!

<details>
<summary><b>🌍 Same message in 32 languages (A–Z + Indian languages)</b></summary>

- **A — Arabic** (العربية): ⭐ إذا أعجبك هذا التطبيق وعملنا، فيرجى منح المستودع نجمة ومتابعتنا. وإذا لم يعجبك عملنا، فلا مشكلة على الإطلاق — القرار قرارك. شكرًا لزيارتك!
- **B — Bengali** (বাংলা): ⭐ যদি আপনার এই অ্যাপ এবং আমাদের কাজ ভালো লাগে, তবে অনুগ্রহ করে রিপোজিটরিটিকে একটি স্টার দিন এবং আমাদের ফলো করুন। আমাদের কাজ যদি আপনার পছন্দ না হয়, তাতেও কোনো সমস্যা নেই — সিদ্ধান্ত আপনার। আমাদের পেজে আসার জন্য ধন্যবাদ!
- **C — Chinese** (中文): ⭐ 如果你喜欢这个应用和我们的作品，请给仓库点个星标并关注我们。如果我们的作品不适合你，也完全没关系——选择权在你。感谢你的到访！
- **D — Dutch** (Nederlands): ⭐ Vind je deze app en ons werk leuk? Geef de repository dan een ster en volg ons. Vind je ons werk niets, dan is dat helemaal prima — de keuze is aan jou. Bedankt voor je bezoek!
- **E — English** (English): ⭐ If you like this app and our work, please give the repository a star and follow us. If our work isn't for you, that's completely fine — the choice is yours. Thank you for stopping by!
- **F — French** (Français): ⭐ Si cette application et notre travail vous plaisent, mettez une étoile au dépôt et suivez-nous. Si notre travail n'est pas pour vous, aucun souci — c'est votre choix. Merci de votre visite !
- **G — German** (Deutsch): ⭐ Wenn dir diese App und unsere Arbeit gefallen, gib dem Repository einen Stern und folge uns. Wenn unsere Arbeit nichts für dich ist, ist das völlig in Ordnung — die Wahl liegt bei dir. Danke fürs Vorbeischauen!
- **H — Hindi** (हिन्दी): ⭐ अगर आपको यह ऐप और हमारा काम पसंद है, तो कृपया रिपॉजिटरी को स्टार दें और हमें फॉलो करें। अगर हमारा काम आपके लिए नहीं है, तो भी कोई बात नहीं — यह पूरी तरह आपकी मर्ज़ी है। यहाँ आने के लिए आपका धन्यवाद!
- **I — Indonesian** (Bahasa Indonesia): ⭐ Jika kamu suka aplikasi ini dan karya kami, beri bintang pada repositori ini dan ikuti kami. Kalau karya kami bukan untukmu, tidak masalah sama sekali — itu pilihanmu. Terima kasih sudah mampir!
- **J — Japanese** (日本語): ⭐ このアプリと私たちの仕事が気に入ったら、リポジトリにスターを付けてフォローしてください。合わない場合はまったく問題ありません — 選ぶのはあなたです。お越しいただきありがとうございます！
- **K — Korean** (한국어): ⭐ 이 앱과 저희 작업이 마음에 드신다면 저장소에 스타를 눌러 주시고 팔로우해 주세요. 저희 작업이 맞지 않으셔도 전혀 괜찮습니다 — 선택은 여러분의 몫입니다. 방문해 주셔서 감사합니다!
- **L — Lithuanian** (Lietuvių): ⭐ Jei tau patinka ši programa ir mūsų darbas, duok saugyklai žvaigždutę ir sek mus. Jei mūsų darbas ne tau, viskas gerai — pasirinkimas tavo. Ačiū, kad užsukai!
- **M — Malayalam** (മലയാളം): ⭐ നിങ്ങൾക്ക് ഈ ആപ്പും ഞങ്ങളുടെ പ്രവർത്തനവും ഇഷ്ടപ്പെട്ടെങ്കിൽ, ദയവായി റെപ്പോസിറ്ററിക്ക് ഒരു സ്റ്റാർ നൽകുകയും ഞങ്ങളെ ഫോളോ ചെയ്യുകയും ചെയ്യുക. ഞങ്ങളുടെ പ്രവർത്തനം നിങ്ങൾക്ക് അനുയോജ്യമല്ലെങ്കിൽ, അതും തികച്ചും കുഴപ്പമില്ല — തീരുമാനം നിങ്ങളുടേതാണ്. ഇവിടെ വന്നതിന് നന്ദി!
- **N — Nepali** (नेपाली): ⭐ यदि तपाईंलाई यो एप र हाम्रो काम मन पर्‍यो भने, कृपया रिपोजिटरीलाई स्टार दिनुहोस् र हामीलाई फलो गर्नुहोस्। यदि हाम्रो काम तपाईंलाई मन पर्दैन भने, पनि ठीकै छ — यो तपाईंको इच्छा हो। यहाँ आउनुभएकोमा धन्यवाद!
- **O — Odia** (ଓଡ଼ିଆ): ⭐ ଯଦି ଆପଣଙ୍କୁ ଏହି ଆପ୍ ଏବଂ ଆମର କାମ ଭଲ ଲାଗୁଛି, ତେବେ ଦୟାକରି ରିପୋଜିଟୋରୀକୁ ଗୋଟିଏ ଷ୍ଟାର୍ ଦିଅନ୍ତୁ ଏବଂ ଆମକୁ ଫଲୋ କରନ୍ତୁ। ଯଦି ଆମର କାମ ଆପଣଙ୍କ ପାଇଁ ନୁହେଁ, ତେବେ କିଛି ଅସୁବିଧା ନାହିଁ — ପସନ୍ଦ ଆପଣଙ୍କର। ଆମ ପାଖକୁ ଆସିଥିବାରୁ ଧନ୍ୟବାଦ!
- **P — Polish** (Polski): ⭐ Jeśli podoba Ci się ta aplikacja i nasza praca, daj repozytorium gwiazdkę i obserwuj nas. Jeśli nasza praca nie jest dla Ciebie, to zupełnie w porządku — wybór należy do Ciebie. Dziękujemy za odwiedziny!
- **Q — Quechua** (Runa Simi): ⭐ Kay app, ñuqaykup ruwasqanchikpas gustasunki chayqa, repositorioman huk estrella churay, ñuqaykutapas qatipayku. Mana gustasunki chayqa, allinmi — qampa akllayki. Hamusqaykimanta sulpayki!
- **R — Russian** (Русский): ⭐ Если вам нравится это приложение и наша работа, поставьте звёздочку репозиторию и подпишитесь на нас. Если наша работа не для вас — это совершенно нормально, выбор за вами. Спасибо, что заглянули!
- **S — Spanish** (Español): ⭐ Si te gusta esta app y nuestro trabajo, dale una estrella al repositorio y síguenos. Si nuestro trabajo no es para ti, no pasa nada — la decisión es tuya. ¡Gracias por pasarte!
- **T — Tamil** (தமிழ்): ⭐ உங்களுக்கு இந்த ஆப்பும் எங்க வேலையும் பிடிச்சிருந்தா, தயவுசெஞ்சு இந்த ரெபாசிட்டரிக்கு ஒரு ஸ்டார் குடுத்து, எங்களை ஃபாலோ பண்ணுங்க. எங்க வேலை உங்களுக்கு செட் ஆகலைன்னாலும் பரவாயில்லை — அது உங்க விருப்பம். இங்கே வந்ததுக்கு நன்றி!
- **U — Urdu** (اردو): ⭐ اگر آپ کو یہ ایپ اور ہمارا کام پسند آئے تو براہ کرم ریپازٹری کو سٹار دیں اور ہمیں فالو کریں۔ اگر ہمارا کام آپ کے لیے نہیں ہے، تو یہ بالکل ٹھیک ہے — انتخاب آپ کا ہے۔ یہاں تشریف لانے کا شکریہ!
- **V — Vietnamese** (Tiếng Việt): ⭐ Nếu bạn thích ứng dụng này và công việc của chúng tôi, hãy tặng kho lưu trữ một ngôi sao và theo dõi chúng tôi. Nếu công việc của chúng tôi không phù hợp với bạn, điều đó hoàn toàn ổn — lựa chọn là của bạn. Cảm ơn bạn đã ghé qua!
- **W — Welsh** (Cymraeg): ⭐ Os ydych chi'n hoffi'r ap hwn a'n gwaith, rhowch seren i'r storfa a dilynwch ni. Os nad yw ein gwaith ar eich cyfer, mae hynny'n hollol iawn — chi sydd i benderfynu. Diolch am alw heibio!
- **X — Xhosa** (isiXhosa): ⭐ Ukuba uyayithanda le app nomsebenzi wethu, nika irepo inkwenkwezi kwaye usilandele. Ukuba umsebenzi wethu awukufanelanga, akukho ngxaki — lukhetho lwakho. Enkosi ngokutyelela!
- **Y — Yoruba** (Yorùbá): ⭐ Bí app yìí àti iṣẹ́ wa bá wù ọ́, jọ̀wọ́ fún àkójọpọ̀ náà ní ìràwọ̀ kí o sì tẹ̀lé wa. Bí iṣẹ́ wa kò bá yẹ ọ́, kò séwu — ìwọ ni ó yàn. Ẹ ṣé fún wíwá!
- **Z — Zulu** (isiZulu): ⭐ Uma uthanda le app nomsebenzi wethu, nikeza isitoreji inkanyezi bese usilandela. Uma umsebenzi wethu ungakufanele, akunankinga — wukukhetha kwakho. Siyabonga ngokuvakasha!

**More Indian languages**

- **Assamese** (অসমীয়া): ⭐ যদি আপুনি এই এপটো আৰু আমাৰ কাম ভাল পায়, তেন্তে অনুগ্ৰহ কৰি ৰিপজিটৰীটোক এটা ষ্টাৰ দিয়ক আৰু আমাক ফ'ল' কৰক। যদি আমাৰ কাম আপোনাৰ বাবে নহয়, তেন্তে সেয়া একেবাৰে ঠিক আছে — সিদ্ধান্তটো আপোনাৰ। আমাৰ ইয়ালৈ অহাৰ বাবে ধন্যবাদ!
- **Gujarati** (ગુજરાતી): ⭐ જો તમને આ એપ અને અમારું કામ ગમતું હોય, તો પ્લીઝ રિપોઝિટરીને સ્ટાર આપો અને અમને ફોલો કરો. જો અમારું કામ તમને ન ગમતું હોય, તો તે પણ બરાબર છે — પસંદગી તમારી છે. અહીં આવવા બદલ આભાર!
- **Kannada** (ಕನ್ನಡ): ⭐ ನಿಮಗೆ ಈ ಆ್ಯಪ್ ಮತ್ತು ನಮ್ಮ ಕೆಲಸ ಇಷ್ಟವಾದರೆ, ದಯವಿಟ್ಟು ರೆಪೊಸಿಟರಿಗೆ ಸ್ಟಾರ್ ಕೊಡಿ ಮತ್ತು ನಮ್ಮನ್ನು ಫಾಲೋ ಮಾಡಿ. ನಮ್ಮ ಕೆಲಸ ನಿಮಗೆ ಸರಿಹೊಂದದಿದ್ದರೆ, ಅದೂ ಪರವಾಗಿಲ್ಲ — ಆಯ್ಕೆ ನಿಮ್ಮದು. ಭೇಟಿ ನೀಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು!
- **Marathi** (मराठी): ⭐ जर तुम्हाला हे ॲप आणि आमचं काम आवडलं असेल, तर कृपया रिपॉझिटरीला स्टार द्या आणि आम्हाला फॉलो करा. जर आमचं काम तुम्हाला आवडत नसेल, तर काही हरकत नाही — ती तुमची निवड आहे. इथे आल्याबद्दल धन्यवाद!
- **Punjabi** (ਪੰਜਾਬੀ): ⭐ ਜੇ ਤੁਹਾਨੂੰ ਇਹ ਐਪ ਅਤੇ ਸਾਡਾ ਕੰਮ ਪਸੰਦ ਹੈ, ਤਾਂ ਕਿਰਪਾ ਕਰਕੇ ਰਿਪੋਜ਼ਟਰੀ ਨੂੰ ਸਟਾਰ ਦਿਓ ਅਤੇ ਸਾਨੂੰ ਫਾਲੋ ਕਰੋ। ਜੇ ਸਾਡਾ ਕੰਮ ਤੁਹਾਡੇ ਲਈ ਨਹੀਂ ਹੈ, ਤਾਂ ਕੋਈ ਗੱਲ ਨਹੀਂ — ਚੋਣ ਤੁਹਾਡੀ ਹੈ। ਸਾਡੇ ਪੇਜ 'ਤੇ ਆਉਣ ਲਈ ਧੰਨਵਾਦ!
- **Telugu** (తెలుగు): ⭐ మీకు ఈ యాప్, మా పని నచ్చితే, దయచేసి రిపోజిటరీకి ఒక స్టార్ ఇచ్చి మమ్మల్ని ఫాలో అవ్వండి. ఒకవేళ మా పని మీకు నచ్చకపోతే, పర్వాలేదు — అది మీ ఇష్టం. మమ్మల్ని సందర్శించినందుకు ధన్యవాదాలు!

</details>
