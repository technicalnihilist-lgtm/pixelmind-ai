# PixelMind AI — 20-day optimisation backlog

Ten items per day, 1–20 November 2026. **200 items total.**

Rules for every item:

* The UI does not change. No layout, colour, typography, spacing, animation, navigation or copy
  changes beyond making an error message true and actionable.
* No new features, no new permissions. The app keeps declaring zero platform permissions.
* No change to the model format, the tensor contract, or the bundled verification model.
* Every item ends with `./gradlew :app:testReleaseUnitTest` green and an `assembleRelease` that
  produces an APK no larger than the previous one unless the change justifies the growth.
* If an item turns out to be already handled, tick it and say so — do not invent work.
* Never claim device behaviour that was not measured on a device.

A run takes the **next ten unchecked items**, top to bottom.

---

## Day 1 — crash and lifecycle sweep

- [x] 1. Audit every `!!` and `requireNotNull` in the engine; replace with explicit failure states where a user could trigger them.
- [x] 2. Ensure `OnnxDiffusionEngine.generate` cannot leak `OnnxTensor`/`Result` on the exception path (close in `finally`, not only on success).
- [x] 3. Handle configuration change (rotation) during generation without restarting the job.
- [x] 4. Handle process death mid-generation: no half-written PNG left in the gallery.
- [x] 5. Guard `GenerationManager.runJob` against the engine being released while a job is in flight.
- [x] 6. Make `PixelMindApp` lifecycle observer idempotent (repeated `onStop` must not double-release).
- [x] 7. Replace `System.gc()` after generation with a measured, documented alternative or justify keeping it.
- [x] 8. Audit every `catch (t: Throwable)` — none should swallow an `OutOfMemoryError` silently.
- [x] 9. Verify `MainActivity` survives `savedInstanceState` restoration with an in-flight job.
- [x] 10. Add a unit test for the provisioning helper's failure path (read-only directory, truncated asset).

## Day 2 — concurrency

- [ ] 11. Make `GenerationManager.cancelActive` cancel the actual worker job, not a wrapper that has already completed.
- [ ] 12. Prove queue ordering is FIFO under concurrent `enqueue` calls; add a test.
- [ ] 13. Ensure `retry` cannot run a job twice if tapped quickly.
- [ ] 14. Replace `synchronized(lock)` around the whole generation with a `Mutex`, so cancellation is not blocked by the lock.
- [ ] 15. Ensure progress callbacks never arrive after the job is marked finished.
- [ ] 16. Make `engineStatus` transitions monotonic — no `Ready → Loading` flicker while a job runs.
- [ ] 17. Audit `Dispatchers.Default` vs `IO` usage across engine, importer and repository.
- [ ] 18. Ensure `SharedFlow`/`Channel` result events cannot be dropped when the UI is not collecting.
- [ ] 19. Guard against two `initialize()` calls racing (double session creation).
- [ ] 20. Add a stress test that enqueues 50 jobs and asserts exactly 50 terminal states.

## Day 3 — memory

- [ ] 21. Avoid the full `FloatArray` copy of every UNet output when the tensor can be read in place.
- [ ] 22. Reuse latent buffers across steps instead of allocating per step.
- [ ] 23. Recycle the intermediate `Bitmap` in `decodeLatents` when the request is cancelled.
- [ ] 24. Audit `ImageStore` LRU cache sizing against real device heap and cap it sensibly.
- [ ] 25. Ensure `previewCache` entries are evicted when an image is deleted.
- [ ] 26. Check `ClipTokenizer` caches are bounded (they grow with prompt vocabulary).
- [ ] 27. Measure and document peak RSS for 512×512 at each quality tier.
- [ ] 28. Free the embedding tensor immediately after the last diffusion step.
- [ ] 29. Avoid retaining `GenerationRequest`/`Bitmap` in `lastResult` after the UI has consumed them.
- [ ] 30. Add a test that a cancelled generation releases its buffers (no growth across 20 cancels).

## Day 4 — startup and perceived performance

- [ ] 31. Defer tokenizer parsing until the first generation instead of at session creation.
- [ ] 32. Move `ModelLocator.resolve()` checksum verification off the main thread.
- [ ] 33. Skip re-verifying SHA-256 on every launch once a model has been verified (store a marker).
- [ ] 34. Measure cold start to first frame and record it in docs/MEASUREMENTS.md.
- [ ] 35. Warm up only the graphs the first generation will actually use.
- [ ] 36. Avoid recomposing Home when only the engine status changes.
- [ ] 37. Make the gallery thumbnail decode honour a smaller `maxEdge` on low-RAM devices.
- [ ] 38. Pre-size the SQLite database and enable WAL.
- [ ] 39. Measure session creation time per graph and log it.
- [ ] 40. Reduce the APK's dex count where a dependency is only used in one place.

## Day 5 — I/O and storage

- [ ] 41. Batch gallery inserts/updates in a transaction.
- [ ] 42. Add an index usage review for `created_at` and `favorite` queries.
- [ ] 43. Ensure PNG encode uses the smallest acceptable config (no ARGB_8888 where RGB suffices).
- [ ] 44. Stream model import instead of buffering whole files.
- [ ] 45. Free disk space check before importing a model; fail early with a clear message.
- [ ] 46. Delete `.part` staging files on any failed import.
- [ ] 47. Ensure `ImageStore.save` is atomic (write temp, rename).
- [ ] 48. Audit file descriptor usage in the importer.
- [ ] 49. Make prompt history insertion idempotent under rapid generation.
- [ ] 50. Add a storage-usage breakdown test that matches what Settings displays.

## Day 6 — error handling and messages

- [ ] 51. Every `EngineStatus.Failed` detail should name the failing graph.
- [ ] 52. Distinguish "weights absent" from "weights corrupt" from "weights unreadable" in the UI.
- [ ] 53. Surface OOM as a specific, actionable message with the suggested lower setting.
- [ ] 54. Log the resolved accelerator and thread count once per session.
- [ ] 55. Ensure cancellation produces no error toast.
- [ ] 56. Make import failures say which file and why.
- [ ] 57. Replace generic `t.message ?: "unknown"` with a mapped, human-readable reason.
- [ ] 58. Ensure the "generation failed" card offers Retry only when a retry can succeed.
- [ ] 59. Add a debug-only diagnostics dump (model, accelerator, RAM, timings) copyable to clipboard.
- [ ] 60. Verify no error string leaks a filesystem path to the UI.

## Day 7 — model import robustness

- [ ] 61. Handle a zip with a single top-level folder (strip the common prefix).
- [ ] 62. Reject archives that contain no ONNX at all with a precise message.
- [ ] 63. Handle duplicate entries in a zip deterministically.
- [ ] 64. Cap per-entry size as well as total size.
- [ ] 65. Verify sha256 of imported components when a manifest provides them.
- [ ] 66. Handle `content://` URIs whose display name has no extension.
- [ ] 67. Ensure an interrupted import leaves the previous model intact.
- [ ] 68. Test rollback when the imported model loads but produces wrong tensor shapes.
- [ ] 69. Handle storage revoked mid-import (SecurityException).
- [ ] 70. Add tests for zip-slip variants: encoded `..%2f`, NUL bytes, backslash paths.

## Day 8 — inference correctness

- [ ] 71. Verify `ClipTokenizer` against a reference vector of known prompt→token ids.
- [ ] 72. Verify the LCM scheduler against a reference implementation at 4/6/8 steps.
- [ ] 73. Confirm CFG maths: `uncond + scale * (cond - uncond)` with a numeric test.
- [ ] 74. Verify seed determinism across two runs with the same seed.
- [ ] 75. Confirm latent scaling (0.18215) is applied once and only once.
- [ ] 76. Check VAE output channel order (RGB, not BGR) against the manifest.
- [ ] 77. Verify aspect-ratio latents are all multiples of 8 and produce matching output sizes.
- [ ] 78. Ensure negative prompt influences the unconditional pass only.
- [ ] 79. Verify timestep dtype (int64) and shape [1] match the graph.
- [ ] 80. Add a golden-image test at low resolution with a fixed seed and a tolerance.

## Day 9 — code quality

- [ ] 81. Remove dead code and unused imports across `engine/`, `manager/`, `data/`.
- [ ] 82. Extract duplicated tensor-creation boilerplate into one helper.
- [ ] 83. Give every public engine type a KDoc sentence explaining its contract.
- [ ] 84. Rename ambiguous identifiers (`fitted`, `located`) where the intent is unclear.
- [ ] 85. Replace magic numbers (512, 0.18215, 77) with named constants.
- [ ] 86. Consolidate the two places that compute a model's byte size.
- [ ] 87. Add tests for `MemoryGuard.estimatePeakMb` across device classes.
- [ ] 88. Ensure every `TODO` is either resolved or turned into a tracked item here.
- [ ] 89. Make `ModelManifest.parse` fail with a field-specific message.
- [ ] 90. Review package structure: does anything in `ui/` import `engine/` internals?

## Day 10 — build health

- [ ] 91. Fix all Kotlin compiler warnings.
- [ ] 92. Review ProGuard rules; ensure R8 keeps the JNI surface.
- [ ] 93. Verify the release APK shrinks with R8 on a machine that can run it.
- [ ] 94. Trim unused transitive dependencies.
- [ ] 95. Confirm `abiFilters` and that no x86 libs sneak in.
- [ ] 96. Check the APK has no debug symbols or leftover test assets.
- [ ] 97. Enable Gradle configuration cache if compatible.
- [ ] 98. Document the reproducible build command in docs/BUILD.md.
- [ ] 99. Add a CI-ready script that runs tests + assemble in one command.
- [ ] 100. Record APK size and dex/lib composition in docs/MEASUREMENTS.md.

## Day 11 — battery and thermal

- [ ] 101. Ensure the engine releases sessions within N seconds of backgrounding.
- [ ] 102. Avoid holding a wake lock (verify none exists).
- [ ] 103. Cap intra-op threads so a generation does not saturate every core.
- [ ] 104. Measure and document energy per generation on a device (or state that it is unmeasured).
- [ ] 105. Ensure no polling loops exist anywhere.
- [ ] 106. Make the ambient background animation stop when the app is not visible.
- [ ] 107. Verify the infinite transition does not run during generation (competing for GPU).
- [ ] 108. Reduce per-step logging in release builds (timings only, not per-element).
- [ ] 109. Ensure gallery thumbnail decoding does not run while generating.
- [ ] 110. Document thermal throttling expectations for long sessions.

## Day 12 — resilience

- [ ] 111. Survive a corrupted database by rebuilding it rather than crashing.
- [ ] 112. Survive a missing/corrupt manifest by falling back to the bundled model.
- [ ] 113. Handle a device with zero free space gracefully.
- [ ] 114. Handle an ONNX graph that loads but rejects the input shape, with a clear message.
- [ ] 115. Handle the case where the accelerator accepts the session but fails on first run — fall back mid-session.
- [ ] 116. Add a retry-once policy for transient session creation failures.
- [ ] 117. Ensure the app still opens when the model directory is a file, not a directory.
- [ ] 118. Handle clock changes and timezone shifts in "Recent" ordering.
- [ ] 119. Ensure deletion of an already-deleted image does not throw.
- [ ] 120. Fuzz the manifest parser with malformed JSON.

## Day 13 — accessibility and input

- [ ] 121. Add content descriptions to every icon that carries meaning (no visual change).
- [ ] 122. Ensure touch targets are at least 48 dp.
- [ ] 123. Verify contrast ratios of body text against the glass panels.
- [ ] 124. Support font scaling to 200% without clipping.
- [ ] 125. Ensure the IME does not cover the prompt field when focused.
- [ ] 126. Announce generation progress to screen readers.
- [ ] 127. Respect "remove animations" system setting.
- [ ] 128. Ensure haptics are used only if enabled by the system.
- [ ] 129. Verify keyboard navigation and focus order.
- [ ] 130. Test RTL layout does not break the studio controls.

## Day 14 — performance profiling

- [ ] 131. Profile one generation end to end and attribute time per stage.
- [ ] 132. Find the hottest Kotlin function outside the graph execution and optimise it.
- [ ] 133. Reduce allocations in the per-step loop.
- [ ] 134. Measure tokenizer time for a 200-word prompt.
- [ ] 135. Measure PNG encode time at 512 and 640.
- [ ] 136. Check for unnecessary `toFloatArray()` copies in the decode path.
- [ ] 137. Verify XNNPACK thread count vs core count on a big.LITTLE device.
- [ ] 138. Compare `OptLevel.ALL_OPT` vs `EXTENDED_OPT` for session creation time.
- [ ] 139. Cache the compiled tokenizer across generations.
- [ ] 140. Document the measured breakdown in docs/MEASUREMENTS.md.

## Day 15 — data integrity

- [ ] 141. Verify gallery records are removed when their PNG is missing.
- [ ] 142. Verify favourites survive a database upgrade.
- [ ] 143. Ensure prompt history deduplicates case-insensitively.
- [ ] 144. Add a repair pass that reconciles the DB with the filesystem.
- [ ] 145. Ensure seeds are stored exactly (long, not int).
- [ ] 146. Verify timestamps are monotonic for ordering.
- [ ] 147. Handle very long prompts without truncating mid-token.
- [ ] 148. Ensure import replaces records correctly when the same model is re-imported.
- [ ] 149. Verify "restore bundled model" clears derived state.
- [ ] 150. Add a test that a full delete + reinstall path works.

## Day 16 — developer experience

- [ ] 151. Make the build reproducible with one script.
- [ ] 152. Document every Gradle property that matters.
- [ ] 153. Add a `--verbose` engine logging switch for debugging.
- [ ] 154. Document the tensor contract with shapes in one table.
- [ ] 155. Add a smoke-test script that runs the pipeline on a desktop against any model dir.
- [ ] 156. Ensure `tools/verify_export.py` covers the manifest fields the app requires.
- [ ] 157. Document how to make a tiny model from scratch.
- [ ] 158. Keep CHANGELOG entries factual and dated.
- [ ] 159. Ensure docs match the shipped version numbers.
- [ ] 160. Add an architecture diagram that matches the code.

## Day 17 — hardening

- [ ] 161. Bound every user-supplied string (prompt length, seed digits).
- [ ] 162. Reject NaN/Inf in latents before decode.
- [ ] 163. Guard against integer overflow in resolution maths.
- [ ] 164. Validate imported manifest fields against sane ranges.
- [ ] 165. Ensure no path from a URI is used without sanitising.
- [ ] 166. Cap the number of queued jobs.
- [ ] 167. Ensure the DB cannot grow without bound (prune policy).
- [ ] 168. Verify no `Runtime.exec` or reflection to networking exists.
- [ ] 169. Confirm the manifest declares no permissions after every dependency change.
- [ ] 170. Add a test asserting the INTERNET permission is absent.

## Day 18 — polish without redesign

- [ ] 171. Make the progress ring's stage labels accurate for every path.
- [ ] 172. Ensure the timing card shows "—" rather than zeros when unmeasured.
- [ ] 173. Make snackbars non-duplicating.
- [ ] 174. Ensure the import status clears when leaving Settings and returning.
- [ ] 175. Verify the empty states still appear after deleting everything.
- [ ] 176. Ensure "reuse prompt" fills both prompt and negative prompt.
- [ ] 177. Verify share intents carry the right MIME type.
- [ ] 178. Ensure long model names truncate instead of wrapping.
- [ ] 179. Verify the gallery grid keeps scroll position after returning from detail.
- [ ] 180. Confirm no screen blocks the UI thread for more than one frame.

## Day 19 — measurement and documentation

- [ ] 181. Record real device timings if any device is available; otherwise state clearly that none were.
- [ ] 182. Update docs/MEASUREMENTS.md with the current APK size and composition.
- [ ] 183. Document the memory guard's formula and thresholds.
- [ ] 184. Document what each accelerator choice costs.
- [ ] 185. Document the import formats accepted.
- [ ] 186. Update the status table in README to match reality.
- [ ] 187. Ensure VERIFICATION.md reflects what has actually been run.
- [ ] 188. List known limitations honestly.
- [ ] 189. Document the verification model's provenance and licence.
- [ ] 190. Ensure the changelog has no unverified claims.

## Day 20 — final pass

- [ ] 191. Full read-through of the engine for anything left unhandled.
- [ ] 192. Full read-through of the UI layer for anything that could crash.
- [ ] 193. Run every test; fix any flakiness.
- [ ] 194. Verify a clean clone builds without manual steps.
- [ ] 195. Verify the release APK installs over the previous version.
- [ ] 196. Summarise the 20 days: what changed, what improved, what remains.
- [ ] 197. Confirm no UI regression across the four screens (compare against design/ renders).
- [ ] 198. Confirm the app still declares zero permissions.
- [ ] 199. Tag a final release with the accumulated changes.
- [ ] 200. Write a short honest retrospective, including what could not be verified.

---

## Log

| Day | Date | Items done | Tests | APK size | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | 5 Nov 2026 | 1–10 | 34/34 pass (`:app:testReleaseUnitTest`) | 23,287,301 bytes (22.2 MB), arm64-v8a, no bundled weights | Crash/lifecycle sweep. 7 `requireNotNull` sites became explicit `EngineStatus.Failed` states; new `describeFailure` maps OOM/SecurityException/FileNotFound to actionable sentences and OOM is logged at error level instead of being retried as a fallback; post-generation `System.gc()` removed (see docs/MEASUREMENTS.md); `ImageStore.save` stages to `*.part` and renames; `releaseEngine` is idempotent and marks the in-flight job CANCELLED; a queued job reloads the engine instead of failing with a false "open the app again". Items 2, 3 and 9 were already handled / inspection-verified only — no device or emulator. Tests 25 → 34. APK built with R8 disabled via a machine-local init script (R8 needs >5 min on this 2-core build pod, whose commands are capped at 300 s); repo build config unchanged. |
