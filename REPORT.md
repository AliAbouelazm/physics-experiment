# Physics experiment: research report

**Negative result. Learned adaptation did not add demonstrated value over simple identification.** This distribution is a working recorded-data inspector plus the complete direct-contact and cached-comparison evidence. It is not a visual world model, successful control policy or claim of generalization. No experiment or fit was repeated to prepare it.

## Direct-contact qualification

All 32 physical cells and 96 target contexts were retained: 288 recorded trajectories, 69,696 controls, 432 resets, 30.708 seconds and 326 MiB peak RSS. Both probes and every non-hold action contacted the body. All setup matching checks passed. The original threshold remains unchanged.

| Gate | Result |
| --- | --- |
| Complete matched roster | Passed |
| Controllability | 72/96, required 72: passed |
| Informative choice | 16 groups, required 6: passed |
| Absolute geometry-aware oracle headroom | 0.122335, required 0.1: passed |
| Relative geometry-aware oracle headroom | 19.5133%, required 20%: **failed** |

Mean terminal losses: hold 1.240000; global fixed 0.709177; geometry-and-target-aware control 0.626929; finite oracle 0.504594. The larger 28.85% headroom versus global fixed is not the qualification gate. The geometry-aware comparator selects by known geometry/target while averaging all four physics, avoiding that confound. This full-roster gate is still FAILED; no threshold was weakened or successful subset selected.

## Cached learned comparison

The corrected 115-parameter model fits crossbar contacts and evaluates stem contacts. Three seeds each completed exactly 300 fixed updates, followed by a fresh three-coefficient correction from two probes per cell. The simple estimator has the same probes/ridge strength without a learned prior. No new native simulation occurred. All recorded numerical files and checkpoints are byte-identical to the executed evidence.

The single execution took 1.2704 seconds and 336.82 MiB. Reported timing is on the original CPU runtime; it is not a performance guarantee for another machine. The scientific protocol is restated in [PROTOCOL.md](PROTOCOL.md), with full raw results in the two evidence directories.

| Seed / method | Prediction MSE | Decision loss | Regret | Landings / 48 |
| --- | ---: | ---: | ---: | ---: |
| 9101 frozen | 0.962986 | 0.896617 | 0.172721 | 0 |
| 9101 adapted | 0.369008 | 0.955304 | 0.231407 | 4 |
| 9102 frozen | 0.953786 | 0.849570 | 0.125673 | 0 |
| 9102 adapted | 0.396241 | 0.971647 | 0.247751 | 4 |
| 9103 frozen | 0.965885 | 0.857002 | 0.133106 | 0 |
| 9103 adapted | 0.243799 | 0.900454 | 0.176557 | 8 |
| simple | 0.137565 | 0.741850 | 0.017954 | 8 |
| global fixed | N/A | 0.872217 | 0.148320 | 8 |
| fit fixed | N/A | 0.872217 | 0.148320 | 8 |
| geometry target | N/A | 0.801815 | 0.077918 | 0 |
| oracle | N/A | 0.723896 | 0.000000 | 8 |
| hold | N/A | 1.240000 | 0.516104 | 0 |

All four composite criteria failed in every seed. Adaptation improved prediction against frozen by about 58-75%, but still lost to the simple estimator and made decisions worse in all seeds. Zero of four response groups improved against both comparators. Even simple identification fails 40/48 landings. The finite loss oracle is not a success-maximizing oracle.

## Prediction and landing errors

Prediction error compares forecast endpoints with actual endpoints across every non-hold action. Landing error describes the action selected for a target. Lower angular error can come with greater translation error. These metrics must not be conflated.

| Seed / method | Prediction centroid MAE (px) | Prediction angle MAE (deg) | Landing centroid error (px) | Landing angle error (deg) |
| --- | ---: | ---: | ---: | ---: |
| 9101 frozen | 14.910 | 30.616 | 12.923 | 9.576 |
| 9101 adapted | 10.456 | 10.490 | 16.494 | 4.525 |
| 9102 frozen | 14.625 | 30.472 | 14.605 | 7.025 |
| 9102 adapted | 10.893 | 10.719 | 16.681 | 4.303 |
| 9103 frozen | 14.826 | 30.691 | 13.886 | 8.084 |
| 9103 adapted | 7.278 | 10.021 | 15.370 | 5.993 |
| Simple estimator | 4.901 | 5.793 | 10.028 | 13.100 |

## Limits and earlier development

All physics are familiar TRAIN values. Rotation/reflection makes many records near-exact duplicates: only four response groups per contact family remain. The 48 held-out target cases are correlated, not 48 independent experiments. There is only one held-out geometry family. Probe updates may correct geometry shift as well as physics. These are internal development results after earlier inspection, not untouched generalization or population-significance evidence. Reserved FINAL physics was not evaluated.

Earlier development included a small trained 1D pilot (decision-directed and uncertainty downstream actions tied in all 36 pairs), a 162-condition update-aware diagnostic (same actions as uncertainty at about 54.47x scoring cost), and successive routed PushT geometry/model diagnostics. The routed Stage A retained all 432 trajectories but failed controllability (19/48) and oracle headroom (0.896 percentage points). These stages motivated direct contact; they are not new successful benchmarks or included as extra independent evaluation cases. The complete earlier development record is retained separately by the author. This focused distribution contains the full two final study datasets, models and outcomes rather than private operating history.

## Inspect and verify

The viewer draws actual saved native observations. Solid shapes are recorded poses, dashed shapes are landing targets and dotted shapes are saved endpoint forecasts. No new physics or live inference is performed by the UI. In-sample cases remain labeled. Desktop/mobile screenshots show genuine browser captures, including failures.

Read-only checks reconstruct predictions from all three saved checkpoints and two recorded probe endpoints. They reproduce all 288 method/cell/seed prediction sets and verify 2,304 method/cell/seed/target decision records. The retained access log records zero held-out candidate reads before selections were sealed and exactly 112 afterwards. Each method is charged 726 controls and 3-6 recorded resets per target trial. Sharing a cache for inspection does not make real probes free.

[PUBLIC_PROVENANCE.json](PUBLIC_PROVENANCE.json) identifies unchanged scientific/media bytes. [PUBLIC_MANIFEST.json](PUBLIC_MANIFEST.json) verifies the sanitized distribution. Historical manifest hashes inside saved summaries identify the original experimental freeze; they were not rewritten into a fictitious new experiment. Screenshots and numerical files are unchanged. Documentation, portability, license notices and the verification entry points are publication edits.

Research iteration is stopped. This artifact demonstrates measured failure and the distinction between prediction and control, without implying novelty or learned-adaptation benefit.
