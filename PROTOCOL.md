# Scientific protocols preserved by this distribution

No experiment was rerun for this distribution. Scientific data, results and trained checkpoint bytes are unchanged. This document restates the executed specifications without private operating instructions. Historical manifest identifiers remain in saved summaries; PUBLIC_MANIFEST.json verifies the sanitized package, not a newly conducted experiment.

## Direct-contact qualification

Four TRAIN physics settings (damping 0.2/0.8, COG-x -15/+15), four public contact surfaces and initial orientations 0/pi/4 form 32 physical cells. Body origin is (256,236); pusher starts 17 pixels outward from its surface midpoint, giving 2-pixel geometric clearance for radius 15. Seeds 8800-8831 follow cell order. Initial observations match across physics through bounded observation-based reset correction, at most three resets plus two history-hold controls. Every case is retained.

Targets independently translate the initial geometric centroid 8, 20 or 32 pixels inward, preserving orientation. This yields 96 correlated target contexts. Decisions are hold or all combinations of 0.25/0.75/1.5 pixels target advance per control and 20/40 control durations. Two fixed probes use (0.25,20) and (1.0,20). After pushing, the target withdraws at 2 pixels/control for 40 controls, then holds through control 240. Requested targets are capped at 2 pixels from observed pusher position and executed by the native public PD tracker.

Every probe/action begins a separate matched reset under the same hidden physics. A method pays 726 controls for two probe episodes plus one decision, including history holds, with at most nine reset attempts. The qualification caches two probes and seven actions per cell: 288 trajectories and 69,696 controls. This is repeated identification across reset trials, not continuous adaptation through one episode. Native contact flags are evaluator-only.

Terminal loss is (squared geometric-centroid error + (50 times wrapped angle error in radians)^2)/400. Success separately requires centroid error <=5 pixels and angular error <=5 degrees. Gates: complete matched roster; >=72/96 contexts with contact and >=0.1 loss improvement over hold; informative action flips in >=6/24 geometry/target groups; oracle headroom >=0.1 AND >=20% over geometry-and-target-aware no-physics control.

A flip requires each physics setting's optimum to beat the other's action by >=0.1, with corresponding probes contacting in both and histories separating by >=1 pixel or >=1 degree. The no-physics comparator picks one action per orientation/surface/target group by averaging all four physics. Global fixed control is reported separately. Both are evaluator references, not individual-physics learner inputs. Hard resource ceiling: 120 seconds / 2 GiB, one CPU thread, one complete attempt. No threshold adjustment.

## Exploratory cached comparison

Crossbar surfaces 0/1 form the fitting family (cells 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29). Stem surfaces 2/3 form the held-out model-fitting family (2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31). Rotated/reflected counterparts and all target distances stay within their family. All physics are familiar TRAIN values. One held-out geometry family and four symmetry-reduced response groups do not establish population generalization.

Fit 128 endpoint records: two probes and six non-hold decisions for each fitting cell. Hold prediction is exactly zero and excluded from fitting. Model input is [speed/1.5, duration/40, speed*duration/60]. The crossbar-constant lever-arm feature was removed before execution because it was unidentifiable in fitting. Public geometry is retained only for canonicalization.

Let e be the inward normal, t=R(initial_angle)*(0,1) and f the inward local-x sign. Output label is [centroid displacement dot e/20, displacement dot t/20, 50*f*wrapped angle displacement/20] at control 240. Model: Linear(3,16), Tanh, Linear(16,3), 115 parameters. CPU float64; seeds 9101/9102/9103; explicit per-layer uniform initialization +/-1/sqrt(fan_in). Exactly 300 full-batch Adam steps per seed, lr .001, betas (.9,.999), epsilon 1e-8, weight decay .0001, component/record mean squared error, gradient norm cap 1. No sweep or checkpoint selection.

Frozen and adapted use the same trained checkpoint. Per cell, adaptation freezes the base and solves three independent scalar ridge regressions from exactly two probe endpoints. With phi=speed*duration/20 and probe phi values .25 and 1:

`b = sum(phi * (observed_y - base_y)) / (0.1 + sum(phi^2))`

Adapted prediction is base_y + phi*b. The simple estimator uses the same formula with base_y=0, so it has the same probes and ridge strength without a learned prior. Coefficients reset per cell. Targets and candidate outcomes do not enter updates. Corrections can address geometry shift as well as physics; they do not identify true damping or prove unseen-physics adaptation.

Each method picks the lowest predicted original loss among all seven actions; exact ties choose lowest index. Checkpoint hashes/predictions/choices were saved before the evaluator read the 112 held-out candidate trajectories. Full 32-cell predictions and 96 target contexts are retained, with fitting cases explicitly in-sample. Only the complete 48-target held-out family determines the exploratory criteria.

Every seed must meet all criteria: adapted prediction MSE over six non-hold actions >=5% below frozen and no worse than simple; decision loss at least .01 below both frozen and simple; lower loss than both in >=3/4 response groups; mean loss no worse than geometry-aware control. Groups are indexed evaluator-only by damping and inward-sign times COG-x. Report prediction, centroid/angular landing errors and success separately. No bootstrap over symmetry duplicates or population significance claim.

One execution, three fits/900 updates, one analytic correction per cell/seed. Hard ceiling 120 seconds / 2 GiB, one CPU thread, no new native physics. All criteria failed. Benchmark iteration stopped; direct-contact qualification remains failed.
