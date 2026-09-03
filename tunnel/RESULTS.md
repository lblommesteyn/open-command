# Are pitch tunnels encoded in where pitchers intend to throw?

Intent-vs-actual tunneling on OpenCommand's frozen inferred targets, 2025 regular season,
with a 2024 replication at the end. Everything here is produced by `tunnel/*.py`; numbers are
in `tunnel/out/`, figures in `tunnel/fig/`. OpenCommand's target inference, camera pose and
command methodology were not touched.

## Answer in three lines

1. **Yes, tunnels are encoded in the targets.** Pairs whose *intended* trajectories tunnel
   (top quartile, thresholds frozen on the first half of the season) go on to tunnel in the
   *actual* trajectories 50% of the time, against 15.5% for pairs whose intent does not tunnel:
   a 3.2x lift on the held-out second half, robust across every checkpoint from 45 ft to 5 ft.
2. **But pitchers do not choose consecutive targets to tunnel much more than chance.** Against
   a within-pitcher, same-game, same-pitch-type shuffle null the mean intended tunnel score is
   only 0.035 log units above chance at 23.8 ft (a 3.5% better late/early separation ratio);
   against a within-PA order shuffle it is 0.005, and on the unadjusted targets it is 0.005.
   Per-pitcher excess repeats only weakly (split-half r = 0.35-0.40). The excess is a property
   of the pitch-pair type (split-half r = 0.87-0.94), not of the pitcher.
3. **Designed tunnels draw more swings than visually similar accidental ones (+2.6 pp, z = 5.7),
   but no more whiffs (+0.0 pp).** The tunnel pays as a swing/chase effect, not a miss effect.

So deliberate tunneling exists as a *pitch-mix* phenomenon (a slider after a four-seam is
aimed to share a corridor) and is measurable in the targets. As a per-pitcher sequencing skill
beyond that, the evidence is marginal and is not leaderboarded here.

## Data

- `data/2025/targets.csv.gz` (`inferred_x_in`, `inferred_z_in`), frozen, `status == ok` and
  `plausible`; `pbp_info.csv.gz` (Statcast 9-parameter trajectory, plate location, pitcher, pitch
  type, description); `pitch_context.csv.gz` (PA number, pitch number, count, batter side).
- 640,287 regular-season pitches, 821 pitchers, 2,359 games after merge. Ten pitch types (FF SI
  SL CH ST FC CU FS KC SV).
- Robustness file: `targets_unadjusted.csv.gz` (raw glove, no per-clip shrink toward the ball).

## Method

**1. Actual trajectories** (`trajectory.py`). Constant-acceleration solution of the 9-parameter
model at fixed distances from the plate apex: 50, 45, 40, 35, 30, 25, 23.8, 20, 15, 10, 5 ft and
the plate (17/12 ft). Validation against recorded `plate_x`/`plate_z`: median error 0.002 in,
p99 0.010 in, zero pitches over 1 in (`out/trajectory_validation_2025.txt`).

**2. Intended trajectories** (`intent_model.py`). Per pitcher x pitch type, at each checkpoint,
`P(d) = mu_d + B_d (plate_xz - mu_plate)` fit by centered OLS on training-fold pitches only,
`B_d` shrunk toward the pitch-type pooled within-pitcher slope with 50 pseudo-pitches. Folds are
by game (5), so a pitch's own plate location never enters its intended trajectory. The intended
trajectory is the same map evaluated at the OpenCommand target, so it terminates at the target
(median 0.003 in). Held-out check that the shape model is adequate: evaluating the map at a
held-out pitch's *actual* plate location reproduces its trajectory to 1.84 in at 50 ft, 1.11 in
at 23.8 ft, 0 at the plate (that column is validation only and is never used in a score). The
intent-vs-actual residual grows from 2.2 in at 50 ft to the 9.95 in target miss at the plate
(`out/intent_validation_2025.txt`).

**3. Tunnel scores** (`pairs.py`). Consecutive pitches in the same PA (438,300 pairs, 737
pitchers, 36% same-type). At each checkpoint `sep(d)` is the Euclidean x/z distance between the
two trajectories; the tunnel score is `ts(d) = log(sep_plate / sep(d))`, large when the pitches
share a corridor early and separate late. Intent uses the two intended trajectories and the two
targets; actual uses the two Statcast trajectories and plate locations. Both are computed at
every checkpoint; 23.8 ft (about 175 ms before the plate, the literature tunnel point) is the
pre-declared primary and every table is repeated across the sweep.

**4. Labels** (`classify_outcomes.py`). Thresholds = 75th percentile of `ts_intent` and
`ts_actual` on games before 2025-06-28 (exploratory half), plus a plate separation of at least
6 in so that two identical pitches are not a tunnel. Labels are applied to the second half only.
Designed = intent and actual tunnel; accidental = actual only; failed = intent only. Continuous
scores for every pair and checkpoint are in `out/pitch_pair_tunneling_2025.csv.gz`.

**5. Nulls** (`null_test.py`). Keep pitch A, replace pitch B by a random other pitch of the same
pitcher from a stratum, recompute both scores with the donor's trajectories; 200 draws.
`season_type` = same pitch type, batter side, strikes; `game_type` = same game, pitch type, side
(removes per-game target and camera-pose bias shared by a PA); `season_any` = same side and
strikes, any type (pitch selection + target); `pa_any` = another pitch of the same PA (pitch-order
shuffle); `game_any` = same game and side, any type.

**6. Repeatability** (`repeat.py`, `cross_season.py`). Excess = observed minus same-game null,
per pitcher and per pair type, first vs second half of each pitcher's games, and 2025 vs 2024.

## Results, 2025

### Intent predicts actual tunneling (evaluation half, frozen thresholds)

| checkpoint | designed | accidental | failed | neither | P(actual given intent) | P(actual given no intent) | lift |
|---|---|---|---|---|---|---|---|
| 40 ft | .117 | .125 | .130 | .627 | .475 | .167 | 2.85x |
| 30 ft | .123 | .118 | .126 | .633 | .495 | .157 | 3.15x |
| **23.8 ft** | .123 | .117 | .122 | .638 | .501 | .155 | **3.23x** |
| 15 ft | .117 | .118 | .111 | .654 | .512 | .153 | 3.34x |
| 5 ft | .101 | .123 | .092 | .684 | .522 | .153 | 3.42x |

Continuous: corr(ts_intent, ts_actual) = 0.31 at 50 ft falling to 0.10 at 5 ft
(`fig/intent_vs_actual_2025.png`). Unadjusted targets: lift 2.90x at 23.8 ft
(`out/classify_outcomes_2025_unadj.txt`). Target noise (`out/noise_sensitivity_2025.txt`):
adding N(0, 2 in) to every target keeps the lift at 3.04x and 91% of intent labels; N(0, 4 in)
gives 2.74x and 84%. The 3x is not a knife edge.

### Do consecutive targets tunnel more than chance? (mean ts_intent, observed minus null)

| null | 40 ft | 30 ft | 23.8 ft | 10 ft | intent plate sep obs / null |
|---|---|---|---|---|---|
| season_type | -0.041 | -0.002 | +0.016 | +0.029 | 9.41 / 10.96 in |
| season_any | -0.036 | -0.001 | +0.016 | +0.027 | 9.42 / 10.99 |
| game_type | +0.045 | +0.040 | **+0.035** | +0.020 | 9.28 / 9.55 |
| game_any | -0.011 | | +0.008 | +0.009 | 9.42 / 9.92 |
| pa_any (order shuffle) | +0.009 | | +0.005 | -0.001 | 9.49 / 9.52 |
| game_type, unadjusted targets | -0.022 | | +0.005 | +0.010 | 9.48 / 10.04 |

Null SDs are 0.0003-0.001, so every nonzero row is "significant" (p < 0.005); the point is the
size. Against season-level nulls consecutive targets are simply closer *everywhere* (plate
separation 9.4 vs 11.0 in): same-game, same-batter, same-outing target persistence, not a
corridor. Once the null is drawn from the same game the excess is +0.035 log units at 23.8 ft
(early separation 4.51 vs 4.73 in), shrinking to +0.005 for a within-PA order shuffle and
+0.005 on the unadjusted targets. For comparison the *actual* trajectories exceed the same
nulls by +0.042 (game_type) and +0.011 (pa_any). All rows in `out/null_test_2025*.txt`.

### Repeatability (excess over same-game null, split-half by each pitcher's games)

| unit | n | ts 23.8 ft raw r | excess r | Spearman-Brown | ts 40 ft excess r |
|---|---|---|---|---|---|
| pitcher, >= 100 pairs/half | 494 | 0.87 | 0.35 | 0.52 | 0.25 |
| pitcher, >= 300 pairs/half | 184 | 0.87 | 0.40 | 0.57 | 0.28 |
| pair type, >= 500 pairs/half | 55 | 0.99 | 0.87 | 0.93 | 0.91 |
| pair type, >= 2000 pairs/half | 30 | 0.99 | 0.89 | 0.94 | 0.94 |

The raw score repeats because pitch mix repeats. The excess over the null (the part that could
be called design) repeats strongly by pair type and weakly by pitcher, with a mean pitcher
excess of +0.008 log units. No per-pitcher leaderboard is published: half-season reliability
of 0.35-0.40 does not support one.

### Outcomes on pitch B (evaluation half; designed vs accidental among actual tunnels only)

Pitcher fixed effects, cluster-robust SE by pitcher, controls: pair type, count, hand matchup,
velocity, actual tunnel score and actual plate separation (so the comparison is between
visually similar tunnels), with and without zone location of pitch B.

| outcome | designed effect | with zone control | raw designed / accidental |
|---|---|---|---|
| swing% | +3.0 pp (z 6.0) | +2.6 pp (z 5.7) | 57.9 / 56.3 |
| whiff% given swing | +0.1 pp (z 0.2) | -0.1 pp | 26.0 / 24.5 |
| called strike% given take | -0.5 pp (z -1.9) | -0.6 pp (z -2.4) | 7.5 / 9.7 |
| in play% | +0.9 pp (z 2.2) | +0.7 pp (z 1.7) | 20.4 / 20.6 |

Unadjusted targets: swing +1.6 pp (z 3.5-3.8), whiff +0.2-0.5 pp (ns), called strike -0.1 (ns).
Intended tunneling regardless of execution (designed + failed vs rest, all pairs): swing +2.3 pp
(z 8.1), whiff +0.0, called strike given take -1.2 pp (z -7.7). Continuous: one unit of
`ts_intent` is worth +1.7 pp swing after controlling for `ts_actual` (+5.7 pp per unit). No
contact-quality or run-value columns exist in `pbp_info`, so those are not tested.

### Figures

- `fig/examples_2025.png`: one designed, one accidental and one failed pair, top and side view,
  solid = actual, dashed = intended, x = target, o = actual plate location.
- `fig/sweeps_2025.png`: observed-minus-null tunnel score, observed/null separation and the
  intent-to-actual lift across all checkpoints.
- `fig/intent_vs_actual_2025.png`: joint distribution of the two scores at 23.8 ft.

## 2024 replication

Same scripts on `data/2024` (context fetched with `src/fetch_pitch_context.py 2024`; the 2024
targets file is the published one). Trajectory validation median 0.002 in. Thresholds were NOT
re-selected: the 2025 exploratory-half thresholds were applied to all 451,947 2024 pairs.

| quantity | 2025 | 2024 |
|---|---|---|
| lift P(actual tunnel given intent) / P(given no intent), 23.8 ft | 3.23x (.501 / .155) | **3.27x** (.509 / .156) |
| lift at 40 ft / 10 ft | 2.85x / 3.41x | 2.83x / 3.37x |
| intent excess over same-game null, 23.8 ft | +0.035 | **+0.006** |
| intent excess over same-game null, 40 ft | +0.045 | -0.017 |
| actual excess over same-game null, 23.8 ft | +0.042 | +0.014 |
| designed vs accidental swing% (visually matched, pitcher FE) | +2.6 to +3.0 pp | **+3.1 pp** (z 8.7) |
| designed vs accidental whiff% given swing | +0.1 pp (ns) | -0.4 to -0.6 pp (ns) |
| intended tunnel (designed + failed) swing%, all pairs | +2.3 pp (z 8.1) | +2.4 pp (z 11.8) |
| per-pitcher excess, 2025 vs 2024 correlation (>= 300 pairs each) | | r = 0.17 (raw score r = 0.58) |

What replicates: the intent-to-actual lift (3.2x to 3.3x at every checkpoint), the swing
premium of designed over accidental tunnels, and the absence of a whiff premium. What does not:
the size of the "more than chance" excess in the intended scores. In 2024 the same-game null
excess is +0.006 at 23.8 ft and negative at 40 ft, matching the 2025 unadjusted-target result
(+0.005) rather than the 2025 published-target result (+0.035). Per-pitcher excess correlates
0.17 across seasons. On the pre-declared rule (do not call it a pitcher skill unless it
repeats) it is not a pitcher skill, and the "more than chance" claim itself is fragile.

## Verdict

- Tunnels are encoded in the targets: intent-level tunnel scores, built without the pitch's own
  location, predict realized tunnels at 3.2x to 3.3x across checkpoints and seasons, and the
  designed / accidental / failed split is well populated (12 / 12 / 12 / 64%).
- Designed tunnels can be separated from accidental ones, and the separation is consequential:
  designed tunnels draw about 3 pp more swings than visually matched accidental ones in both
  seasons, with no whiff difference in either.
- The hypothesis that pitchers *sequence* consecutive targets to tunnel more than their
  ordinary target distribution would produce is not supported at a useful size: excess is
  0.005 to 0.035 log units depending on null and target file, is near zero on the leakage-free
  targets and in 2024, and is not pitcher-repeatable. What is repeatable is the pair-type
  structure (which pitch after which pitch), which is a pitch-mix decision rather than a
  per-pitch aiming decision.

## Limitations

- The intended trajectory is a linear-in-target shift of a pitcher x type mean shape; it has no
  per-pitch break or release information by design (adding it would leak execution). Early
  intent separations are therefore smoother and smaller than actual ones (medians 3.8 vs 9.2 in
  at 23.8 ft), which is why thresholds are set per score and the two are never compared in inches.
- OpenCommand's published targets are shrunk toward each pitch's own ball location by up to
  about 2.5 in (median 1.15 in). That is a same-pitch leak into the intent score; the unadjusted
  file removes it and the headline lift survives (2.9x) while the same-game null excess mostly
  does not (0.035 to 0.005).
- The nulls preserve pitcher, pitch type, batter side and strikes (or game and PA) but not the
  batter's identity or the full game state; `pa_any` is the closest to a pure order shuffle.
- The 9-parameter trajectories start at y = 50 ft, about 4 ft after release; 50 ft is the
  earliest checkpoint and the noisiest.
- Split halves are by date, so the exploratory/evaluation split also carries any season drift.
- Outcomes are limited to swing, whiff, called strike and in-play from the Statcast description.
