<div align="center">

<img src="artifacts/banner.png" alt="OpenCommand" width="100%">

[![Version](https://img.shields.io/badge/version-1.0.0-6E7681?style=for-the-badge&labelColor=24292F)](https://huggingface.co/datasets/tomdoyo/open-command)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-6E7681?style=for-the-badge&labelColor=24292F)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![GitHub](https://img.shields.io/badge/github.com%2Ftomdoyo%2Fopen--command-00852E?style=for-the-badge&labelColor=24292F)](https://github.com/tomdoyo/open-command)

[How it Works](#how-it-works) · [Data](#data) · [Topics](#topics) · [Citation](#license--citation)

</div>

> [!IMPORTANT]  
> Git history has been rewritten due to restructuring for large file support. If you have an existing clone or fork, delete it and re-clone.

## Let's Measure Command

OpenCommand scores **command** using the pitch location's distance from target.

This repo contains 2025/2026 computer vision object detections and the full inference pipeline for producing **target estimates** and resulting **command scores**.

<p align="center">
  <img src="artifacts/rogers_sinker.gif" alt="Tyler Rogers sinker">
</p>
<p align="center"><sub>Tyler Rogers dots a backdoor sinker (TB @ TOR, 2026/05/13). <b>Yellow box:</b> broadcast strikezone detection. <b>Thin white circle:</b> catcher glove detection. <b>Thick white circle:</b> glove detection projected onto strikezone plane.</sub></p>

## How it Works

### Summary
- Estimate camera position with broadcast strikezone & ball detection
- Estimate camera zoom/pan/tilt with broadcast strikezone & camera position
- Estimate glove location with camera position/zoom/pan/tilt/roll & glove detection
- Estimate target with glove location
- Estimate command with target & actual location

### Install

```
pip install -r requirements.txt          
# or:  conda env create -f environment.yml
```

### Pipeline

Every script in `src/` takes upstream CSVs and writes **one** output.  
And they're standalone: `python src/<script>.py [year=2026] ...`  
(This means you can work on a single stage by regenerating just that stage's file!)  

```
raw/gloveball_tracks  raw/strikezone_tracking
     │       │              │
     │       └──────┬───────┘
     │              ▼                            
     │  1. solve_camera_pose.py ──► camera_poses.csv.gz
     │              │                            
     └──────┬───────┘                            
            ▼                                    
  2. solve_glove_locations.py ──► glove_locations/ 
            │                                    
            ▼                                    
  3. target_inference.py ──► targets.csv.gz
            │              │
            │              └──────────┐  (+ 3a. fetch_pitch_context.py)
            ▼                         ▼
  4. opencommand.py        3b. intent_inference.py ──► intent_targets.csv.gz
     ──► command_scores.csv                     │      pitcher_gain.csv
     (+ artifacts/validations_<year>.txt)       ▼
                                       evaluate_intent.py
                                       (+ artifacts/intent_eval_<year>.txt)

```

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `solve_camera_pose.py` | gloveball_tracks, strikezone_tracking, pbp_info | `camera_poses.csv.gz` |
| 2 | `solve_glove_locations.py` | gloveball_tracks, camera_poses | `glove_locations/<game_pk>.csv.gz` |
| 3 | `target_inference.py` | glove_locations, pbp_info | `targets.csv.gz` |
| 4 | `opencommand.py` | targets, pbp_info, camera_poses, fg_pitching | `command_scores.csv` + `artifacts/validations_<year>.txt` |
| 3a | `fetch_pitch_context.py` | pbp_info (game list) | `pitch_context.csv.gz` *(network; count, side, catcher)* |
| 3b | `intent_inference.py` | targets, pbp_info, pitch_context | `intent_targets.csv.gz` + `pitcher_gain.csv` |
| — | `evaluate_intent.py` | targets, pbp_info, pitch_context | `artifacts/intent_eval_<year>.txt` |
| — | `validate_intent.py` | + fg_pitching | `artifacts/intent_validity_<year>.txt` |
| — | `season_drift.py` | targets, pbp_info, pitch_context | `artifacts/season_drift_<year>.txt` |
| — | `gain_level_sweep.py` | targets, pbp_info, pitch_context | `artifacts/gain_levels_<year>.txt` |
| — | `assembly.py` | targets, pbp_info, fg_pitching | `artifacts/assembly_<year>.txt` |
| — | `simplify.py` | targets, pbp_info, fg_pitching | `artifacts/simplify_<year>.txt` |
| — | `poselib.py` | (library, not a stage) | imported by steps 1 and 2 |
| — | `intentlib.py` | (library, not a stage) | imported by step 3b |

> Steps 3a/3b are an optional branch off step 3: the [intent target](#intent-targets-per-pitcher-glove-gain) model.

> Step 1 is particularly heavy (hours); other steps take minutes.


### In detail

#### **1. Solving camera pose** (every pitch)
- The CF camera is a fixed mount per game that pans/tilts/zooms per pitch.
- Estimate *where* the camera is:
  - Statcast's 9-parameter equation `(xyz_0, xyz_velo, xyz_acc)` gives us ball position in time (through pitch trajectory).
  - Broadcasts draw strikezone as `(17in width, sz_top/sz_bot)` at the front of the plate (middle for 2026).
  - These give us **12+** datapoints per pitch (8+ ball pixels, 4 box corners) to fit<sup>1</sup> **7** parameters: `(Cx, Cy=400`<sup>2</sup>`, Cz, pan, tilt, roll, f, t0)`.
  - Just keep the game median `Cx`/`Cz`<sup>3</sup>.
- Fit (pan, tilt, roll, f) separately with fixed `(Cx, Cy, Cz)`.
  - Use the drawn strikezone at a snapshot pre-pitch<sup>4</sup>.
  - Don't use ball positions because camera often moves mid-ball flight.

<sub><sup>1</sup> Levenberg-Marquardt on the pixel reprojection error, with a soft_l1 loss.<br>
<sup>2</sup> Camera depth (`Cy`) is degenerate against focal length (`f`): moving camera back and zooming in produce nearly the same pixels. Not a big deal down the line so `Cy` is fixed at 400.<br>
<sup>3</sup> Others are nuisance parameters.<br>
<sup>4</sup> Snapshot is taken when glove is at the highest point in the [release-2.0s, release-0.3s] window.</sub>

#### **2. Solving glove location** (every frame)
- `glove_px/pz` is a 2D projection of glove onto the camera.
- Use camera pose `(Cx, Cy, Cz, pan, tilt, roll, f)` to unproject detected `glove_px/pz` into *global* `glove_xyz`<sup>1</sup>.

<sub><sup>1</sup> Like `Cy`, glove depth (`glove_y`) is really hard to estimate. So we assume `glove_y` to be -1.75ft (median catch depth).<br>

#### **3. Inferring target with glove locations**
- Take the highest `glove_xz` in the [release-2.0s, release-0.3s] window<sup>1</sup>. This is the **naive** target.
- Many pitchers like to "start the pitch from the glove and let the ball break away from it". To account for this, add `pitcher × pitch type × season` offset. This is the **inferred target**.
  - This assumes every pitcher is **perfectly calibrated** on a pitch type level.
- Use plausibility filter<sup>2</sup> to filter out extreme targets.

<sub><sup>1</sup> Median in the ±0.05s around highest `glove_xz` frame; dispersed neighborhoods (jitter, detection flips) are skipped to the next stable candidate.<br>
<sup>2</sup> (`|x|` over 20in, `z` outside the pitch-type floor/cap)</sub>

#### **4. Scoring command**
- `miss` = distance from the actual location to target.
- For leaderboards, **median** miss is used, after plausible target filter.

## Data

### Download

The data lives on Hugging Face.

```
pip install huggingface_hub
hf download tomdoyo/open-command --repo-type dataset --local-dir data
```

That puts the tree where every script expects it. To take one file instead of all of them:

```
hf download tomdoyo/open-command 2026/command_scores.csv --repo-type dataset --local-dir data
```

### Layout

Each season lives under `data/<year>/`. 

**Keys:** `(game_pk, play_id)` identify a pitch.

Raw detections (in `data/<year>/raw/`) are produced using YOLO11 glove/ball/strikezone detector models, with postprocessing based on detection confidence.

| File (per season) | One row per | Contents |
|---|---|---|
| `pbp_info.csv.gz` | pitch | Statcast 9-parameter trajectory, `sz_top`/`sz_bot`, plate location, pitcher, pitch type, and the game_date/type/venue |
| `raw/gloveball_tracks/<game_pk>.csv.gz` | frame | glove + ball detections (pixels on screen) |
| `raw/strikezone_tracking.csv.gz` | clip | broadcast strikezone detections (pixels on screen) |
| `camera_poses.csv.gz` | clip | camera pose (+ vote diagnostics & reprojection accuracies) |
| `glove_locations/<game_pk>.csv.gz` | detection | solved glove location (real-world) |
| `targets.csv.gz` | clip | naive/inferred targets |
| `command_scores.csv` | pitcher, pitch type | n, naive and inferred median miss |

### Coverage

OpenCommand tracks nearly all the pitches that it *can*, with most clips lost being due to *no strikezone detected*<sup>1</sup> and *late center field camera cut*<sup>2</sup>.

**For 2025:** 90.09 / 92.80% possible

| Funnel loss | Clips Lost (%) | Remaining | Coverage |
|---|---:|---:|---:|
| All pitches | — | 724,005 | 100.00% |
| Clip never published | 763 (-0.11%) | 723,242 | 99.89% |
| No strikezone detected | 30,447 (-4.21%) | 692,795 | 95.69% |
| No ball release detected | 11,719 (-1.62%) | 681,076 | 94.07% |
| Late center field camera cut | 20,888 (-2.89%) | 660,188 | 91.19% |
| Low detection quality | 5,854 (-0.81%) | 654,334 | 90.38% |
| Implausible target | 2,055 (-0.28%) | **652,279** | **90.09%** |

<sub><sup>1</sup> Sometimes broadcasts don't draw a strikezone box on the screen<br>
<sup>2</sup> Sometimes camera cuts to CF-cam (i.e. pitcher-batter view) too late</sub>

## Topics

### Target maps

A nice feature of this is that you can tell where the pitcher was *trying* to throw, which is really hard just looking at the final location.

<p align="center">
  <img src="artifacts/degrom_target_map_2025_example.png" alt="Jacob deGrom inferred targets and actual four-seam locations, 2025" width="720">
</p>

### Intent targets: per-pitcher glove gain

The step-3 inferred target says *the glove is the target, up to one constant per
`pitcher × pitch type`*. Three things in the [caveats](#which-pitchers-are-represented-wellworse) break that, and all three are the same
kind of break: how much of the glove a pitcher actually uses is a **pitcher-level
parameter**, not a constant.

So `intent_inference.py` fits, per pitch, with cell `c` = (pitcher, pitch type, two-strike, batter side, target cluster):

$$\text{intent} = \alpha_c + s_p\,(\text{glove} - \bar{g}_c)$$

- `α_c` — where the pitch actually lands in that cell on the training split (the pitcher's own spot)
- `ḡ_c` — where the glove sits in that cell
- `s_p` — the pitcher's **glove gain**: how much of a glove deviation from his own norm carries into intent

`s = 1` recovers the current inferred target (at cell resolution); `s = 0` says the
glove is decoration. `w = 1 - s` is the "ignores the glove" number.

Fitting `s_p` is a regression through the origin of `(ball - α_c)` on `(glove - ḡ_c)`
with x and z stacked, which gives a slope **and** a standard error per pitcher for
free; those feed a normal-normal shrinkage with `τ²` by method of moments, so thin
samples get pulled to the league gain instead of chasing noise.

Two other terms come along:
- **target clusters.** One pitch type can have two targets — Hendricks' sinker works both sides, so a single pitch-type mean sits between the clusters and is biased on every pitch. A 1-D Gaussian mixture on the glove cloud splits a `pitcher × pitch type × side` group in two only if it clears a minority share, 8 inches of separation, Ashman *D* ≥ 2, and BIC. Splitting is decided on the **glove**, never the ball, so the same rule assigns a cluster at prediction time.
- **catcher bias.** A shrunk per-catcher mean of `(ball - glove)`. How a catcher presents is a glove-side measurement bias shared by every pitcher he catches, not command.

#### Results

Held out **by game** (two pitches from one outing share a camera solve, a catcher
and an umpire, so a pitch-level split leaks), refit on the training games only,
5 seeds. Median miss, inches.

Everything is reported twice, on the **published** targets and on an **unadjusted**
export of the same clips (raw glove, before the post-hoc detection-accuracy
adjustment). That is not a stylistic choice: see the box below for why the
published target cannot answer this question about itself.

**2026** (330,808 clips unadjusted / 326,851 published, 1,236 games, 717 pitchers):

| Target | Unadjusted | | Published | |
|---|---:|---:|---:|---:|
| | overall | 800+ pitches | overall | 800+ pitches |
| naive glove | 12.006 | 11.845 | 10.847 | 10.649 |
| inferred (`pitcher × type` offset) | 10.941 | 10.637 | 9.900 | 9.575 |
| intent, pooled league `s`, no clusters | 10.337 | 10.030 | 9.763 | 9.438 |
| intent, per-pitcher `s`, no clusters | 10.324 | 10.008 | 9.759 | 9.425 |
| **intent, per-pitcher `s` + clusters** | **10.308** | **9.989** | **9.757** | **9.422** |
| intent, no catcher bias | 10.320 | 10.004 | 9.809 | 9.473 |
| *+ outing LOO offset (online)* | *10.200* | *9.897* | *9.632* | *9.334* |

The gain is **flat across the season** and survives a forward split, which
held-out-by-game does not test since it interleaves April with September. 2025
unadjusted, monthly medians under one model: naive stays inside a 0.16 in band from
March to October and the gain over inferred never leaves −0.75 to −0.85 in. Fitting
on the first 40% of the calendar and scoring the remaining 1,394 games still gives
−0.748 in, and −0.785 in at a 70% cut (`artifacts/season_drift_2025_unadjusted.txt`).

2025 replicates every row (`artifacts/intent_eval_2025*.txt`): unadjusted
12.119 → 11.016 → 10.278, published 10.955 → 9.947 → 9.747. League `s` comes out
at **0.300 in both seasons** unadjusted, and 0.619/0.654 published.

Seed-to-seed sd is 0.02-0.05, so, on the unadjusted targets:

- Intent targets beat the current inferred target by **0.63 in overall** and **0.90 in on two-strike breaking balls**, which is where the glove-is-the-target assumption is worst. On the published targets the same model shows 0.14 in, because most of the effect has already been applied upstream.
- **Nearly all of it is the pooled `s ≈ 0.30` and the finer cell**, not the per-pitcher fit. Per-pitcher `s` is worth 0.013 in, clusters 0.016 in, catcher identity 0.012 in. That is not a sample-size artifact, and restricting to pitchers with 800+ pitches barely changes it (0.022 in); see [the per-pitcher section](#the-per-pitcher-term-does-not-survive-season-scale).
- The **outing** term is the biggest single add-on at **0.108 in**, more than the per-pitcher, cluster and catcher terms combined. That is the pitcher who has his catcher move the glove to cancel *that day's* bias: it lives entirely within an outing, so no season-level parameter can see it. It is in italics because computing it reads the pitcher's other pitches from the same game (never the pitch itself), so it is an **online** estimate, not a held-out-by-game one.

#### It buys inches and no validity

Fewer inches is necessary but not sufficient. A target can shave miss by absorbing
real command into itself and describe pitchers *worse*. The correlations in
[Some correlations](#some-correlations) are the acceptance test, so
`validate_intent.py` reuses `opencommand.corr_ci` and the same population
(≥ 100 scored pitches, ≥ 50 IP, 339 pitchers in 2025) rather than reimplementing
them. The intent column is cross-fit over 5 folds by game, because it has far more
parameters than a per-pitcher-per-pitch-type offset and an in-sample column would
flatter it.

Two correlations should never be compared through their separate CIs when they
share the same pitchers, and one season of 339 pitchers puts the standard error on
a paired difference near 0.02, which cannot resolve a move of that size. So
`validate_intent_pooled.py` pools both seasons to **566 pitcher-seasons over 410
pitchers** and bootstraps by *pitcher*, keeping a two-season pitcher as one unit of
evidence.

It reports the paired difference and, more usefully, the **partial correlation in
both directions**. The two metrics correlate about 0.95, so a marginal difference
is a poor instrument; what settles it is whether each carries outcome signal the
other lacks. Unadjusted targets, spearman:

| | paired Δ | intent \| inferred | inferred \| intent |
|---|---:|---:|---:|
| **BB%** | **+0.033** [−0.002, +0.069] | **+0.226** [+0.143, +0.305] | +0.075 [−0.013, +0.165] |
| Stuff+ | −0.001 [−0.039, +0.038] | +0.054 [−0.029, +0.139] | +0.058 [−0.023, +0.139] |
| xERA | −0.015 [−0.056, +0.023] | −0.034 [−0.116, +0.046] | +0.026 [−0.054, +0.110] |
| **xERA \| Stuff+** | −0.021 [−0.062, +0.019] | +0.001 [−0.079, +0.083] | **+0.084** [+0.003, +0.168] |

**BB% improves.** The paired delta is +0.033 with P(Δ≤0) = 0.036, and the asymmetry
is the firm part: intent carries BB% signal inferred lacks, while inferred carries
little that intent lacks.

**xERA | Stuff+ does not, and leans the other way.** Incremental information is dead
zero, and the reverse direction clears zero in *inferred's* favour. Any improvement
larger than **+0.019** is excluded.

On the **published** targets neither improves: BB% Δ = −0.001 [−0.025, +0.023] with
symmetric partials (+0.102 vs +0.110), so improvement above +0.023 is excluded.
The adjustment does not just hide the model's inches, it removes its BB% gain too.
Adding the outing offset helps neither metric on either target file, so the
within-game term is an inches lever only.

The reason the wins are this small is arithmetic. A per-pitcher median over hundreds
to thousands of pitches has already averaged away per-pitch noise, so removing more
of it barely moves that pitcher relative to anyone else. Only a **per-pitcher-varying
bias** can, which is exactly what the one historically large step was: naive →
inferred lifts BB% from +0.456 to +0.547 and *is* a per-pitcher correction. Most of
what the intent model adds is within-pitcher refinement, which sharpens every pitch
and reorders almost nobody.

If the goal is validity rather than inches, that is the design brief: look for terms
that differ **between** pitchers and are currently being charged to their command.

#### Microadjustment: pitchers repeat their miss, they don't correct it

A pitcher can work from one glove position and still be moving his own aim pitch to
pitch. The glove can't see that, so the only observable trace is whether he responds
to where his *last* pitch actually went. Fit `gamma`, the slope of this pitch's
residual on the previous pitch's residual **within the same plate appearance**,
through the origin, x and z stacked. Unlike the outing offset this uses strictly
earlier pitches, so it is causally clean and belongs in the held-out table.

`gamma = +0.066`. It is *positive*: a pitcher slightly **repeats** his last miss
rather than correcting it. Whatever is happening inside a plate appearance, it is
not a correction loop.

Held out, 2026 / 2025 unadjusted, against the 10.308 / 10.278 model:

| | 2026 | 2025 |
|---|---:|---:|
| full count in the cell instead of the 2K flag | 10.309 | 10.264 |
| + within-PA correction (global `gamma`) | 10.269 | 10.245 |
| + within-PA correction (per-pitcher `gamma`) | 10.273 | 10.251 |

Conditioning the cell on the full count is worth nothing over the two-strike flag
(and makes 2K breaking *worse* on 2026, at 11.320 against 11.263, which is the finer
cells starting to overfit). The within-PA term is worth a real but small 0.04 in,
and per-pitcher `gamma` is again slightly worse than one league number.

And it is not separate from the outing offset. Stacking them on 2026:

| | median miss | vs base |
|---|---:|---:|
| gain model | 10.310 | |
| + within-PA correction | 10.270 | −0.041 |
| + outing offset | 10.201 | −0.109 |
| + both | 10.193 | −0.117 |

The within-PA term adds 0.008 in once the outing offset is in. They are the same
persistent within-game bias measured over two window lengths, and the longer window
captures nearly all of it. **There is one within-game effect worth chasing, not two.**

#### The gain belongs to the pitch, not the pitcher

If the gain isn't a property of the man, is it a property of the pitch? A catcher
sets up on a breaking ball where he expects to *receive* it, after break, so a glove
that moves six inches on a curveball need not mean the same as one that moves six
inches on a four-seam. `gain_level_sweep.py` refits `s` at every grouping.

The fitted gains, raw targets, pitch types with 2,000+ pitches:

| pitch | `s` 2025 | `s` 2026 |
|---|---:|---:|
| FF four-seam | 0.425 | 0.409 |
| FC cutter | 0.365 | 0.359 |
| SI sinker | 0.329 | 0.320 |
| SL slider | 0.204 | 0.199 |
| ST sweeper | 0.185 | 0.180 |
| CU curve | 0.146 | 0.114 |
| KC knuckle-curve | 0.142 | 0.122 |
| FS splitter | 0.141 | 0.132 |
| CH change | 0.137 | 0.151 |

Monotone, physical, and it **replicates season to season at r = 0.994**, against
0.47 for the per-pitcher gain. The glove is worth about three times as much on a
four-seam as on a curveball or a changeup. That is the cleanest descriptive result
here: how much the glove tells you is a property of the *pitch*.

Held out, though, no grouping is an accuracy lever:

| gain fit at | groups | 2025 vs league | 2026 vs league |
|---|---:|---:|---:|
| league | 1 | +0.000 | +0.000 |
| pitcher | 696-838 | −0.013 | −0.002 |
| pitch group | 3 | −0.019 | −0.015 |
| pitch type | 16-17 | −0.017 | −0.018 |
| pitch type × cluster | 21-24 | −0.018 | −0.019 |
| pitcher × pitch type | 3.2-3.8k | −0.022 | −0.022 |

Everything lands between 0.01 and 0.02 in of one league number. Pitch-type beats
per-pitcher by 0.016 in on 2026 but only 0.004 in on 2025, so the accuracy ordering
is not stable even though the parameters are. **Fit the gain at pitch-type level
because it is the level that reproduces, not because it scores better.**

#### Simplifying the assembled target

`chain/assembly.py` on command-plus builds the target in three parts: ball-gated k-means
setup spots, a glove slope from a two-level hierarchy found by searching a 2D grid of
medians with its sampling noise priced from an even/odd game split, and a four-rung offset
carrying a per (pitch type x hand) prior distribution. It is vendored here as
`src/assembly.py`, changed only in that `load()` and `main()` take a targets filename and
the label columns are wide enough for longer method names.

**Score it on raw gloves, not on the published tree.** `assembly.py` reads
`data/<year>/targets.csv.gz`, which in command-plus is the raw-glove tree and here is the
adjusted release. The difference is not cosmetic: the adjustment pulls each glove toward
that clip's own ball, which flatters the naive baseline and compresses the gaps between
every method. Measured both ways, the assembled model beats the shipped offset by 0.62 in
per pitcher on raw gloves and by 0.15 in on the adjusted tree, and it is better for 98% of
pitchers against 78%. Everything below is raw: 659,863 pitches, 870 pitchers, 2,399 games,
split-half by game, 5 seeds.

`src/simplify.py` takes the recipe apart one piece at a time and races every reduction on
assembly's own six validations, building each candidate from assembly's own primitives so
that nothing but the piece under test changes.

| method | miss | per pitcher | n=10 | stab 100-300 | sticky | drift |
|---|---:|---:|---:|---:|---:|---:|
| naive | 12.11 | 12.20 | 12.11 | 0.542 | +0.557 | +0.02 |
| fixed offset (shipped) | 11.03 | 11.36 | 13.07 | 0.489 | +0.635 | +0.28 |
| assembled | 10.44 | 10.68 | 11.35 | 0.665 | +0.686 | +0.21 |
| **no ball-gated spots** | **10.42** | **10.64** | **11.29** | 0.653 | **+0.686** | +0.22 |
| OLS slope, per cell | 10.43 | 10.71 | 11.78 | 0.673 | +0.687 | +0.23 |
| OLS slope, pitch type | 10.41 | 10.68 | 11.26 | 0.639 | +0.679 | +0.21 |
| OLS slope, one league | 10.43 | 10.71 | 11.27 | 0.644 | +0.687 | +0.21 |
| OLS slope, per cluster | 10.44 | 10.73 | 11.92 | 0.666 | +0.684 | +0.23 |
| flat offset hierarchy | 10.52 | 10.74 | 11.70 | 0.654 | +0.653 | +0.22 |
| one spot + OLS slope | 10.41 | 10.63 | 11.16 | 0.641 | +0.653 | +0.21 |

**The ball-gated clustering is worth less than nothing.** Deleting the stage outright
scores 10.42 / 10.64 against the assembled 10.44 / 10.68, holds stickiness at +0.686
exactly, and is better at every truncation. It is 34 of the 38 seconds a fold costs, so
the single most expensive stage in the model is also the one stage that pays nothing.

**The slope is worth 0.03 in however it is fit.** Grid search with even/odd noise pricing,
closed-form regression through the origin, and league, pitch type, cell or cluster
granularity all land between 10.41 and 10.44. Fitting a slope per cluster, the recipe as
literally stated, is the worst of them per pitcher at 10.73. This is the same conclusion
the [gain-level sweep](#the-gain-belongs-to-the-pitch-not-the-pitcher) reached from the
other direction, on different data and a different estimator.

**Granularity costs history.** Truncate each pitcher's training to his first ten pitches
and a per-cell slope reads 11.78 and a per-cluster slope 11.92, against 11.27 for a slope
shared across the league. Coarse slopes are not merely as good, they are better in April,
because a thin group has nothing of its own to say.

**The offset hierarchy is the one piece that earns.** Flattening it to cell-shrunk-to-
pitcher costs 0.11 in pooled and drops stickiness from +0.686 to +0.653. It is worth more
than every slope decision put together.

**The two safe cuts are not safe together.** Dropping the clustering holds stickiness at
+0.686; dropping the grid-search slope holds it at +0.679 to +0.687. Doing both drops it
to +0.648 to +0.653, and the same happens to the flattened-offset variant. No mechanism is
offered here for the interaction; it reproduces across the variants that make both cuts,
and that is the whole claim.

So: **drop the clustering and keep the rest.** It is a net improvement on the miss, on
early-season behaviour and on per-pitcher accuracy, it preserves year-to-year persistence
to three decimal places, and it takes a fold from 38 seconds to 5. Going further to a
closed-form pitch-type slope buys the last 5 seconds and 0.13 in of April accuracy, and
costs 0.033 of stickiness; that trade is a judgement call, not something the numbers
settle.

**Do not use stabilization as a tiebreaker.** `naive` scores 0.683 in the 300-700 bucket
and 0.909 at 700+, beating every fitted method including the assembled one. A reliability
metric that ranks "fit nothing" first is reading smoothness, not skill.

The checked-in artifacts label the last row `recommended`, from the pass on the adjusted
tree where it was; the raw numbers moved the recommendation up to `no ball-gated spots`.

```bash
python src/assembly.py 2025 2026 --targets targets_unadjusted.csv.gz
python src/simplify.py 2025 2026 --full --targets targets_unadjusted.csv.gz
```


#### The per-pitcher term does not survive season scale

It is tempting to read the small per-pitcher number as a sample-size problem:
most pitchers are thin, they get shrunk to the league gain, the effect dilutes.
That is not what is happening. Sliced by how many pitches a pitcher actually threw
(2025, unadjusted, same protocol):

| pitcher season volume | pitches | pooled `s` | per-pitcher `s` | delta |
|---|---:|---:|---:|---:|
| <200 | 30,953 | 11.626 | 11.621 | −0.005 |
| 200-400 | 54,342 | 10.969 | 10.953 | −0.015 |
| 400-800 | 156,424 | 10.642 | 10.635 | −0.006 |
| 800-1500 | 300,140 | 10.316 | 10.304 | −0.012 |
| 1500+ | 448,128 | 9.987 | 9.977 | −0.011 |

It is worth about a hundredth of an inch at **every** volume, including workhorse
starters with 1,500+ pitches where the slope is estimated tightly. Giving a pitcher
his own glove weight does not measurably improve his target.

The convex form makes the same point through `τ`. On the 11-pitcher July set it
fits `τ = 0.27`; run unchanged on the full 2025 season it fits **`τ = 0.027`**, and
per-pitcher `w` beats global `w` by 0.008 in:

| 2025 unadjusted, convex form | all 870 pitchers | 11-pitcher subset |
|---|---:|---:|
| base | 11.063 | 10.258 |
| global `w` | 10.508 | 9.993 |
| per-pitcher `w` | 10.500 | 9.877 |
| fitted `τ` | **0.027** | 0.272 |

So the between-pitcher spread that looked like the headline was mostly a property
of eleven hand-picked pitchers. At season scale the pitchers who ignore the glove
are real but rare, and averaging over 870 of them leaves a population that is
described well by one number.

**What survives at season scale is the pooled blend and the cell structure**, not
the hierarchy: global `w` alone is worth 0.555 in in the convex form, and the gain
form's finer cells take it to 0.744 in with per-pitcher `s` contributing 0.010 of
that. Keep `w` as a descriptive per-pitcher statistic if it is interesting on its
own terms — Misiorowski really does come out at 0.87 — but it is not a modelling
win, and `w` correlates only 0.47 season to season even unadjusted.

> [!IMPORTANT]
> **The published target has already been moved toward the pitch, and it moves `s`.**
> The README notes above that "glove detections get post-hoc adjustments based on
> detection accuracies". That adjustment is per clip and sized from that clip's own
> measured miss, so `naive_x_in` / `naive_z_in` are not raw glove observations: they
> are glove observations pulled toward the ball, a median of **1.15 in** and capped
> around 2.5 in. Shrinking the glove toward where the pitch went is *the same
> operation `s` performs*, so fitting `s` on published targets is fitting it on a
> glove that is already part ball. Measured against the unadjusted export, that
> roughly halves the apparent glove weight (`s` = 0.65/0.62 published vs **0.300**
> unadjusted, both seasons), shrinks the model's own gain from 0.63 in to 0.14 in,
> and drags `w`'s year-over-year correlation from 0.47 down to 0.32. Splitting by
> game does not protect against any of it, because the leak is inside the row.
> **Fit `w` on unadjusted targets or not at all.**

And the ranking is the interesting output on its own. Some familiar names (2026;
the raw top and bottom 15 are in the artifact, and are mostly low-workload
relievers at both ends):

| High `w` (own spot) | `w` | | Low `w` (see glove, hit glove) | `w` |
|---|---:|---|---|---:|
| Jacob Misiorowski | 0.52 | | Aaron Nola | 0.10 |
| Jacob deGrom | 0.56 | | Paul Skenes | 0.11 |
| Trevor Rogers | 0.55 | | Nathan Eovaldi | 0.11 |
| Shohei Ohtani | 0.44 | | Jhoan Duran | 0.21 |

Misiorowski, the standing example of a pitcher who doesn't look at the glove, comes
out ~4 standard errors above the league mean without being told anything about him.

> [!NOTE]
> `s` is attenuated by glove **detection** noise the same way any errors-in-variables
> slope is, so the absolute level of `s` is not interpretable — only the ranking is.
> The `w` form `intent = (1-w)·glove + w·prior` is also implemented (`--form w`) and
> scores worse (10.106), because it doesn't carry the cell offset on the glove term.

#### The adjustment, isolated

An earlier version of this model, fit in July on a 7,341-pitch / 11-pitcher export
that predates the adjustment, used the convex form `intent = (1-w)·glove + w·prior`
with a glove-based cell prior. Running **that same code**, same 11 pitchers, same
half/half-by-game protocol, is a clean A/B on the target file alone:

| | base | global `w` | per-pitcher `w` | fitted `w` |
|---|---:|---:|---:|---:|
| July export (7,341 pitches) | 10.25 | 10.02 | 9.93 | 0.47 |
| published 2026, same 11 | 9.250 | 9.270 | 9.279 | 0.10 |
| **unadjusted 2026, same 11** | **10.258** | **9.993** | **9.877** | **0.44** |

On published targets the ladder is flat and `w` collapses to 0.10. On the
unadjusted export of the *same clips* it comes back and lands within 0.05 in of
the July numbers on every rung, with `τ = 0.272` against July's 0.268. The effect
was never absent; the target had absorbed it.

Per-pitcher `w` agreement with the July fit also runs **0.79 pearson** on unadjusted
targets versus 0.55 on published: Misiorowski 0.87 vs 0.84, Mason Miller 0.77 vs
0.72, Skenes 0.52 vs 0.47, with Yamamoto (0.61 vs 0.23) the one real disagreement.

Note the per-pitcher rung beats the global one by 0.12 in on these eleven, which
does **not** generalise: see the next section.

Run it:

```
python src/fetch_pitch_context.py 2026     # count/side/catcher per play_id, ~20 min
python src/intent_inference.py 2026
python src/evaluate_intent.py 2026 --seeds 5 --stability 2025

# against an unadjusted-glove export dropped in as data/2026/targets_unadjusted.csv.gz
python src/intent_inference.py 2026 --targets targets_unadjusted.csv.gz
python src/evaluate_intent.py 2026 --targets targets_unadjusted.csv.gz --stability 2025
```

The unadjusted export is not part of the public release; the numbers above come
from one supplied by the maintainer. Everything on the published targets
reproduces from the Hugging Face data as-is.

### Command distribution

Couple notes:
- Inferring targets (pitcher × pitch-type offset) shaves off about **1 inch** off of naive miss. 
- MLB pitchers miss by 9-11 inches
- Pitchers command fastballs ~1 inch better
- The miss distribution has some right skew

**2025, naive median miss** — min. 50 pitches

| Pitch type | Pitchers | Min | p10 | p25 | Median | p75 | p90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All pitches | 716 | 8.39 | 9.85 | 10.42 | 11.06 | 11.88 | 12.75 | 15.94 |
| Four-seam (FF) | 581 | 7.33 | 8.38 | 9.22 | 10.07 | 10.88 | 11.81 | 14.22 |
| Sinker (SI) | 380 | 6.10 | 8.50 | 9.18 | 9.92 | 10.95 | 12.14 | 18.34 |
| Cutter (FC) | 215 | 6.84 | 8.42 | 9.04 | 9.82 | 10.72 | 11.71 | 15.90 |
| Slider (SL) | 393 | 7.31 | 9.68 | 10.54 | 11.53 | 13.11 | 14.32 | 20.56 |
| Sweeper (ST) | 230 | 8.66 | 10.01 | 10.92 | 11.95 | 13.38 | 14.51 | 19.65 |
| Curveball (CU+KC) | 248 | 8.32 | 10.82 | 11.79 | 13.07 | 14.60 | 16.10 | 21.30 |
| Changeup (CH) | 301 | 8.26 | 10.48 | 11.59 | 12.78 | 14.42 | 16.48 | 28.97 |
| Splitter (FS) | 95 | 7.59 | 10.65 | 12.11 | 13.70 | 15.73 | 17.28 | 22.26 |

**2025, inferred median miss** — naive + pitcher × pitch-type offset

| Pitch type | Pitchers | Min | p10 | p25 | Median | p75 | p90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All pitches | 716 | 7.37 | 8.91 | 9.43 | 9.97 | 10.56 | 11.13 | 14.54 |
| Four-seam (FF) | 581 | 6.77 | 8.17 | 8.70 | 9.39 | 10.13 | 10.87 | 12.76 |
| Sinker (SI) | 380 | 6.09 | 7.99 | 8.49 | 9.11 | 9.88 | 10.65 | 13.40 |
| Cutter (FC) | 215 | 6.87 | 8.09 | 8.64 | 9.37 | 10.12 | 10.77 | 13.28 |
| Slider (SL) | 393 | 6.97 | 8.97 | 9.68 | 10.39 | 11.25 | 12.05 | 14.60 |
| Sweeper (ST) | 230 | 8.11 | 9.39 | 9.88 | 10.66 | 11.51 | 12.46 | 15.01 |
| Curveball (CU+KC) | 248 | 8.37 | 9.84 | 10.62 | 11.40 | 12.37 | 13.37 | 17.17 |
| Changeup (CH) | 301 | 7.20 | 9.07 | 9.78 | 10.56 | 11.32 | 12.04 | 14.85 |
| Splitter (FS) | 95 | 7.41 | 9.30 | 9.96 | 10.99 | 12.06 | 13.12 | 15.56 |

### Some correlations

**2025** — 339 pitchers, min. 50 innings

| | Naive | Inferred |
|---|---:|---:|
| BB% | +0.456 [+0.366, +0.541] | +0.547 [+0.465, +0.626] |
| Stuff+ | +0.198 [+0.096, +0.297] | +0.251 [+0.147, +0.345] |
| xERA | -0.071 [-0.180, +0.034] | -0.069 [-0.175, +0.039] |
| xERA \| Stuff+ | +0.085 [-0.022, +0.190] | +0.137 [+0.029, +0.243] |

In particular, we can see a strong correlation between command and walk rates.
<p align="center">
  <img src="artifacts/bb_vs_command_2025.png" alt="2025 inferred median miss against walk rate, 339 pitchers" width="380">
</p>

#### **Why (~~mean~~) median miss?** 
- Median is more robust to extreme values (e.g. due to bad inferred targets/glove detections/etc.)
- Median (50th percentile) better answers "what's pitcher x's *typical* miss?". A pitcher can't miss by less than 0 in, but can spike one and get a 100 inch miss, which takes 100 pitches with 1 inch above mean miss to make up for it. So coloquially, median makes more sense as an "average".

#### **How accurate is OpenCommand at measuring command?**
- This is really hard to tell because there's no *ground truth* (unless we ask "hey where did you aim?" every pitch).
- True median miss for **fastballs** is probably [7 to 10 inches](https://x.com/tomdoyo/status/2082066794404294671?s=20).
- Glove detections get post-hoc adjustments based on detection accuracies. This adjustment makes miss distances unbiased, but doesn't remove the pitch-level variance. 
- Inferred miss assumes every pitcher perfectly calibrates his pitches, but most pitchers are probably an inch or two off. At the same time, most pitchers fine tune their targets (beyond the catcher's glove) every pitch, depending on the situation. Perhaps these two cancel off on a season-level. 
- So, on a season-level, OpenCommand has a good chance of being accurate within <1 inch. On a pitch-level, certainly not. 

#### **Which pitchers are represented well/worse?**
- Some pitchers see-glove-hit-glove (especially ones that throw down the middle). These pitchers have the best OpenCommand representations.
- Some pitchers do the **opposite** of micro-adjustment. They make their catchers adjust the glove depending on their miss patterns that day so that their end locations stay the same.
- Some pitchers **don't look at the glove at all** (e.g. Misiorowski). These pitchers are heavily misrepresented. 

## License & citation

Everything in this repository, data and code, is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): use it, build on it, publish with it, with **attribution** (cite OpenCommand; see [CITATION.cff](CITATION.cff)) and **not commercially**. 

Data derived from MLB broadcast video and Statcast public feeds. MLB and Statcast are trademarks of MLB Advanced Media, L.P.; this project is not affiliated with MLB.
