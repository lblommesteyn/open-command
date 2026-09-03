"""Target-inference uncertainty: add isotropic Gaussian noise (sd = sigma in) to every OpenCommand target,
refit the cross-fit intent model, rebuild pairs, and report how the intent tunnel score and the
intent->actual lift move (evaluation half, frozen primary thresholds)."""
import sys
import numpy as np
import pandas as pd
from intent_model import fit_predict
from pairs import build_pairs

year = sys.argv[1] if len(sys.argv) > 1 else "2025"
sigmas = [float(s) for s in sys.argv[2:]] or [2.0, 4.0]
base = pd.read_csv(f"tunnel/out/pitch_pair_tunneling_{year}.csv.gz", usecols=["play_id_A", "play_id_B", "eval_half", "ts_intent_y23.8", "ts_actual_y23.8", "plate_sep_intent", "plate_sep_actual", "thr_intent_primary", "thr_actual_primary"], low_memory=False)
P = pd.read_parquet(f"tunnel/out/pitches_intent_{year}.parquet")
from trajectory import CKPT_NAMES
drop = [f"{p}{a}_{c}" for p in ("i", "s") for a in ("x", "z") for c in CKPT_NAMES] + ["fold", "n_train_cell"]
P = P.drop(columns=[c for c in drop if c in P.columns])
rng = np.random.default_rng(7)
thr_i, thr_a = base.thr_intent_primary.iloc[0], base.thr_actual_primary.iloc[0]
ev = base[base.eval_half == True]
ti0 = (ev["ts_intent_y23.8"] >= thr_i) & (ev.plate_sep_intent >= 6)
ta0 = (ev["ts_actual_y23.8"] >= thr_a) & (ev.plate_sep_actual >= 6)
lines = [f"Target-noise sensitivity, {year}, evaluation half, frozen primary thresholds (23.8 ft).",
         f"  sigma 0.0: lift {ta0[ti0].mean()/ta0[~ti0].mean():.2f}x, P(actual|intent)={ta0[ti0].mean():.3f}, share intent-tunnel {ti0.mean():.3f}"]
for s in sigmas:
    Q = P.copy()
    Q["target_x"] = Q.target_x + rng.normal(0, s, len(Q))
    Q["target_z"] = Q.target_z + rng.normal(0, s, len(Q))
    Q = fit_predict(Q)
    pr = build_pairs(Q)[["play_id_A", "play_id_B", "ts_intent_y23.8", "plate_sep_intent"]]
    m = ev.merge(pr, on=["play_id_A", "play_id_B"], suffixes=("", "_n"))
    ti = (m["ts_intent_y23.8_n"] >= thr_i) & (m.plate_sep_intent_n >= 6)
    ta = (m["ts_actual_y23.8"] >= thr_a) & (m.plate_sep_actual >= 6)
    tib = (m["ts_intent_y23.8"] >= thr_i) & (m.plate_sep_intent >= 6)
    lines.append(f"  sigma {s:.1f}: lift {ta[ti].mean()/ta[~ti].mean():.2f}x, P(actual|intent)={ta[ti].mean():.3f}, share intent-tunnel {ti.mean():.3f}; "
                 f"corr(noisy, base ts_intent) {m['ts_intent_y23.8_n'].corr(m['ts_intent_y23.8']):.3f}; intent-label agreement {(ti == tib).mean():.3f}")
    del Q, pr, m
txt = "\n".join(lines); print(txt)
open(f"tunnel/out/noise_sensitivity_{year}.txt", "w").write(txt + "\n")
