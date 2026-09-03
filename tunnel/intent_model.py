"""Step 2: intent-conditioned trajectories, cross-fit by game.

Model, per pitcher x pitch type, at each checkpoint d:
    P(d) = mu_d + B_d @ (plate_xz - mu_plate)
fit by centered OLS on TRAINING-fold pitches only (plate_xz = where those pitches actually crossed),
B_d shrunk toward the pitch-type pooled within-group slope with N0 pseudo-pitches.
The intended trajectory of a held-out pitch is the same map evaluated at its OpenCommand target:
    P_intent(d) = mu_d + B_d @ (target_xz - mu_plate)
so the pitch's own plate location never enters its intended trajectory (fold = game).
Also emits P_shape(d) = same map evaluated at the pitch's actual plate location, used ONLY to validate
that the shape model reproduces trajectories (it is never used in the tunnel scores).
"""
import sys
import numpy as np
import pandas as pd
from trajectory import CKPT_NAMES

N0 = 50          # shrinkage pseudo-count for B toward pitch-type pooled slope
MIN_TRAIN = 20   # minimum training pitches for a pitcher x type cell
K = 5
XCOLS = [f"x_{c}" for c in CKPT_NAMES]
ZCOLS = [f"z_{c}" for c in CKPT_NAMES]
YCOLS = XCOLS + ZCOLS


def assign_folds(d, k=K, seed=0):
    games = d.game_pk.unique()
    rng = np.random.default_rng(seed)
    f = dict(zip(games, rng.integers(0, k, len(games))))
    return d.game_pk.map(f).values


def centered_ols(X, Y):
    """X (n,2), Y (n,m). returns mean_x, mean_y, B (2,m) with Y-my = (X-mx) @ B."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    B, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    return mx, my, B


def fit_predict(d, seed=0):
    d = d.copy()
    d["fold"] = assign_folds(d, seed=seed)
    intent = np.full((len(d), len(YCOLS)), np.nan)
    shape = np.full((len(d), len(YCOLS)), np.nan)
    ncell = np.zeros(len(d))
    P = d[["plate_x_in", "plate_z_in"]].values
    T = d[["target_x", "target_z"]].values
    Y = d[YCOLS].values
    fold = d.fold.values
    cells = [(k, g.index.values) for k, g in d.groupby(["pitcher_id", "pitch_type"])]
    for f in range(K):
        tr = d.fold.values != f
        te = ~tr
        # pooled within-group slope per pitch type (train fold), computed on group-centered data
        pooled = {}
        for pt, g in d[tr].groupby("pitch_type"):
            gi = g.index.values
            Xc = P[gi] - g.groupby(["pitcher_id"])[["plate_x_in", "plate_z_in"]].transform("mean").values
            Yc = Y[gi] - g.groupby(["pitcher_id"])[YCOLS].transform("mean").values
            Bp, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
            pooled[pt] = Bp
        for (pid, pt), gi in cells:
            if pt not in pooled:
                continue
            fo = fold[gi]
            trm = gi[fo != f]
            tem = gi[fo == f]
            n = len(trm)
            if n < MIN_TRAIN or len(tem) == 0:
                continue
            mx, my, B = centered_ols(P[trm], Y[trm])
            B = (n * B + N0 * pooled[pt]) / (n + N0)
            intent[tem] = my + (T[tem] - mx) @ B
            shape[tem] = my + (P[tem] - mx) @ B
            ncell[tem] = n
    out = pd.DataFrame(intent, columns=[f"i{c}" for c in YCOLS], index=d.index)
    sh = pd.DataFrame(shape, columns=[f"s{c}" for c in YCOLS], index=d.index)
    d = pd.concat([d, out, sh], axis=1)
    d["n_train_cell"] = ncell
    return d


def validate(d, tag):
    ok = d.n_train_cell > 0
    lines = [f"Intent model validation ({tag}): scored {ok.sum()} of {len(d)} pitches (cells with >= {MIN_TRAIN} train pitches)"]
    lines.append("checkpoint | shape-model residual median (in) [P(actual plate) vs actual] | intent residual median (in) [P(target) vs actual] | target-plate miss")
    miss = np.hypot(d.target_x - d.plate_x_in, d.target_z - d.plate_z_in)
    for c in CKPT_NAMES:
        rs = np.hypot(d[f"sx_{c}"] - d[f"x_{c}"], d[f"sz_{c}"] - d[f"z_{c}"])[ok]
        ri = np.hypot(d[f"ix_{c}"] - d[f"x_{c}"], d[f"iz_{c}"] - d[f"z_{c}"])[ok]
        lines.append(f"  {c:>6} | {np.median(rs):6.2f} | {np.median(ri):6.2f} | {np.median(miss[ok]):6.2f}")
    # how much of a plate shift shows up early: pooled slope magnitude per checkpoint
    lines.append("Held-out plate residual of intent trajectory vs target (should be ~0 by construction at plate):")
    rp = np.hypot(d.ix_plate - d.target_x, d.iz_plate - d.target_z)[ok]
    lines.append(f"  median {np.median(rp):.3f} in, p90 {np.percentile(rp,90):.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2025"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    d = pd.read_parquet(f"tunnel/out/pitches_{year}{tag}.parquet")
    d = fit_predict(d)
    txt = validate(d, f"{year}{tag}")
    print(txt)
    open(f"tunnel/out/intent_validation_{year}{tag}.txt", "w").write(txt + "\n")
    d.to_parquet(f"tunnel/out/pitches_intent_{year}{tag}.parquet")
