"""Latent-intent target model (library; imported by intent_inference.py and
evaluate_intent.py).

The repo's step-3 target is the catcher's glove plus a `pitcher x pitch type`
offset. That assumes the glove *is* the target, up to a constant per pitch type.
Three things break that assumption in practice:

  1. Some pitchers barely use the glove as a target (they throw to their own
     spot). Others see-glove-hit-glove.
  2. Glove movement doesn't have to scale 1:1 with target movement: a catcher
     can move 6 inches and the pitcher's intent moves 3.
  3. One pitch type can have two targets. Hendricks' sinker goes to both sides
     of the plate, so a single pitch-type mean sits between the two clusters and
     is biased for every pitch.

So the model here is, per pitch, with cell `c` = (pitcher, pitch type, two-strike,
batter side, target cluster):

    intent = alpha_c + s_p * (glove - gbar_c)

`alpha_c` is where the pitch actually ends up in that cell on the training split
(the "prior spot"), `gbar_c` is where the glove sits in that cell, and `s_p` is
the pitcher's **glove gain**: how much of a glove deviation from his own norm
carries into intent. s = 1 recovers the repo's current inferred target
(cell-level); s = 0 says the glove is decoration and the cell prior is the target.
Luke's `w` from the Discord thread is `w = 1 - s`.

Both terms address (2) directly and (3) via the cluster in the cell key.

Estimating `s_p` is a regression through the origin of `(ball - alpha_c)` on
`(glove - gbar_c)`, stacking the x and z coordinates, which hands you a slope
**and** a standard error per pitcher for free. Those go into a normal-normal
shrinkage with tau^2 by method of moments, so pitchers with thin samples get
pulled to the league gain instead of chasing noise.

Two options that can be switched off for ablations:
  * `catcher_bias`: per-catcher mean (ball - glove), shrunk. Catchers differ in
    how they present, which is a glove-side measurement bias, not command.
  * `outing_loo`: per-outing (pitcher x game) glove offset, computed
    leave-one-out inside the outing. Some pitchers have the catcher move the
    glove to cancel that day's bias (Mason Miller), which is invisible to any
    season-level term. NOTE this reads other pitches from the same game, so an
    evaluation using it is an online evaluation, not a held-out-by-game one.
"""
import numpy as np
import pandas as pd

# --- cell definition -------------------------------------------------------
PITCH_GROUP = {
    "FF": "FST", "SI": "FST", "FC": "FST", "FA": "FST",
    "SL": "BRK", "ST": "BRK", "CU": "BRK", "KC": "BRK", "SV": "BRK", "CS": "BRK",
    "CH": "OFF", "FS": "OFF", "FO": "OFF", "SC": "OFF", "EP": "OFF", "KN": "OFF",
}
AXES = ("x", "z")

# --- fit constants ---------------------------------------------------------
SHRINK_K = 25.0         # pseudo-counts pulling a cell mean toward its parent cell
MIN_PITCHER_N = 50      # below this a pitcher gets the league gain, no own fit
MIN_CLUSTER_N = 80      # min pitches in a (pitcher, pitch type, side) to try 2 targets
MIN_CLUSTER_FRAC = 0.25 # smaller cluster must hold this share, else it's one target
CLUSTER_SEP_IN = 8.0    # centroid separation (inches) needed to call it two targets
CLUSTER_ASHMAN_D = 2.0  # separation in pooled sds; the classic bimodality threshold
CLUSTER_BIC_MARGIN = 10.0
CATCHER_K = 300.0       # pseudo-counts on the catcher bias
OUTING_K = 12.0         # pseudo-counts on the outing offset
GAIN_LO, GAIN_HI = -0.5, 1.5   # sanity clip on a raw per-pitcher gain


def prepare(targets, pbp, ctx):
    """Join the three sources into the one frame the model works on.

    Keeps only clips with a plausible naive target and a known pitch type, and
    labels each pitch with its cell keys."""
    t = targets[targets["plausible"] & (targets["status"] == "ok")].copy()
    info = pbp.set_index("play_id")[["date", "pitcher_id", "pitcher", "pitch_type"]]
    t = t.join(info, on="play_id")
    c = ctx.set_index("play_id")[["stand", "p_throws", "catcher", "pre_balls", "pre_strikes",
                                  "ab_number", "pitch_number"]]
    t = t.join(c, on="play_id")

    t = t.dropna(subset=["pitch_type", "pitcher_id", "stand", "naive_x_in", "naive_z_in",
                         "plate_x_in", "plate_z_in"])
    t["pgroup"] = t["pitch_type"].map(PITCH_GROUP).fillna("OFF")
    t["two_strike"] = (t["pre_strikes"].fillna(0) >= 2).astype(int)
    t["count"] = (t["pre_balls"].fillna(0).astype(int).astype(str) + "-"
                  + t["pre_strikes"].fillna(0).astype(int).astype(str))
    t["catcher"] = t["catcher"].fillna(-1).astype("int64")
    t["pitcher_id"] = t["pitcher_id"].astype("int64")
    # glove_* is the observed target, plate_* is where the ball ended up
    for ax in AXES:
        t[f"glove_{ax}"] = t[f"naive_{ax}_in"].astype(float)
        t[f"ball_{ax}"] = t[f"plate_{ax}_in"].astype(float)
    return t.reset_index(drop=True)


# --- clustering ------------------------------------------------------------
def _principal_axis(xy):
    """Unit vector along the largest-variance direction of an (n, 2) cloud."""
    v = xy - xy.mean(0)
    cov = v.T @ v
    tr, det = cov[0, 0] + cov[1, 1], cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    lam = tr / 2 + np.sqrt(max(tr * tr / 4 - det, 0.0))
    u = np.array([cov[0, 1], lam - cov[0, 0]])
    if not np.any(u):
        u = np.array([1.0, 0.0])
    return u / np.linalg.norm(u)


def _em_1d(t):
    """Two-component 1-D Gaussian mixture by EM. Returns (weights, means, sds).

    Soft assignment on purpose: hard 2-means splits a single blob every time, so
    it can't be used to *decide* whether there are two targets."""
    lo, hi = np.quantile(t, [0.15, 0.85])
    mu = np.array([lo, hi], float)
    sd = np.full(2, max(t.std(), 1e-3))
    pi = np.array([0.5, 0.5])
    for _ in range(200):
        p = pi * np.exp(-0.5 * ((t[:, None] - mu) / sd) ** 2) / sd
        tot = p.sum(1, keepdims=True)
        resp = p / np.maximum(tot, 1e-300)
        nk = resp.sum(0)
        if nk.min() < 2:
            break
        new_mu = (resp * t[:, None]).sum(0) / nk
        new_sd = np.sqrt(np.maximum((resp * (t[:, None] - new_mu) ** 2).sum(0) / nk, 1e-6))
        if np.allclose(new_mu, mu, atol=1e-4):
            mu, sd, pi = new_mu, new_sd, nk / len(t)
            break
        mu, sd, pi = new_mu, new_sd, nk / len(t)
    return pi, mu, sd


def _loglik_1d(t, pi, mu, sd):
    p = (pi * np.exp(-0.5 * ((t[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))).sum(1)
    return float(np.log(np.maximum(p, 1e-300)).sum())


def fit_clusters(df):
    """Per (pitcher, pitch type, batter side), decide whether the glove targets
    are one spot or two, and return the accepted centroids.

    Splitting is decided on the *glove*, never on the ball, so the same rule can
    assign a cluster at prediction time without seeing the outcome. A split has to
    clear all four of: enough pitches, a real minority share, 8 inches of raw
    separation, Ashman D >= 2 (the standard "these are two modes, not one wide
    blob" bar), and a BIC that actually prefers two components."""
    out = {}
    cols = ["pitcher_id", "pitch_type", "stand"]
    for key, g in df.groupby(cols, sort=False):
        if len(g) < MIN_CLUSTER_N:
            continue
        xy = g[["glove_x", "glove_z"]].to_numpy(float)
        u = _principal_axis(xy)
        t = xy @ u
        pi, mu, sd = _em_1d(t)
        if min(pi) < MIN_CLUSTER_FRAC or not np.all(np.isfinite(mu)):
            continue
        gap = abs(mu[1] - mu[0])
        ashman = np.sqrt(2) * gap / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
        if gap < CLUSTER_SEP_IN or ashman < CLUSTER_ASHMAN_D:
            continue
        n = len(t)
        ll2 = _loglik_1d(t, pi, mu, sd)
        ll1 = _loglik_1d(t, np.array([1.0]), np.array([t.mean()]),
                         np.array([max(t.std(), 1e-3)]))
        if (2 * np.log(n) - 2 * ll1) - (5 * np.log(n) - 2 * ll2) < CLUSTER_BIC_MARGIN:
            continue
        # centroids in 2-D from the hard split of the accepted mixture
        lab = (np.abs(t - mu[1]) < np.abs(t - mu[0])).astype(int)
        out[key] = np.stack([xy[lab == 0].mean(0), xy[lab == 1].mean(0)])
    return out


def assign_clusters(df, clusters):
    """Cluster id per row: 0 when the group is single-target, else nearest centroid."""
    cid = np.zeros(len(df), dtype=int)
    keys = list(zip(df["pitcher_id"], df["pitch_type"], df["stand"]))
    xy = df[["glove_x", "glove_z"]].to_numpy(float)
    idx_by_key = {}
    for i, k in enumerate(keys):
        if k in clusters:
            idx_by_key.setdefault(k, []).append(i)
    for k, idx in idx_by_key.items():
        cent = clusters[k]
        d = ((xy[idx][:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        cid[idx] = d.argmin(1)
    return cid


# --- cell means ------------------------------------------------------------
def _level_means(df, keys):
    """Mean glove and mean ball per key tuple, plus the count."""
    agg = df.groupby(keys, sort=False).agg(
        n=("ball_x", "size"), gbar_x=("glove_x", "mean"), gbar_z=("glove_z", "mean"),
        alpha_x=("ball_x", "mean"), alpha_z=("ball_z", "mean"))
    return agg


CELL_KEYS = ["pitcher_id", "pitch_type", "two_strike", "stand", "cluster"]
PARENT_KEYS = [
    ["pitcher_id", "pitch_type", "stand", "cluster"],
    ["pitcher_id", "pgroup", "stand"],
    ["pitcher_id"],
]


def fit_cells(df):
    """Cell means, shrunk up a chain of progressively coarser cells.

    Full cell is (pitcher, pitch type, two-strike, side, cluster); its parents
    drop the count, then collapse pitch type to a group, then drop everything but
    the pitcher. `SHRINK_K` pseudo-counts of the parent get mixed into each cell,
    which is what keeps thin two-strike cells from overfitting (the failure mode
    that killed the per-cell count offsets earlier)."""
    levels = [_level_means(df, CELL_KEYS)] + [_level_means(df, k) for k in PARENT_KEYS]
    # shrink from the coarsest level down
    for i in range(len(levels) - 2, -1, -1):
        child, parent = levels[i], levels[i + 1]
        pkeys = ([CELL_KEYS] + PARENT_KEYS)[i + 1]
        idx = child.index.to_frame(index=False)
        if "pgroup" in pkeys and "pgroup" not in idx:   # pgroup is a function of pitch_type
            idx["pgroup"] = idx["pitch_type"].map(PITCH_GROUP).fillna("OFF")
        par = parent.reindex(pd.MultiIndex.from_frame(idx[pkeys]) if len(pkeys) > 1
                             else pd.Index(idx[pkeys[0]], name=pkeys[0]))
        n = child["n"].to_numpy(float)
        for col in ("gbar_x", "gbar_z", "alpha_x", "alpha_z"):
            pv = par[col].to_numpy(float)
            cv = child[col].to_numpy(float)
            pv = np.where(np.isnan(pv), cv, pv)
            child[col] = (n * cv + SHRINK_K * pv) / (n + SHRINK_K)
        levels[i] = child
    league = {"gbar_x": df["glove_x"].mean(), "gbar_z": df["glove_z"].mean(),
              "alpha_x": df["ball_x"].mean(), "alpha_z": df["ball_z"].mean()}
    return {"cell": levels[0], "parents": list(zip(PARENT_KEYS, levels[1:])),
            "league": league}


def _lookup(df, cells):
    """Per-row (gbar, alpha), falling back up the cell chain for unseen cells."""
    cols = ["gbar_x", "gbar_z", "alpha_x", "alpha_z"]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    tables = [(CELL_KEYS, cells["cell"])] + cells["parents"]
    for keys, tab in tables:
        miss = out["alpha_x"].isna()
        if not miss.any():
            break
        sub = df.loc[miss, keys]
        idx = pd.MultiIndex.from_frame(sub) if len(keys) > 1 else pd.Index(sub[keys[0]])
        got = tab.reindex(idx)[cols]
        got.index = sub.index
        out.loc[miss, cols] = got.to_numpy()
    # league fallback (from training) for a pitcher never seen in training
    for col in cols:
        out[col] = out[col].fillna(cells["league"][col])
    return out


# --- nuisance offsets ------------------------------------------------------
def fit_catcher_bias(df):
    """Shrunk per-catcher mean of (ball - glove). A catcher who sets up soft or
    drifts late shifts every glove reading he's in, for every pitcher."""
    r = pd.DataFrame({"catcher": df["catcher"].to_numpy(),
                      "rx": df["ball_x"] - df["glove_x"],
                      "rz": df["ball_z"] - df["glove_z"]})
    g = r.groupby("catcher").agg(n=("rx", "size"), rx=("rx", "mean"), rz=("rz", "mean"))
    gx, gz = r["rx"].mean(), r["rz"].mean()
    for col, gm in (("rx", gx), ("rz", gz)):
        g[col] = (g["n"] * g[col] + CATCHER_K * gm) / (g["n"] + CATCHER_K) - gm
    return g[["rx", "rz"]]


def apply_catcher_bias(df, bias):
    """Glove corrected for who was catching. Unknown catchers get no correction."""
    b = bias.reindex(df["catcher"].to_numpy()).fillna(0.0)
    return (df["glove_x"].to_numpy() + b["rx"].to_numpy(),
            df["glove_z"].to_numpy() + b["rz"].to_numpy())


def prev_pitch_residual(df, resid_x, resid_z):
    """Each pitch's *previous* pitch's residual, within the same plate appearance.

    This is the handle on "microadjusts from the same glove position": a pitcher
    whose glove never moves can still be correcting his own aim pitch to pitch, and
    the only observable trace of that is whether he responds to where the last one
    actually went. Uses strictly earlier pitches, so unlike the outing offset it is
    causally clean and can sit in a held-out table."""
    o = df[["pitcher_id", "game_pk", "ab_number", "pitch_number"]].copy()
    o["rx"], o["rz"] = resid_x, resid_z
    o["_i"] = np.arange(len(o))
    o = o.sort_values(["game_pk", "ab_number", "pitch_number"], kind="stable")
    key = ["game_pk", "ab_number"]
    same_pa = o.groupby(key, sort=False)
    prev_x = same_pa["rx"].shift(1)
    prev_z = same_pa["rz"].shift(1)
    out_x = np.zeros(len(o))
    out_z = np.zeros(len(o))
    out_x[o["_i"].to_numpy()] = prev_x.fillna(0.0).to_numpy()
    out_z[o["_i"].to_numpy()] = prev_z.fillna(0.0).to_numpy()
    has = np.zeros(len(o), bool)
    has[o["_i"].to_numpy()] = prev_x.notna().to_numpy()
    return out_x, out_z, has


def fit_prev_gamma(df, resid_x, resid_z, per_pitcher=False):
    """Slope of this pitch's residual on the previous pitch's residual, through the
    origin, x and z stacked. Negative gamma = he corrects; positive = he repeats."""
    px, pz, has = prev_pitch_residual(df, resid_x, resid_z)
    d = np.concatenate([px[has], pz[has]])
    r = np.concatenate([resid_x[has], resid_z[has]])
    if not per_pitcher:
        return float((d * r).sum() / max((d * d).sum(), 1e-9))
    pid = np.concatenate([df["pitcher_id"].to_numpy()[has]] * 2)
    fr = pd.DataFrame({"pid": pid, "dd": d * d, "dr": d * r})
    g = fr.groupby("pid").sum()
    return (g["dr"] / g["dd"].replace(0, np.nan)).clip(-1.0, 1.0)


def outing_offsets(df, resid_x, resid_z):
    """Leave-one-out (pitcher x game) mean residual, shrunk. Reads the rest of the
    outing, so anything using it is an online estimate."""
    key = pd.MultiIndex.from_arrays([df["pitcher_id"], df["game_pk"]])
    out = {}
    for ax, r in (("x", resid_x), ("z", resid_z)):
        s = pd.Series(r, index=key)
        tot = s.groupby(level=[0, 1]).transform("sum")
        n = s.groupby(level=[0, 1]).transform("size")
        loo = (tot - s) / np.maximum(n - 1, 1)
        out[ax] = np.where(n > 1, loo * (n - 1) / (n - 1 + OUTING_K), 0.0)
    return out["x"], out["z"]


# --- the model -------------------------------------------------------------
def fit(df, catcher_bias=True, form="gain", prior="ball"):
    """Fit the whole thing on a training frame. `form` is "gain" (centered,
    the default) or "w" (Luke's convex form). `prior` picks what the pitcher is
    shrunk toward in the "w" form: his own ball spot or his own glove spot."""
    df = df.copy()
    bias = fit_catcher_bias(df) if catcher_bias else None
    if bias is not None:
        df["glove_x"], df["glove_z"] = apply_catcher_bias(df, bias)

    clusters = fit_clusters(df)
    df["cluster"] = assign_clusters(df, clusters)
    cells = fit_cells(df)
    ref = _lookup(df, cells)

    # regression through the origin, x and z stacked, one slope per pitcher
    if form == "gain":
        d = np.concatenate([df["glove_x"] - ref["gbar_x"], df["glove_z"] - ref["gbar_z"]])
        r = np.concatenate([df["ball_x"] - ref["alpha_x"], df["ball_z"] - ref["alpha_z"]])
    else:
        tx = ref["alpha_x"] if prior == "ball" else ref["gbar_x"]
        tz = ref["alpha_z"] if prior == "ball" else ref["gbar_z"]
        d = np.concatenate([tx - df["glove_x"], tz - df["glove_z"]])
        r = np.concatenate([df["ball_x"] - df["glove_x"], df["ball_z"] - df["glove_z"]])
    pid = np.concatenate([df["pitcher_id"].to_numpy()] * 2)

    fr = pd.DataFrame({"pid": pid, "d": d, "r": r, "dd": d * d, "dr": d * r})
    g = fr.groupby("pid").agg(n=("d", "size"), sdd=("dd", "sum"), sdr=("dr", "sum"))
    slope = g["sdr"] / g["sdd"].replace(0, np.nan)
    # residual variance -> standard error of the slope
    fr["fit"] = fr["pid"].map(slope).fillna(0.0) * fr["d"]
    sse = (fr["r"] - fr["fit"]).pow(2).groupby(fr["pid"]).sum()
    dof = (g["n"] - 1).clip(lower=1)
    var = (sse / dof) / g["sdd"].replace(0, np.nan)
    se = np.sqrt(var)

    raw = slope.clip(GAIN_LO, GAIN_HI)
    ok = g["n"] >= 2 * MIN_PITCHER_N
    mu = float(raw[ok].mean()) if ok.any() else (1.0 if form == "gain" else 0.0)
    # method-of-moments tau^2: spread of the estimates minus their own noise
    tau2 = float(max(raw[ok].var(ddof=1) - (se[ok] ** 2).mean(), 1e-4)) if ok.sum() > 1 else 1e-4
    prec_d, prec_p = 1.0 / (se ** 2), 1.0 / tau2
    shrunk = (raw * prec_d + mu * prec_p) / (prec_d + prec_p)
    shrunk = shrunk.where(ok, mu).fillna(mu)

    return {"form": form, "prior": prior, "clusters": clusters, "cells": cells,
            "catcher_bias": bias, "gain": shrunk, "gain_raw": raw, "gain_se": se,
            "gain_n": g["n"] // 2, "mu": mu, "tau": float(np.sqrt(tau2))}


def predict(model, df):
    """Intent target (x_in, z_in) for each row. Never touches the row's ball."""
    df = df.copy()
    if model["catcher_bias"] is not None:
        df["glove_x"], df["glove_z"] = apply_catcher_bias(df, model["catcher_bias"])
    df["cluster"] = assign_clusters(df, model["clusters"])
    ref = _lookup(df, model["cells"])
    s = df["pitcher_id"].map(model["gain"]).fillna(model["mu"]).to_numpy()

    if model["form"] == "gain":
        ix = ref["alpha_x"].to_numpy() + s * (df["glove_x"].to_numpy() - ref["gbar_x"].to_numpy())
        iz = ref["alpha_z"].to_numpy() + s * (df["glove_z"].to_numpy() - ref["gbar_z"].to_numpy())
    else:
        tx = (ref["alpha_x"] if model["prior"] == "ball" else ref["gbar_x"]).to_numpy()
        tz = (ref["alpha_z"] if model["prior"] == "ball" else ref["gbar_z"]).to_numpy()
        ix = (1 - s) * df["glove_x"].to_numpy() + s * tx
        iz = (1 - s) * df["glove_z"].to_numpy() + s * tz
    return ix, iz


def miss(df, tx, tz):
    """Miss distance in inches from a target to where the pitch actually went."""
    return np.hypot(df["ball_x"].to_numpy() - tx, df["ball_z"].to_numpy() - tz)
