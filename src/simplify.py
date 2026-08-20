"""How much of the assembled target can be thrown away without losing the inches?

`assembly.py` (Tom's assembly-validation branch, vendored here unchanged) builds the
target in three parts: ball-gated k-means setup spots, a two-level glove weight found by
searching a 2D grid of medians with its sampling noise priced from an even/odd game split,
and a four-level offset with a per (pitch type x hand) prior distribution at one rung.

That is a lot of machinery for `target = glove * slope + offset`. This file takes the same
recipe apart one piece at a time and races every reduction on assembly's own harness, so a
simplification is only accepted if it ties the full model on the number Tom is judging by.

The pieces, and the candidates that drop them:

  spots          ball-gated k-means, or one setup spot per cell (the cell's mean glove)
  slope          grid search + even/odd noise pricing, or regression through the origin,
                 at league / pitch type / (pitcher, pitch type) granularity
  offset         four rungs with a pitch-type x hand prior, or cell shrunk to pitcher

Every candidate is built from assembly's own primitives so nothing but the piece under
test changes.

Reads:      data/<year>/{targets,pbp_info}.csv.gz, data/fg_pitching_<year>.csv.gz
Writes:     artifacts/simplify_<year>.txt
Run:        python src/simplify.py [year=2025] [prev_year] [--targets targets.csv.gz] [--full]
            --full adds flatness, stabilization and stickiness, which are slow.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assembly as A

ART = Path(__file__).resolve().parents[1] / "artifacts"
CELL, LOBE = A.CELL, A.LOBE
SLOPE_LO, SLOPE_HI = -0.5, 1.5      # a target that runs from the glove or chases past it is a fit artifact


# ─────────────────────────────────────────────  the pieces, in simple form

def one_spot(tr):
    """No clustering: one setup spot per cell, its mean glove."""
    m = tr.groupby(CELL)[["naive_x_in", "naive_z_in"]].mean()
    return {k: v.to_numpy()[None, :] for k, v in m.iterrows()}


def ols_slope(a, level):
    """Slope by regression through the origin on deviations from the setup spot.

    How far the ball moves off the spot per inch the glove moves off it. One line of
    algebra per group where assembly runs a grid search, and it needs no noise model
    because the standard error falls out of the same sums.

    `level` is the finest rung the slope is allowed to vary over. Every rung shrinks to the
    one above it, so a thin group reads as its parent and only a thick one keeps its own.
    "lobe" is the recipe as literally stated, a slope per (pitcher, pitch type, setup spot).

    Returns per axis a (lookup dict keyed by the level, pitcher slope, league slope) triple.
    """
    RUNGS = {"league": [], "pitch_type": [["pitch_type"]], "cell": [CELL], "lobe": [CELL, LOBE]}[level]
    out = {}
    for ax, (nv, kc, pl) in A.AXES.items():
        g = (a[nv] - a[kc]).to_numpy()
        b = (a[pl] - a[kc]).to_numpy()
        f = pd.DataFrame({"gb": g * b, "gg": g * g, "r2": b * b}, index=a.index)
        f[LOBE] = a[LOBE]
        SUMS = ["gb", "gg", "r2"]
        L = float(f.gb.sum() / max(f.gg.sum(), 1e-9))

        def fit(keys, parent):
            grp = f.groupby(keys)
            s, n = grp[SUMS].sum(), grp.size()
            est = s.gb / s.gg.clip(lower=1e-9)
            # residual variance of the ball about the fitted line, per unit of sum(g^2)
            sig2 = ((s.r2 - est * s.gb) / (n - 1).clip(lower=1)).clip(lower=1e-9)
            se2 = sig2 / s.gg.clip(lower=1e-9)
            par = pd.Series(parent, index=est.index) if np.isscalar(parent) else parent.reindex(est.index)
            return A.shrink(est, se2, par.fillna(L))[0].clip(SLOPE_LO, SLOPE_HI)

        # pitch type hangs off the league; anything keyed on the pitcher hangs off the pitcher
        wp = fit("pitcher_id", L) if level in ("cell", "lobe") else pd.Series(dtype=float)
        wc, par = pd.Series(dtype=float), None
        for k in RUNGS:
            idx = f.groupby(k).size().index
            if par is None:
                par = L if level == "pitch_type" else pd.Series(
                    idx.get_level_values(0).map(wp).to_numpy(), index=idx)
            else:                                     # this rung's parent is the rung above it
                par = pd.Series(wc.reindex(pd.MultiIndex.from_arrays(
                    [idx.get_level_values(i) for i in range(wc.index.nlevels)])).to_numpy(), index=idx)
            wc = fit(k, par)
        out[ax] = (wc, wp, L)
    return out


def simple_offsets(a, e):
    """The leftover, cell shrunk to pitcher shrunk to league. No spot rung, no group prior."""
    out = {}
    for col in ("rx", "rz"):
        g0, S = float(a[col].mean()), float(a[col].var())
        p = a.groupby("pitcher_id")[col].agg(["mean", "count"])
        pp, _ = A.shrink(p["mean"], S / p["count"], pd.Series(g0, index=p.index))
        c = a.groupby(CELL)[col].agg(["mean", "count"])
        par = pd.Series(c.index.get_level_values(0).map(pp).to_numpy(), index=c.index)
        cp, _ = A.shrink(c["mean"], S / c["count"], par)
        j = e[CELL].merge(cp.rename("v"), left_on=CELL, right_index=True, how="left")
        j = j.merge(pp.rename("vp"), left_on="pitcher_id", right_index=True, how="left")
        out[col] = np.nan_to_num(j.v.fillna(j.vp).to_numpy(), nan=g0)
    return out["rx"], out["rz"]


# ─────────────────────────────────────────────  candidate assembly

def apply_slopes(f, W):
    """target = spot + slope x (glove - spot), the slope looked up finest rung first."""
    for ax, (nv, kc, pl) in A.AXES.items():
        wc, wp, L = W[ax]
        if len(wc):
            keys = list(wc.index.names)               # whichever rung the slope was fit at
            k = f[keys[0]] if len(keys) == 1 else pd.MultiIndex.from_frame(f[keys])
            w = pd.Series(k.map(wc).to_numpy(dtype=float), index=f.index)
        else:
            w = pd.Series(np.nan, index=f.index)
        w = w.fillna(f.pitcher_id.map(wp) if len(wp) else np.nan).fillna(L).to_numpy()
        f["t" + ax] = f[kc].to_numpy() + w * (f[nv].to_numpy() - f[kc].to_numpy())
    return f


def build(spot_fn, slope_fn, offset_fn):
    """One target method from a choice of each piece. Same order as assembly.assembled."""
    def method(tr, te):
        M = spot_fn(tr)
        a, e = tr.assign(cid=A.assign(tr, M)), te.assign(cid=A.assign(te, M))
        cen = a.groupby(LOBE)[["naive_x_in", "naive_z_in"]].mean().rename(
            columns={"naive_x_in": "kx", "naive_z_in": "kz"})
        a = a.merge(cen, left_on=LOBE, right_index=True, how="left")
        e = e.merge(cen, left_on=LOBE, right_index=True, how="left")
        e["kx"], e["kz"] = e.kx.fillna(e.naive_x_in), e.kz.fillna(e.naive_z_in)
        W = slope_fn(a)
        ap = A.apply_weights if slope_fn is A.weights else apply_slopes
        a, e = ap(a, W), ap(e, W)
        a["rx"], a["rz"] = a.plate_x_in - a.tx, a.plate_z_in - a.tz
        ox, oz = offset_fn(a, e)
        return e.tx.to_numpy() + ox, e.tz.to_numpy() + oz
    return method


CANDIDATES = [
    ("naive", A.naive),
    ("fixed offset", A.fixed_offset),
    ("assembled", A.assembled),
    ("no ball-gated spots", build(one_spot, A.weights, A.offsets)),
    ("OLS slope, per cell", build(A.spots, lambda a: ols_slope(a, "cell"), A.offsets)),
    ("OLS slope, pitch type", build(A.spots, lambda a: ols_slope(a, "pitch_type"), A.offsets)),
    ("OLS slope, one league", build(A.spots, lambda a: ols_slope(a, "league"), A.offsets)),
    ("flat offset hierarchy", build(A.spots, A.weights, simple_offsets)),
    ("OLS slope, per cluster", build(A.spots, lambda a: ols_slope(a, "lobe"), A.offsets)),
    ("one spot + OLS slope", build(one_spot, lambda a: ols_slope(a, "pitch_type"), A.offsets)),
    ("simplest that could work", build(one_spot, lambda a: ols_slope(a, "pitch_type"), simple_offsets)),
]


def main():
    argv = sys.argv[1:]
    pos = [x for x in argv if not x.startswith("--")]
    year = pos[0] if pos else "2025"
    prev_year = pos[1] if len(pos) > 1 else None
    tgt = argv[argv.index("--targets") + 1] if "--targets" in argv else "targets.csv.gz"
    full = "--full" in argv

    A.METHODS = CANDIDATES
    d = A.load(year, tgt)
    prev = A.load(prev_year, tgt) if prev_year else None
    fg = pd.read_csv(A.DATA / f"fg_pitching_{year}.csv.gz")
    print(f"{year} ({tgt}): {len(d)} pitches, {d.pitcher_id.nunique()} pitchers, "
          f"{d.game_pk.nunique()} games", flush=True)

    L = [f"SIMPLIFYING THE ASSEMBLED TARGET — {year} ({tgt})", "=" * 64, ""]
    blocks = [lambda: A.v_median_miss(d), lambda: A.v_correlations(d, fg), lambda: A.v_out_of_sample(d)]
    if full:
        blocks += [lambda: A.v_flatness(d), lambda: A.v_stabilization(d),
                   lambda: A.v_stickiness(d, prev, prev_year)]
    for mk in blocks:
        block = mk()
        L += block
        print("\n".join(block), flush=True)
    ART.mkdir(exist_ok=True)
    tag = "" if tgt == "targets.csv.gz" else "_" + tgt.split(".")[0].replace("targets_", "")
    (ART / f"simplify_{year}{tag}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote artifacts/simplify_{year}{tag}.txt")


if __name__ == "__main__":
    main()
