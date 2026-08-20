"""Assembled target inference, and the six validations that judge it.

The shipped target is naive + one season offset per (pitcher, pitch type). That offset is a
single number for a cell no matter how many spots the pitcher sets up in, and it ignores
the catcher's glove entirely once the naive peak is read. The assembly replaces it with a
model in three parts, each fit on training games only:

  1 SETUP SPOTS   a pitcher's gloves for one pitch type are clustered, and a split survives
                  only if the two spots' BALL clouds separate too. A catcher who moves
                  around behind a pitcher throwing to one spot does not create two spots.
  2 GLOVE WEIGHT  how far the target moves with the glove, per axis, from a hierarchy:
                  league, then pitcher, then pitch type given pitcher. Sideways following
                  is about twice vertical following league-wide, and fastballs are followed
                  far more than splitters, so both axis and pitch type get their own
                  prior distribution rather than one league number.
  3 OFFSET        the same hierarchy on what is left over, one level deeper (per spot).

Every level shrinks toward its parent by the evidence behind it, so a cell with 40 pitches
reads close to its pitch type's league behaviour and a cell with 2000 keeps its own.

The six validations compare three targets on identical folds:

  median miss     the headline: median distance from target to actual, held out
  flatness        the same, with each pitcher's training history truncated to n pitches.
                  A method that needs a full season is useless in April.
  correlations    external validity: does the per-pitcher number rank pitchers the way
                  independent outcome data (walk rate, Stuff+, xERA) says it should?
  stabilization   split-half reliability against pitch count: how many pitches before a
                  pitcher's command number means something.
  stickiness      the same pitcher's number in one season against the next.
  out of sample   train on the first half of the season, test on the second, against a
                  random game split as the control. Drift, not just resampling.

Reads:   data/<year>/targets.csv.gz + pbp_info.csv.gz, data/fg_pitching_<year>.csv.gz
Writes:  artifacts/assembly_<year>.txt
Run:     python src/assembly.py [year=2025] [prev_year]
         The second argument, if given, is the earlier season used for stickiness.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
DATA = SRC.parent / "data"
ART = SRC.parent / "artifacts"

CELL = ["pitcher_id", "pitch_type"]
LOBE = CELL + ["cid"]
GROUP = ["pitch_type", "hand"]
AXES = {"x": ("naive_x_in", "kx", "plate_x_in"), "z": ("naive_z_in", "kz", "plate_z_in")}
WS = np.round(np.arange(0, 1.01, 0.1), 2)      # the weight grid, 0 = ignore glove, 1 = aim at it

KMAX = 5                 # most setup spots one pitcher shows for one pitch type
SEEDS = range(5)         # split-half repeats
MIN_N_PITCHER = 100      # a pitcher needs this many scored pitches to enter a per-pitcher read
MIN_IP = 50              # and this many innings to enter the external-validity table
TRUNC = [10, 25, 100, 500]


# ─────────────────────────────────────────────  1. setup spots

def _kmeans(P, k, iters=25):
    """Tiny 2D k-means with k-means++ init."""
    rng = np.random.default_rng(0)
    C = P[rng.integers(len(P))][None, :]
    for _ in range(k - 1):
        d2 = ((P[:, None, :] - C[None]) ** 2).sum(2).min(1)
        s = d2.sum()
        C = np.vstack([C, P[rng.choice(len(P), p=d2 / s) if s > 0 else rng.integers(len(P))]])
    for _ in range(iters):
        a = ((P[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        new = np.array([P[a == i].mean(0) if (a == i).any() else C[i] for i in range(k)])
        if np.allclose(new, C): break
        C = new
    return C


def _pick_k(P):
    """How many glove clusters, by BIC on a soft spherical mixture fit by EM.

    Scoring k-means' own hard assignments instead would count the assignment as evidence
    and buy the largest k on every cell, single blobs included.
    """
    best, out, n = None, None, len(P)
    for k in range(1, min(KMAX, max(n - 1, 1)) + 1):    # k <= n-1 leaves the variance a degree of freedom
        C, pi = _kmeans(P, k).astype(float), np.full(k, 1.0 / k)
        v, ll = max(float(P.var(0).mean()), 1e-6), -np.inf
        for _ in range(100):
            lr = np.log(pi + 1e-300) - ((P[:, None, :] - C[None]) ** 2).sum(2) / (2 * v) - np.log(2 * np.pi * v)
            m = lr.max(1, keepdims=True)
            r = np.exp(lr - m)
            s = r.sum(1, keepdims=True)
            new = float((np.log(s) + m).sum())
            if new - ll < 1e-4: ll = new; break
            ll, r = new, r / s
            nk = r.sum(0) + 1e-12
            pi, C = nk / n, (r.T @ P) / nk[:, None]
            v = max(float((r * ((P[:, None, :] - C[None]) ** 2).sum(2)).sum() / (2 * n)), 1e-6)
        bic = -2 * ll + 3 * k * np.log(n)
        if best is None or bic < best: best, out = bic, C
    return out


def _merge_unproven(P, B, C):
    """Drop any split the BALLS do not confirm, worst pair first.

    Two glove clusters are two intended spots only if the pitches thrown from them land in
    two places. Otherwise the catcher moved and the pitcher did not follow.
    """
    lab = ((P[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
    groups = [np.flatnonzero(lab == i) for i in range(len(C)) if (lab == i).any()]
    while len(groups) > 1:
        worst = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                bi, bj = B[groups[i]], B[groups[j]]
                nt = len(bi) + len(bj)
                if nt < 3:                        # two means and a variance need more than 5 numbers
                    gain = -np.inf
                else:
                    both = np.r_[bi, bj]
                    rss1 = ((both - both.mean(0)) ** 2).sum()
                    rss2 = ((bi - bi.mean(0)) ** 2).sum() + ((bj - bj.mean(0)) ** 2).sum()
                    gain = (rss1 - rss2) / max(rss2 / (2 * nt - 4), 1e-9) - 2 * np.log(nt)
                if gain < 0 and (worst is None or gain < worst[0]): worst = (gain, i, j)
        if worst is None: break
        _, i, j = worst
        groups[i] = np.r_[groups[i], groups[j]]; del groups[j]
    return np.array([P[g].mean(0) for g in groups])


def spots(tr):
    """Setup-spot centres per (pitcher, pitch type), clustered on gloves and ball-gated."""
    out = {}
    for key, g in tr.groupby(CELL):
        P = g[["naive_x_in", "naive_z_in"]].to_numpy()
        C = _pick_k(P)
        out[key] = _merge_unproven(P, g[["plate_x_in", "plate_z_in"]].to_numpy(), C) if len(C) > 1 else C
    return out


def assign(d, M):
    """Nearest setup spot in glove space. Cells unseen in training get spot 0."""
    cid = np.zeros(len(d), dtype=int)
    P = d[["naive_x_in", "naive_z_in"]].to_numpy()
    for key, idx in d.groupby(CELL).indices.items():
        C = M.get(key)
        if C is None or len(C) == 1: continue
        cid[idx] = ((P[idx][:, None, :] - C[None]) ** 2).sum(2).argmin(1)
    return cid


# ─────────────────────────────────────────────  shared shrinkage

def shrink(est, se2, parent):
    """Posterior toward a parent, with the spread between units estimated from the units.

    DerSimonian-Laird: each unit is weighted by its own precision, so the many thin units
    cannot vote the real spread down to zero and flatten everybody onto the parent.
    """
    if len(est) < 2: return parent.copy(), 0.0
    prec = 1 / se2
    S1, S2 = float(prec.sum()), float((prec ** 2).sum())
    Q = float((prec * (est - parent) ** 2).sum())
    tau2 = max((Q - (len(est) - 1)) / (S1 - S2 / S1), 0.0)
    return ((tau2 * est + se2 * parent) / (tau2 + se2) if tau2 > 0 else parent.copy()), tau2


def by_group(dev, se2, keys, fallback):
    """Each (pitch type x hand) gets its own prior DISTRIBUTION, centre and spread.

    Fastball following varies widely between pitchers while splitter following is uniformly
    slight, so one league spread over-shrinks the first and under-shrinks the second.
    """
    f = pd.DataFrame({"d": dev, "se2": se2, "g": keys})
    rows = {}
    for g, s in f.groupby("g"):
        prec = 1 / s.se2
        S1, S2 = float(prec.sum()), float((prec ** 2).sum())
        mu = float((s.d * prec).sum() / S1)
        tau2 = (max((float((prec * (s.d - mu) ** 2).sum()) - (len(s) - 1)) / (S1 - S2 / S1), 0.0)
                if len(s) >= 2 else fallback)
        rows[g] = (mu, tau2, 1 / S1, len(s))
    t = pd.DataFrame(rows, index=["mu", "tau2", "se_mu2", "m"]).T
    t["mu"] = np.where(t.m >= 2, shrink(t.mu, t.se_mu2, pd.Series(0.0, index=t.index))[0], 0.0)
    return t


# ─────────────────────────────────────────────  2. how far the target follows the glove

def _weight_curves(r):
    """Each unit's median miss over the weight grid, per axis, and the league's own pair.

    A one-axis curve is too shallow to locate a unit's weight, so every candidate is scored
    on the 2D miss with the other axis held at the league value.
    """
    key = [r.pitcher_id, r.pitch_type, r.cid]
    cen = {}
    for ax, (nv, kc, pl) in AXES.items():
        cen[ax] = {}
        for w in WS:
            o = r[pl] - (r[kc] + w * (r[nv] - r[kc]))
            cen[ax][w] = (o - o.groupby(key).transform("median")).to_numpy()
    G = np.array([[float(np.median(np.hypot(cen["x"][a], cen["z"][b]))) for b in WS] for a in WS])
    i, j = np.unravel_index(G.argmin(), G.shape)
    L = {"x": float(WS[i]), "z": float(WS[j])}
    out = {}
    for ax in AXES:
        base = cen["z" if ax == "x" else "x"][L["z" if ax == "x" else "x"]]
        cur = {"pitcher": [], "cell": []}
        for w in WS:
            m = pd.Series(np.hypot(cen[ax][w], base), index=r.index)
            cur["pitcher"].append(m.groupby(r.pitcher_id).median())
            cur["cell"].append(m.groupby([r.pitcher_id, r.pitch_type]).median())
        out[ax] = {k: pd.concat(v, axis=1).set_axis(WS, axis=1).idxmin(axis=1).astype(float)
                   for k, v in cur.items()}
    return L, out


def _noise(r, fit):
    """The sampling noise of a fitted weight, priced from the data rather than assumed.

    Fit every unit on even-numbered games and again on odd, and let the disagreement set
    the price: E[(w_even - w_odd)^2] = 4 * kappa / n, one kappa per level and axis.
    """
    g = np.sort(r.game_pk.unique())
    h = r.game_pk.isin(g[::2])
    A, B = fit(r[h])[1], fit(r[~h])[1]
    size = {"pitcher": r.groupby("pitcher_id").size(), "cell": r.groupby(CELL).size()}
    return {(ax, lv): float((((A[ax][lv] - B[ax][lv]) ** 2) * size[lv]).dropna().mean()) / 4
            for ax in AXES for lv in ("pitcher", "cell")}


def weights(r):
    """League, then pitcher, then pitch type given pitcher. Per axis, all shrunk."""
    L, fit = _weight_curves(r)
    kap = _noise(r, _weight_curves)
    n_p, n_c = r.groupby("pitcher_id").size(), r.groupby(CELL).size()
    hand = r.drop_duplicates("pitcher_id").set_index("pitcher_id").hand
    multi = n_c.index[n_c.index.get_level_values(0).map(n_c.groupby(level=0).size()) >= 2]
    out = {}
    for ax in AXES:
        wp, _ = shrink(fit[ax]["pitcher"], kap[(ax, "pitcher")] / n_p.reindex(fit[ax]["pitcher"].index),
                       pd.Series(L[ax], index=fit[ax]["pitcher"].index))
        wc_hat = fit[ax]["cell"]
        pid = wc_hat.index.get_level_values(0)
        par = pd.Series(pid.map(wp).to_numpy(), index=wc_hat.index)
        se2 = kap[(ax, "cell")] / n_c.reindex(wc_hat.index)
        keys = pd.MultiIndex.from_arrays([wc_hat.index.get_level_values(1), pid.map(hand)])
        m = wc_hat.index.isin(multi)
        t = by_group((wc_hat - par)[m], se2[m], keys[m], 0.0)
        mu = pd.Series(keys.map(t.mu).to_numpy(), index=wc_hat.index).fillna(0.0)
        tau2 = pd.Series(keys.map(t.tau2).to_numpy(), index=wc_hat.index).fillna(0.0)
        centre = par + mu
        wc = centre.copy()
        wc[m] = ((tau2 * wc_hat + se2 * centre) / (tau2 + se2))[m]
        out[ax] = (wc, wp, L[ax])
    return out


def apply_weights(f, W):
    """Target position = setup spot + weight x (glove - spot), looked up cell then pitcher."""
    for ax, (nv, kc, pl) in AXES.items():
        wc, wp, L = W[ax]
        w = pd.Series(pd.MultiIndex.from_frame(f[CELL]).map(wc).to_numpy(dtype=float), index=f.index)
        w = w.fillna(f.pitcher_id.map(wp)).fillna(L).to_numpy()
        f["t" + ax] = f[kc].to_numpy() + w * (f[nv].to_numpy() - f[kc].to_numpy())
    return f


# ─────────────────────────────────────────────  3. the leftover offset

def offsets(a, e):
    """League, pitcher, pitch type given pitcher, then setup spot. Per axis, all shrunk."""
    hand = a.drop_duplicates("pitcher_id").set_index("pitcher_id").hand
    out = {}
    for col in ("rx", "rz"):
        g0, S = float(a[col].mean()), float(a[col].var())
        p = a.groupby("pitcher_id")[col].agg(["mean", "count"])
        pp, _ = shrink(p["mean"], S / p["count"], pd.Series(g0, index=p.index))

        c = a.groupby(CELL)[col].agg(["mean", "count"])
        pid = c.index.get_level_values(0)
        par = pd.Series(pid.map(pp).to_numpy(), index=c.index)
        se2 = S / c["count"]
        keys = pd.MultiIndex.from_arrays([c.index.get_level_values(1), pid.map(hand)])
        t = by_group(c["mean"] - par, se2, keys, 0.0)
        mu = pd.Series(keys.map(t.mu).to_numpy(), index=c.index).fillna(0.0)
        tau2 = pd.Series(keys.map(t.tau2).to_numpy(), index=c.index).fillna(0.0)
        centre = par + mu
        cp = (tau2 * c["mean"] + se2 * centre) / (tau2 + se2)

        k = a.groupby(LOBE)[col].agg(["mean", "count"])
        kpar = pd.Series(cp.reindex(pd.MultiIndex.from_arrays(
            [k.index.get_level_values(0), k.index.get_level_values(1)])).to_numpy(), index=k.index)
        kp, _ = shrink(k["mean"], S / k["count"], kpar)

        j = e[LOBE].merge(kp.rename("v"), left_on=LOBE, right_index=True, how="left")
        j = j.merge(cp.rename("vc"), left_on=CELL, right_index=True, how="left")
        j = j.merge(pp.rename("vp"), left_on="pitcher_id", right_index=True, how="left")
        out[col] = np.nan_to_num(j.v.fillna(j.vc).fillna(j.vp).to_numpy(), nan=g0)
    return out["rx"], out["rz"]


# ─────────────────────────────────────────────  the three targets

def naive(tr, te):
    """The peak-glove target itself, nothing fitted."""
    return te.naive_x_in.to_numpy(), te.naive_z_in.to_numpy()


def fixed_offset(tr, te):
    """The shipped recipe: naive plus one mean residual per (pitcher, pitch type)."""
    o = tr.assign(ox=tr.plate_x_in - tr.naive_x_in, oz=tr.plate_z_in - tr.naive_z_in)
    m = o.groupby(CELL)[["ox", "oz"]].mean()
    j = te[CELL].merge(m, left_on=CELL, right_index=True, how="left").fillna(0.0)
    return te.naive_x_in.to_numpy() + j.ox.to_numpy(), te.naive_z_in.to_numpy() + j.oz.to_numpy()


def assembled(tr, te):
    """Setup spots, then the glove weight, then the leftover offset."""
    M = spots(tr)
    a, e = tr.assign(cid=assign(tr, M)), te.assign(cid=assign(te, M))
    cen = a.groupby(LOBE)[["naive_x_in", "naive_z_in"]].mean().rename(
        columns={"naive_x_in": "kx", "naive_z_in": "kz"})
    a = a.merge(cen, left_on=LOBE, right_index=True, how="left")
    e = e.merge(cen, left_on=LOBE, right_index=True, how="left")
    e["kx"], e["kz"] = e.kx.fillna(e.naive_x_in), e.kz.fillna(e.naive_z_in)
    W = weights(a)
    a, e = apply_weights(a, W), apply_weights(e, W)
    a["rx"], a["rz"] = a.plate_x_in - a.tx, a.plate_z_in - a.tz
    ox, oz = offsets(a, e)
    return e.tx.to_numpy() + ox, e.tz.to_numpy() + oz


METHODS = [("naive", naive), ("fixed offset", fixed_offset), ("assembled", assembled)]


# ─────────────────────────────────────────────  scoring

def folds(d, seed):
    """Split-half by GAME, both directions. Games, not pitches, so no game leaks across."""
    g = np.random.default_rng(seed).permutation(np.sort(d.game_pk.unique()))
    a, b = set(g[:len(g) // 2]), set(g[len(g) // 2:])
    for tr, te in ((a, b), (b, a)):
        yield d[d.game_pk.isin(tr)].reset_index(drop=True), d[d.game_pk.isin(te)].reset_index(drop=True)


def run(d, fn, seeds=SEEDS, cut=None):
    """Every fold through one method. Returns per-pitch miss and the pitcher it belongs to."""
    res, pid = [], []
    for s in seeds:
        for tr, te in folds(d, s):
            if cut is not None:                    # keep only each pitcher's first `cut` training pitches
                tr = tr.sort_values(["date", "play_id"]).groupby("pitcher_id").head(cut).reset_index(drop=True)
            tx, tz = fn(tr, te)
            res.append(np.hypot(te.plate_x_in.to_numpy() - tx, te.plate_z_in.to_numpy() - tz))
            pid.append(te.pitcher_id.to_numpy())
    return np.concatenate(res), np.concatenate(pid)


def per_pitcher(res, pid, floor):
    s = pd.Series(res).groupby(pid).agg(["median", "size"])
    return s[s["size"] >= floor]["median"]


# ─────────────────────────────────────────────  the six validations

def v_median_miss(d):
    """1. The headline. Median held-out miss, pooled over pitches and over pitchers."""
    L = ["MEDIAN MISS (held out, split-half by game, 5 seeds)", "-" * 64,
         f"  {'':26s}{'pooled':>10s}{'per pitcher':>14s}{'pitchers':>10s}"]
    keep = {}
    for name, fn in METHODS:
        res, pid = run(d, fn)
        pp = per_pitcher(res, pid, MIN_N_PITCHER * len(SEEDS))
        keep[name] = (res, pid)
        L.append(f"  {name:26s}{np.median(res):10.2f}{pp.median():14.2f}{len(pp):10d}")
    base = per_pitcher(*keep["fixed offset"], MIN_N_PITCHER * len(SEEDS))
    got = per_pitcher(*keep["assembled"], MIN_N_PITCHER * len(SEEDS))
    dl = (got - base).dropna()
    L.append(f"")
    L.append(f"  assembled vs fixed offset: {dl.median():+.2f} in per pitcher, "
             f"better for {(dl < 0).mean():.0%} of {len(dl)} pitchers")
    return L


def v_flatness(d):
    """2. How much history each method needs before it works."""
    L = ["", "FLATNESS (each pitcher's training truncated to his first n pitches)", "-" * 64,
         "  " + f"{'':26s}" + "".join(f"{c:>12s}" for c in [f"n={n}" for n in TRUNC] + ["full"])]
    for name, fn in METHODS:
        row = f"  {name:26s}"
        for cut in TRUNC + [None]:
            res, _ = run(d, fn, seeds=range(2), cut=cut)
            row += f"{np.median(res):12.2f}"
        L.append(row)
    L.append("")
    L.append("  naive fits nothing, so truncation cannot touch it: that row is the control.")
    return L


def _rank_corr(x, y, control=None):
    a, b = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if control is not None:
        c = pd.Series(control).rank().to_numpy()
        a = a - np.polyval(np.polyfit(c, a, 1), c)
        b = b - np.polyval(np.polyfit(c, b, 1), c)
    return float(np.corrcoef(a, b)[0, 1])


def v_correlations(d, fg):
    """3. External validity: does the number rank pitchers the way outcomes say it should?"""
    L = ["", "CORRELATIONS to independent outcome data (Spearman, whole season)", "-" * 64]
    cols = {}
    for name, fn in METHODS:
        tx, tz = fn(d, d)
        m = pd.Series(np.hypot(d.plate_x_in - tx, d.plate_z_in - tz))
        s = m.groupby(d.pitcher_id).agg(["median", "size"])
        cols[name] = s[s["size"] >= MIN_N_PITCHER]["median"]
    t = pd.DataFrame(cols).join(
        fg.set_index("xMLBAMID")[["BB%", "sp_stuff", "xERA", "IP"]], how="inner")
    t = t[t.IP >= MIN_IP]
    assert len(t) > 100, f"the Fangraphs join found only {len(t)} pitchers"
    L.append(f"  {len(t)} pitchers with {MIN_N_PITCHER}+ pitches and {MIN_IP}+ innings")
    L.append("")
    L.append(f"  {'':26s}" + "".join(f"{n:>26s}" for n, _ in METHODS))
    for lab, col, ctrl in [("BB%", "BB%", None), ("Stuff+", "sp_stuff", None),
                           ("xERA", "xERA", None), ("xERA | Stuff+", "xERA", "sp_stuff")]:
        s = t.dropna(subset=[col] + ([ctrl] if ctrl else []))
        L.append(f"  {lab:26s}" + "".join(
            f"{_rank_corr(s[n], s[col], s[ctrl] if ctrl else None):+26.3f}" for n, _ in METHODS))
    L.append("")
    L.append("  BB% is the check that matters: a better target should rank walk-prone pitchers")
    L.append("  higher, and it does so without ever seeing a walk.")
    return L


def v_stabilization(d):
    """4. How many pitches before a pitcher's number means anything."""
    L = ["", "STABILIZATION (reliability of the per-pitcher number, by pitch count)", "-" * 64,
         "  " + f"{'':26s}" + "".join(f"{b:>14s}" for b in ["100-300", "300-700", "700+"])]
    g = np.sort(d.game_pk.unique())
    h = d.game_pk.isin(g[::2])
    A, B = d[h].reset_index(drop=True), d[~h].reset_index(drop=True)
    n = d.groupby("pitcher_id").size()
    for name, fn in METHODS:
        ax, az = fn(A, A)
        bx, bz = fn(B, B)
        ma = pd.Series(np.hypot(A.plate_x_in - ax, A.plate_z_in - az)).groupby(A.pitcher_id).median()
        mb = pd.Series(np.hypot(B.plate_x_in - bx, B.plate_z_in - bz)).groupby(B.pitcher_id).median()
        j = pd.DataFrame({"a": ma, "b": mb, "n": n}).dropna()
        row = f"  {name:26s}"
        for lo, hi in ((100, 300), (300, 700), (700, 10 ** 9)):
            s = j[(j.n >= lo) & (j.n < hi)]
            r = s.a.corr(s.b) if len(s) > 5 else np.nan
            row += f"{2 * r / (1 + r):14.3f}"          # Spearman-Brown: two halves -> a whole season
        L.append(row)
    L.append("")
    L.append("  Full-season reliability implied by two half-seasons. 0.5 is the usual bar.")
    return L


def v_stickiness(d, prev, prev_year):
    """5. Does a pitcher's number carry from one season to the next?"""
    L = ["", "STICKINESS (the same pitcher, previous season against this one)", "-" * 64]
    if prev is None:
        return L + ["  skipped: no previous season given"]
    rows, n_both = {}, 0
    for name, fn in METHODS:
        s = {}
        for tag, f in (("prev", prev), ("cur", d)):
            tx, tz = fn(f, f)
            m = pd.Series(np.hypot(f.plate_x_in - tx, f.plate_z_in - tz))
            q = m.groupby(f.pitcher_id).agg(["median", "size"])
            s[tag] = q[q["size"] >= MIN_N_PITCHER]["median"]
        j = pd.DataFrame(s).dropna()
        rows[name], n_both = j.prev.corr(j.cur, method="spearman"), len(j)
    L.append(f"  {n_both} pitchers with {MIN_N_PITCHER}+ pitches in both {prev_year} and this season")
    L.append("")
    L.append(f"  {'':26s}" + "".join(f"{n:>26s}" for n, _ in METHODS))
    L.append(f"  {'year to year':26s}" + "".join(f"{rows[n]:+26.3f}" for n, _ in METHODS))
    L.append("")
    L.append("  A target that measures a real skill persists; one that fits noise does not.")
    return L


def v_out_of_sample(d):
    """6. Train on the past, test on the future, against a random split as control."""
    L = ["", "OUT OF SAMPLE (train early season, test late) against a random-split control",
         "-" * 64,
         f"  {'':26s}{'early -> late':>16s}{'random split':>16s}{'drift cost':>14s}"]
    g = d.groupby("game_pk").date.min().sort_values()
    early = set(g.index[:len(g) // 2])
    tr = d[d.game_pk.isin(early)].reset_index(drop=True)
    te = d[~d.game_pk.isin(early)].reset_index(drop=True)
    for name, fn in METHODS:
        tx, tz = fn(tr, te)
        fwd = float(np.median(np.hypot(te.plate_x_in - tx, te.plate_z_in - tz)))
        res, _ = run(d, fn, seeds=range(2))
        rnd = float(np.median(res))
        L.append(f"  {name:26s}{fwd:16.2f}{rnd:16.2f}{fwd - rnd:+14.2f}")
    L.append("")
    L.append("  Drift cost is what a method loses by predicting forward rather than")
    L.append("  interpolating. A method that fits the season's own quirks pays more.")
    return L


# ─────────────────────────────────────────────  entry point

def load(year, targets="targets.csv.gz"):
    """The public season tree, joined and filtered to scorable pitches."""
    t = pd.read_csv(DATA / year / targets)
    p = pd.read_csv(DATA / year / "pbp_info.csv.gz",
                    usecols=["game_pk", "play_id", "date", "pitcher_id", "pitcher", "pitch_type", "x0"])
    d = t[t.plausible].merge(p, on=["game_pk", "play_id"], how="inner")
    d = d.dropna(subset=["pitcher_id", "pitch_type", "plate_x_in", "naive_x_in", "x0"]).reset_index(drop=True)
    d["pitcher_id"] = d.pitcher_id.astype(int)
    # handedness is not a public column; the release side gives it cleanly (lefties release +x)
    d["hand"] = np.where(d.pitcher_id.map(d.groupby("pitcher_id").x0.median()) > 0, "L", "R")
    assert d.play_id.is_unique
    return d


def main():
    argv = sys.argv[1:]
    pos = [x for x in argv if not x.startswith("--")]
    year = pos[0] if pos else "2025"
    prev_year = pos[1] if len(pos) > 1 else None
    tgt = argv[argv.index("--targets") + 1] if "--targets" in argv else "targets.csv.gz"
    d = load(year, tgt)
    prev = load(prev_year, tgt) if prev_year else None
    fg = pd.read_csv(DATA / f"fg_pitching_{year}.csv.gz")
    print(f"{year}: {len(d)} scorable pitches, {d.pitcher_id.nunique()} pitchers, {d.game_pk.nunique()} games"
          + (f"   previous season {prev_year}: {len(prev)} pitches" if prev is not None else ""), flush=True)

    L = [f"ASSEMBLED TARGET INFERENCE — VALIDATIONS {year}", "=" * 64, ""]
    for make_block in (lambda: v_median_miss(d), lambda: v_flatness(d), lambda: v_correlations(d, fg),
                       lambda: v_stabilization(d), lambda: v_stickiness(d, prev, prev_year),
                       lambda: v_out_of_sample(d)):
        block = make_block()                 # one at a time, printed on completion, so a late
        L += block                           # failure cannot eat the finished sections
        print("\n".join(block), flush=True)
    ART.mkdir(exist_ok=True)
    tag = "" if tgt == "targets.csv.gz" else "_" + tgt.split(".")[0].replace("targets_", "")
    (ART / f"assembly_{year}{tag}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote artifacts/assembly_{year}{tag}.txt")


if __name__ == "__main__":
    main()
