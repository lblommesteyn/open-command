"""Held-out evaluation of the latent-intent target model.

Splits **by game** (never by pitch: two pitches from one outing share a camera
solve, a catcher, and a plate umpire, so a pitch-level split leaks), refits every
variant on the training games only, and scores median miss on the held-out games.
Repeats over seeds because a single split moved earlier results by more than the
effects being measured.

Reads:      data/<year>/{targets,pbp_info,pitch_context}.csv.gz
Writes:     artifacts/intent_eval_<year>.txt
Run:        python src/evaluate_intent.py [year=2026] [--seeds 5] [--test-frac 0.3]
                                          [--stability <other year>]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intentlib as il

ROOT = Path(__file__).resolve().parents[1]
DATA, ART = ROOT / "data", ROOT / "artifacts"
MIN_PITCHER_N_REPORT = 100


def split_by_game(df, seed, test_frac):
    games = np.array(sorted(df["game_pk"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    test = set(games[: int(round(len(games) * test_frac))])
    m = df["game_pk"].isin(test)
    return df[~m], df[m]


# --- variants: each takes (train, test) and returns held-out targets ---------
def v_naive(train, test):
    return test["glove_x"].to_numpy(), test["glove_z"].to_numpy()


def v_repo_inferred(train, test):
    """The current step-3 target: glove + pitcher x pitch-type mean residual."""
    key = ["pitcher_id", "pitch_type"]
    out = []
    for ax in il.AXES:
        r = (train[f"ball_{ax}"] - train[f"glove_{ax}"])
        off = r.groupby([train[k] for k in key]).mean()
        idx = pd.MultiIndex.from_frame(test[key])
        out.append(test[f"glove_{ax}"].to_numpy() + off.reindex(idx).fillna(r.mean()).to_numpy())
    return out[0], out[1]


def _no_cluster_fit(train, **kw):
    saved = il.MIN_CLUSTER_N
    il.MIN_CLUSTER_N = 10 ** 9      # nothing qualifies, so every cell is single-target
    try:
        return il.fit(train, **kw)
    finally:
        il.MIN_CLUSTER_N = saved


def v_gain_nocluster(train, test):
    return il.predict(_no_cluster_fit(train, form="gain"), test)


def v_gain(train, test):
    return il.predict(il.fit(train, form="gain"), test)


def v_gain_pooled(train, test):
    m = _no_cluster_fit(train, form="gain")
    m["gain"] = m["gain"] * 0 + m["mu"]
    return il.predict(m, test)


def v_gain_nocatcher(train, test):
    return il.predict(il.fit(train, form="gain", catcher_bias=False), test)


def v_gain_outing(train, test):
    """ONLINE variant: after the held-out prediction, shift each outing by the
    leave-one-out mean residual of that pitcher's *other* pitches that game.

    Some pitchers have the catcher move the glove to cancel the day's bias, which
    no season-level term can see. This reads the rest of the outing (never the
    pitch itself), so it is not a clean held-out-by-game number - it is what an
    in-game estimate would look like."""
    tx, tz = il.predict(il.fit(train, form="gain"), test)
    ox, oz = il.outing_offsets(test, test["ball_x"].to_numpy() - tx,
                               test["ball_z"].to_numpy() - tz)
    return tx + ox, tz + oz


def v_w_form(train, test):
    return il.predict(il.fit(train, form="w", prior="ball"), test)


VARIANTS = [
    ("naive glove", v_naive),
    ("repo inferred (pitcher x type offset)", v_repo_inferred),
    ("gain, pooled league s, no clusters", v_gain_pooled),
    ("gain, per-pitcher s, no clusters", v_gain_nocluster),
    ("gain, per-pitcher s, clusters", v_gain),
    ("gain, clusters, no catcher bias", v_gain_nocatcher),
    ("w form (convex, ball prior)", v_w_form),
    ("gain + outing LOO offset (ONLINE)", v_gain_outing),
]


def subsets(test):
    brk = test["pgroup"].isin(["BRK", "OFF"])
    return {
        "overall": np.ones(len(test), bool),
        "fastballs": (test["pgroup"] == "FST").to_numpy(),
        "breaking/offspeed": brk.to_numpy(),
        "2K breaking/offspeed": (brk & (test["two_strike"] == 1)).to_numpy(),
    }


def stability(year, other, min_n=400):
    """Year-over-year reliability of the per-pitcher gain.

    Within a season the shrinkage says the spread in `s` is ~4x its own standard
    error, which would put the year-to-year correlation near 0.9 if `s` were a
    stationary pitcher trait. It isn't, so this is the number that says how much
    of `tau` is trait and how much is season-specific (catcher mix, park mix,
    detector quality on that pitcher's clips)."""
    paths = [DATA / y / "pitcher_gain.csv" for y in (year, other)]
    if not all(p.exists() for p in paths):
        return []
    a, b = (pd.read_csv(p).set_index("pitcher_id") for p in paths)
    j = a[a["n"] >= min_n].join(b[b["n"] >= min_n], lsuffix="_a", rsuffix="_b", how="inner")
    if len(j) < 20:
        return []
    return ["", f"per-pitcher gain, {year} vs {other} (min {min_n} pitches each season, "
                f"n = {len(j)} pitchers)",
            f"  pearson  r(w) = {j['w_a'].corr(j['w_b']):.3f}",
            f"  spearman r(w) = {j['w_a'].corr(j['w_b'], method='spearman'):.3f}"]


def main(year="2026", seeds=5, test_frac=0.3, other_year=None):
    base = DATA / year
    df = il.prepare(pd.read_csv(base / "targets.csv.gz"),
                    pd.read_csv(base / "pbp_info.csv.gz"),
                    pd.read_csv(base / "pitch_context.csv.gz"))
    print(f"{len(df)} plausible pitches, {df['game_pk'].nunique()} games, "
          f"{df['pitcher_id'].nunique()} pitchers")

    names = [n for n, _ in VARIANTS]
    keys = list(subsets(df.head(1)))
    acc = {(n, k): [] for n in names for k in keys}

    for seed in range(seeds):
        train, test = split_by_game(df, seed, test_frac)
        sub = subsets(test)
        for name, fn in VARIANTS:
            tx, tz = fn(train, test)
            m = il.miss(test, tx, tz)
            for k, mask in sub.items():
                acc[(name, k)].append(float(np.median(m[mask])))
        print(f"  seed {seed}: " + "  ".join(
            f"{n.split(',')[0][:12]}={acc[(n, 'overall')][-1]:.2f}" for n in names), flush=True)

    lines = [f"OpenCommand latent-intent evaluation - {year}",
             f"{len(df)} plausible pitches / {df['game_pk'].nunique()} games / "
             f"{df['pitcher_id'].nunique()} pitchers",
             f"held out by game, test_frac={test_frac}, {seeds} seeds",
             "", "Median miss (inches), mean over seeds (sd)", ""]
    head = f"{'variant':40s}" + "".join(f"{k:>24s}" for k in keys)
    lines += [head, "-" * len(head)]
    for name in names:
        row = f"{name:40s}"
        for k in keys:
            v = np.array(acc[(name, k)])
            row += f"{v.mean():>18.3f} ({v.std():.3f})"
        lines.append(row)

    # full-sample fit for the per-pitcher gain table
    m = il.fit(df, form="gain")
    gain = pd.DataFrame({"s": m["gain"], "s_raw": m["gain_raw"], "se": m["gain_se"],
                         "n": m["gain_n"]})
    gain["w"] = 1 - gain["s"]
    names_map = df.drop_duplicates("pitcher_id").set_index("pitcher_id")["pitcher"]
    gain["pitcher"] = names_map.reindex(gain.index)
    gain = gain[gain["n"] >= MIN_PITCHER_N_REPORT].sort_values("w", ascending=False)
    lines += ["", f"league mean gain s = {m['mu']:.3f}   tau(s) = {m['tau']:.3f}   "
                  f"({len(m['clusters'])} two-target (pitcher, pitch type, side) groups)",
              "", "w = 1 - s.  w near 1: throws to his own spot, glove barely moves intent.",
              "            w near 0: sees glove, hits glove.", "",
              f"{'pitcher':24s}{'n':>7s}{'w':>8s}{'s':>8s}{'s_raw':>8s}{'se':>8s}"]
    for tag, part in (("--- highest w ---", gain.head(15)), ("--- lowest w ---", gain.tail(15))):
        lines.append(tag)
        for _, r in part.iterrows():
            lines.append(f"{str(r['pitcher'])[:24]:24s}{int(r['n']):>7d}{r['w']:>8.3f}"
                         f"{r['s']:>8.3f}{r['s_raw']:>8.3f}{r['se']:>8.3f}")

    if other_year:
        lines += stability(year, other_year)

    ART.mkdir(exist_ok=True)
    (ART / f"intent_eval_{year}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    argv = sys.argv[1:]
    year = argv[0] if argv and not argv[0].startswith("--") else "2026"
    seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 5
    tf = float(argv[argv.index("--test-frac") + 1]) if "--test-frac" in argv else 0.3
    # --stability <year>: needs that season's pitcher_gain.csv from intent_inference.py
    other = argv[argv.index("--stability") + 1] if "--stability" in argv else None
    main(year, seeds, tf, other)
