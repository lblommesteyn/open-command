"""A firm answer on whether the intent target improves external validity.

One season of 339 pitchers gives a standard error near 0.02 on a paired
correlation difference, which cannot resolve a move of that size. This pools both
seasons as pitcher-seasons and reports three things:

  1. the paired difference in correlation, intent minus inferred, on common
     resamples, since the two correlations share the same pitchers
  2. the SEMI-PARTIAL correlation of intent with the outcome controlling for
     inferred, i.e. does intent carry outcome-relevant signal that inferred does
     not. This is the sharper test: it asks about incremental information rather
     than comparing two marginal correlations
  3. an explicit exclusion bound, so a null reads as "no larger than X" rather
     than "inconclusive"

The bootstrap resamples **pitchers**, not pitcher-seasons, because a pitcher who
appears in both seasons is one unit of evidence and resampling rows would
manufacture power that isn't there.

Reads:      both seasons via validate_intent.build_table
Writes:     artifacts/intent_validity_pooled<tag>.txt
Run:        python src/validate_intent_pooled.py [--targets targets.csv.gz]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opencommand import VALIDITY_ROWS, corr
from validate_intent import build_table

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
YEARS = ["2025", "2026"]
N_BOOT = 4000
SEED = 0


def resid(y, x):
    """y with the linear part of x removed."""
    z = np.column_stack([np.ones(len(x)), x])
    return y - z @ np.linalg.lstsq(z, y, rcond=None)[0]


def partial(target, base, y, rank, ctrl=None):
    """Partial correlation corr(y, target | base): does `target` add signal `base`
    does not have?

    Both the outcome and the challenger are residualised on the incumbent, and on
    the control column when the row has one. Run in BOTH directions below, because
    "intent adds something" only settles the question if "inferred adds something"
    does not hold equally."""
    X = np.column_stack([target, base, y] + ([] if ctrl is None else [ctrl])).astype(float)
    if rank:
        X = np.apply_along_axis(lambda c: pd.Series(c).rank().to_numpy(), 0, X)
    t, b, yy = X[:, 0], X[:, 1], X[:, 2]
    if ctrl is not None:
        c = X[:, 3]
        t, b, yy = resid(t, c), resid(b, c), resid(yy, c)
    return float(np.corrcoef(resid(t, b), resid(yy, b))[0, 1])


def main(targets="targets.csv.gz"):
    parts = []
    for y in YEARS:
        print(f"=== {y}", flush=True)
        parts.append(build_table(y, targets))
    d = pd.concat(parts, ignore_index=True)

    pitchers = d["pitcher_id"].unique()
    by_p = {p: np.flatnonzero(d["pitcher_id"].to_numpy() == p) for p in pitchers}
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(pitchers), len(pitchers))
        boots.append(np.concatenate([by_p[pitchers[i]] for i in pick]))

    tag = "" if targets == "targets.csv.gz" else "_" + targets.split(".")[0].replace("targets_", "")
    L = [f"OpenCommand intent-target validity, POOLED ({targets})",
         f"  {len(d)} pitcher-seasons over {len(pitchers)} distinct pitchers "
         f"({', '.join(f'{y}: {(d.season == y).sum()}' for y in YEARS)})",
         f"  bootstrap resamples PITCHERS ({N_BOOT} draws), so a two-season pitcher "
         f"stays one unit of evidence", ""]

    inc = d["inferred_in"].to_numpy()
    for challenger, cname in (("intent_cf_in", "intent (cross-fit)"),
                              ("intent_outing_in", "intent + outing")):
        ch = d[challenger].to_numpy()
        L += [f"  CHALLENGER: {cname}", "",
              f"  {'':<16}{'paired delta vs inferred':>30}{'partial: intent | inferred':>30}"
              f"{'partial: inferred | intent':>30}", "  " + "-" * 106]
        for label, col, ctrl in VALIDITY_ROWS:
            y = d[col].to_numpy()
            c = None if ctrl is None else d[ctrl].to_numpy()
            rank = True
            cells = []
            obs_d = corr(ch, y, rank, c) - corr(inc, y, rank, c)
            bd = [corr(ch[i], y[i], rank, None if c is None else c[i])
                  - corr(inc[i], y[i], rank, None if c is None else c[i]) for i in boots]
            lo, hi = np.percentile(bd, [2.5, 97.5])
            cells.append(f"{obs_d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

            for tgt_v, base_v in ((ch, inc), (inc, ch)):
                obs_s = partial(tgt_v, base_v, y, rank, c)
                bs = [partial(tgt_v[i], base_v[i], y[i], rank, None if c is None else c[i])
                      for i in boots]
                slo, shi = np.percentile(bs, [2.5, 97.5])
                cells.append(f"{obs_s:+.3f} [{slo:+.3f}, {shi:+.3f}]")
            L.append(f"  {label:<16}" + "".join(f"{x:>30}" for x in cells))
        L.append("")

    # exclusion bounds on the two Tom named
    L += ["  EXCLUSION BOUNDS (spearman, intent cross-fit vs inferred)",
          "  the upper end of the 95% interval is the largest improvement still "
          "compatible with the data", ""]
    for label, col, ctrl in VALIDITY_ROWS:
        if label not in ("BB%", "xERA | Stuff+"):
            continue
        y, c = d[col].to_numpy(), None if ctrl is None else d[ctrl].to_numpy()
        ch = d["intent_cf_in"].to_numpy()
        bd = [corr(ch[i], y[i], True, None if c is None else c[i])
              - corr(inc[i], y[i], True, None if c is None else c[i]) for i in boots]
        obs = corr(ch, y, True, c) - corr(inc, y, True, c)
        lo, hi = np.percentile(bd, [2.5, 97.5])
        p_up = float(np.mean(np.array(bd) <= 0))
        L.append(f"  {label:<16} delta {obs:+.3f}, 95% [{lo:+.3f}, {hi:+.3f}], "
                 f"P(delta <= 0) = {p_up:.3f}")
        L.append(f"  {'':<16} improvement larger than {hi:+.3f} is excluded")

    ART.mkdir(exist_ok=True)
    (ART / f"intent_validity_pooled{tag}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    argv = sys.argv[1:]
    tgt = argv[argv.index("--targets") + 1] if "--targets" in argv else "targets.csv.gz"
    main(tgt)
