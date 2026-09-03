"""Step 4 + 6: frozen-threshold classification and outcome analysis.

Threshold selection (exploratory) uses only games in the FIRST half of the season by date; the outcome
evaluation uses only the SECOND half. Primary checkpoint 23.8 ft (literature tunnel point, ~175 ms before
the plate); sensitivity across checkpoints reported.
A pair 'tunnels' (intent or actual) if its tunnel score is >= the exploratory-half 75th percentile AND its
plate separation is >= 6 in (so two identical pitches do not count).
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from trajectory import CKPT_NAMES

EARLY = [c for c in CKPT_NAMES if c != "plate"]
PRIMARY = "y23.8"
Q = 0.75
MIN_PLATE_SEP = 6.0


def label(pr, thr_i, thr_a, c):
    ti = (pr[f"ts_intent_{c}"] >= thr_i) & (pr.plate_sep_intent >= MIN_PLATE_SEP)
    ta = (pr[f"ts_actual_{c}"] >= thr_a) & (pr.plate_sep_actual >= MIN_PLATE_SEP)
    lab = np.where(ti & ta, "designed", np.where(ta & ~ti, "accidental", np.where(ti & ~ta, "failed", "neither")))
    return pd.Series(lab, index=pr.index)


def fe_ols(df, y, x, controls, cluster):
    """Linear probability model with pitcher fixed effects (within transform), cluster-robust SE by pitcher."""
    cols = [y] + x + controls
    d = df[cols + [cluster]].dropna()
    X = pd.get_dummies(d[x + controls], drop_first=True, dtype=float)
    X = X.loc[:, X.std() > 0]
    Xd = X - X.groupby(d[cluster]).transform("mean")
    yd = d[y].astype(float) - d[y].astype(float).groupby(d[cluster]).transform("mean")
    m = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    return m, len(d)


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2025"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    pr = pd.read_parquet(f"tunnel/out/pairs_{year}{tag}.parquet")
    pr["date"] = pd.to_datetime(pr.date)
    mid = pr.date.median()
    ex, ev = pr[pr.date < mid], pr[pr.date >= mid]
    lines = [f"Classification, {year}{tag}. Exploratory half: {len(ex)} pairs before {mid.date()}; evaluation half: {len(ev)} pairs.",
             f"Tunnel = score >= exploratory {int(Q*100)}th percentile AND plate separation >= {MIN_PLATE_SEP} in."]
    thr_from = sys.argv[3] if len(sys.argv) > 3 else None
    if thr_from:  # cross-season: thresholds frozen from another season's exploratory half; evaluate on ALL pairs
        thr = pd.read_json(f"tunnel/out/thresholds_{thr_from}.json", typ="series").to_dict()
        thr = {c: tuple(v) for c, v in thr.items()}
        ev = pr
        lines.append(f"Thresholds frozen from {thr_from}; evaluating on ALL {year}{tag} pairs.")
    else:
        thr = {c: (float(ex[f"ts_intent_{c}"].quantile(Q)), float(ex[f"ts_actual_{c}"].quantile(Q))) for c in EARLY}
        pd.Series(thr).to_json(f"tunnel/out/thresholds_{year}{tag}.json")
    lines.append("\nFrozen thresholds and label shares on the evaluation half, by checkpoint:")
    lines.append("checkpoint thr_intent thr_actual | designed accidental failed neither | P(actual tunnel | intent tunnel) P(actual | no intent) lift")
    for c in EARLY:
        lab = label(ev, *thr[c], c)
        sh = lab.value_counts(normalize=True)
        pa_i = sh.get("designed", 0) / (sh.get("designed", 0) + sh.get("failed", 0))
        pa_n = sh.get("accidental", 0) / (sh.get("accidental", 0) + sh.get("neither", 0))
        lines.append(f"  {c:>6} {thr[c][0]:8.3f} {thr[c][1]:8.3f} | {sh.get('designed',0):.3f} {sh.get('accidental',0):.3f} {sh.get('failed',0):.3f} {sh.get('neither',0):.3f} | {pa_i:.3f} {pa_n:.3f} {pa_i/pa_n:.2f}x")
    # full-season labels at the primary checkpoint (thresholds from the exploratory half), written to the deliverable
    pr["label"] = label(pr, *thr[PRIMARY], PRIMARY)
    pr["eval_half"] = (pr.date >= mid) if not thr_from else True
    pr["thr_intent_primary"], pr["thr_actual_primary"] = thr[PRIMARY]
    pr.to_csv(f"tunnel/out/pitch_pair_tunneling_{year}{tag}.csv.gz", index=False)
    ev = pr[pr.eval_half].copy()
    lines.append(f"\nPrimary checkpoint {PRIMARY}: label counts, evaluation half:\n{ev.label.value_counts().to_string()}")
    lines.append("\nMost common pair types by label (evaluation half):")
    for l in ["designed", "accidental", "failed"]:
        lines.append(f"  {l}: " + ", ".join(f"{k} {v}" for k, v in ev[ev.label == l].pair_type.value_counts().head(6).items()))

    # ---- Step 6 outcomes, evaluation half only ----
    ev["whiff_given_swing"] = np.where(ev.swing_B, ev.whiff_B.astype(float), np.nan)
    ev["called_given_take"] = np.where(~ev.swing_B, ev.called_strike_B.astype(float), np.nan)
    ev["hand"] = ev.stand + ev.p_throws
    ev["ts_a"] = ev[f"ts_actual_{PRIMARY}"]
    ev["designed"] = (ev.label == "designed").astype(float)
    ev["intent_tunnel"] = ev.label.isin(["designed", "failed"]).astype(float)
    lines.append(f"\nOutcomes on pitch B, evaluation half (raw rates):")
    raw = ev.groupby("label")[["swing_B", "whiff_given_swing", "called_given_take", "inplay_B", "zone_B"]].mean().round(4)
    raw["n"] = ev.label.value_counts()
    lines.append(raw.to_string())
    lines.append("\nDesigned vs accidental (actual tunnels only), pitcher fixed effects, cluster-robust SE by pitcher.")
    lines.append("Controls: pair type, count, hand matchup, velo_B, actual tunnel score + actual plate separation (visual similarity); second row adds zone_B.")
    act = ev[ev.label.isin(["designed", "accidental"])].copy()
    base = ["pair_type", "count_B", "hand", "velo_B", "ts_a", "plate_sep_actual"]
    for y in ["swing_B", "whiff_given_swing", "called_given_take", "inplay_B"]:
        for ctrl in [base, base + ["zone_B"]]:
            m, n = fe_ols(act, y, ["designed"], ctrl, "pitcher_id")
            b, se = m.params["designed"], m.bse["designed"]
            lines.append(f"  {y:18s} {'+zone' if 'zone_B' in ctrl else '     '} n={n:6d}  designed effect = {b:+.4f} (SE {se:.4f}, z={b/se:+.2f})")
    lines.append("\nIntent tunnel (designed+failed) vs not, ALL evaluation pairs, same controls (does intended tunneling pay regardless of execution?):")
    for y in ["swing_B", "whiff_given_swing", "called_given_take", "inplay_B"]:
        m, n = fe_ols(ev, y, ["intent_tunnel"], base + ["zone_B"], "pitcher_id")
        b, se = m.params["intent_tunnel"], m.bse["intent_tunnel"]
        lines.append(f"  {y:18s} n={n:6d}  effect = {b:+.4f} (SE {se:.4f}, z={b/se:+.2f})")
    lines.append("\nContinuous version: outcome on ts_intent (primary) with ts_actual + controls + zone, all evaluation pairs:")
    ev["ts_i"] = ev[f"ts_intent_{PRIMARY}"]
    for y in ["swing_B", "whiff_given_swing", "called_given_take", "inplay_B"]:
        m, n = fe_ols(ev, y, ["ts_i", "ts_a"], ["pair_type", "count_B", "hand", "velo_B", "plate_sep_actual", "zone_B"], "pitcher_id")
        lines.append(f"  {y:18s} n={n:6d}  per unit ts_intent = {m.params['ts_i']:+.4f} (SE {m.bse['ts_i']:.4f})   per unit ts_actual = {m.params['ts_a']:+.4f} (SE {m.bse['ts_a']:.4f})")
    txt = "\n".join(lines)
    print(txt)
    open(f"tunnel/out/classify_outcomes_{year}{tag}.txt", "w").write(txt + "\n")
