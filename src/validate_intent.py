"""External validity of the intent target: does it correlate better with BB%,
Stuff+ and xERA | Stuff+ than the naive and inferred targets do?

Better held-out miss is necessary but not sufficient. A target can shave inches by
absorbing real command into the target and still describe pitchers worse. The
correlations are the acceptance test, so this reuses `opencommand.corr_ci` and the
same population (>= 100 scored pitches, >= 50 IP) rather than reimplementing them.

Two versions of the intent column are reported:

  intent (in-sample)   fit on the whole season, which is how the shipped inferred
                       target is fit too, so this is the like-for-like column
  intent (cross-fit)   5 folds by game; every pitch is scored by a model that never
                       saw its game. The intent model has far more parameters than a
                       per-pitcher-per-pitch-type offset, so the in-sample column
                       flatters it and this one is the honest one

Reads:      data/<year>/{targets,pbp_info,pitch_context}.csv.gz, data/fg_pitching_<year>.csv.gz
Writes:     artifacts/intent_validity_<year>.txt
Run:        python src/validate_intent.py [year=2026] [--targets targets.csv.gz]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intentlib as il
from opencommand import LEADERBOARD_MIN_N, MIN_IP, VALIDITY_ROWS, corr, corr_ci

ROOT = Path(__file__).resolve().parents[1]
DATA, ART = ROOT / "data", ROOT / "artifacts"
FOLDS = 5
N_BOOT_D = 2000


def cross_fit(df, folds=FOLDS):
    """Per-pitch intent target, each fold predicted from a model fit on the others."""
    games = np.array(sorted(df["game_pk"].unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(games)
    assign = {g: i % folds for i, g in enumerate(games)}
    fold = df["game_pk"].map(assign).to_numpy()
    tx, tz = np.empty(len(df)), np.empty(len(df))
    for k in range(folds):
        m = fold == k
        model = il.fit(df[~m], form="gain")
        tx[m], tz[m] = il.predict(model, df[m])
        print(f"  fold {k + 1}/{folds}", flush=True)
    return tx, tz


def main(year="2026", targets="targets.csv.gz"):
    base = DATA / year
    tg = pd.read_csv(base / targets)
    df = il.prepare(tg, pd.read_csv(base / "pbp_info.csv.gz"),
                    pd.read_csv(base / "pitch_context.csv.gz"))
    print(f"{len(df)} scorable pitches, {df['pitcher_id'].nunique()} pitchers", flush=True)

    cols = {}
    cols["naive_in"] = il.miss(df, df["glove_x"].to_numpy(), df["glove_z"].to_numpy())
    cols["inferred_in"] = il.miss(df, df["inferred_x_in"].to_numpy(),
                                  df["inferred_z_in"].to_numpy())
    model = il.fit(df, form="gain")
    ix, iz = il.predict(model, df)
    cols["intent_ins_in"] = il.miss(df, ix, iz)
    cx, cz = cross_fit(df)
    cols["intent_cf_in"] = il.miss(df, cx, cz)
    # A per-pitch noise reduction cannot move a per-pitcher median that is already
    # averaged over hundreds of pitches. Only a per-pitcher-varying BIAS can, so the
    # outing offset is the term with any chance of shifting these correlations.
    ox, oz = il.outing_offsets(df, df["ball_x"].to_numpy() - cx, df["ball_z"].to_numpy() - cz)
    cols["intent_outing_in"] = il.miss(df, cx + ox, cz + oz)

    per = pd.DataFrame({"pitcher_id": df["pitcher_id"].to_numpy(), **cols})
    cmd = per.groupby("pitcher_id").agg(
        n=("naive_in", "size"), **{c: (c, "median") for c in cols}).reset_index()

    fg = pd.read_csv(DATA / f"fg_pitching_{Path(year).parts[0]}.csv.gz")
    d = cmd.merge(fg[["xMLBAMID", "IP", "xERA", "BB%", "sp_stuff"]],
                  left_on="pitcher_id", right_on="xMLBAMID", how="left")
    d = d[(d["n"] >= LEADERBOARD_MIN_N) & (d["IP"] >= MIN_IP)].reset_index(drop=True)
    d = d.dropna(subset=[c for _, c, _ in VALIDITY_ROWS])

    metrics = [("naive", "naive_in"), ("inferred", "inferred_in"),
               ("intent (in-sample)", "intent_ins_in"), ("intent (cross-fit)", "intent_cf_in"),
               ("intent+outing (online)", "intent_outing_in")]
    L = [f"OpenCommand intent-target external validity - {year} ({targets})",
         f"  {len(d)} pitchers: >= {LEADERBOARD_MIN_N} scored pitches and >= {MIN_IP} IP.",
         f"  cross-fit is {FOLDS} folds by game.", ""]
    for rank, name in [(True, "SPEARMAN"), (False, "PEARSON")]:
        L += ["  " + f"{name:<16}" + "".join(f"{lab:>26}" for lab, _ in metrics),
              "  " + "-" * (16 + 26 * len(metrics))]
        for label, col, ctrl in VALIDITY_ROWS:
            cells = ["{:+.3f} [{:+.3f}, {:+.3f}]".format(
                        *corr_ci(d[m].to_numpy(), d[col].to_numpy(), rank,
                                 None if ctrl is None else d[ctrl].to_numpy()))
                     for _, m in metrics]
            L.append(f"  {label:<16}" + "".join(f"{c:>26}" for c in cells))
        L.append("")
    # Comparing two overlapping CIs is the wrong test: the two correlations share
    # the same pitchers, so bootstrap the DIFFERENCE on the same resamples.
    L += ["  PAIRED DIFFERENCE, intent (cross-fit) minus inferred, same resamples",
          "  positive = intent describes pitchers better", "",
          "  " + f"{'':<16}{'spearman delta':>28}{'pearson delta':>28}", "  " + "-" * 72]
    idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT_D, len(d)))
    for challenger in ("intent_cf_in", "intent_outing_in"):
        L.append(f"  vs inferred: {challenger}")
        for label, col, ctrl in VALIDITY_ROWS:
            y = d[col].to_numpy()
            c = None if ctrl is None else d[ctrl].to_numpy()
            a, b = d["inferred_in"].to_numpy(), d[challenger].to_numpy()
            cells = []
            for rank in (True, False):
                obs = corr(b, y, rank, c) - corr(a, y, rank, c)
                boot = [corr(b[i], y[i], rank, None if c is None else c[i])
                        - corr(a[i], y[i], rank, None if c is None else c[i]) for i in idx]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                cells.append(f"{obs:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            L.append(f"  {label:<16}" + "".join(f"{x:>28}" for x in cells))
        L.append("")

    L += ["", "  per-pitcher median miss, mean over the population:",
          "  " + "  ".join(f"{lab} {d[c].mean():.2f}" for lab, c in metrics)]

    ART.mkdir(exist_ok=True)
    tag = "" if targets == "targets.csv.gz" else "_" + targets.split(".")[0].replace("targets_", "")
    (ART / f"intent_validity_{year}{tag}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    argv = sys.argv[1:]
    year = argv[0] if argv and not argv[0].startswith("--") else "2026"
    tgt = argv[argv.index("--targets") + 1] if "--targets" in argv else "targets.csv.gz"
    main(year, tgt)
