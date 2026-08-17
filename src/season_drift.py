"""Is the model flat over the season, and does it survive a forward split?

Two different worries hide behind "are we flat since the beginning of the season":

  1. **Drift.** Broadcasts, camera setups and the detector's behaviour change over a
     season. If the measured miss walks, the metric is partly measuring the calendar.
  2. **Forward validity.** Held-out-by-game interleaves April and September, so the
     model always trains on games adjacent in time to the ones it scores. Fitting on
     the first N% of the season and scoring the rest is the harder, deployable test.

Reads:      data/<year>/{targets,pbp_info,pitch_context}.csv.gz
Writes:     artifacts/season_drift_<year>.txt
Run:        python src/season_drift.py [year=2026] [--targets targets.csv.gz]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intentlib as il

ROOT = Path(__file__).resolve().parents[1]
DATA, ART = ROOT / "data", ROOT / "artifacts"
CUTS = [0.4, 0.5, 0.6, 0.7]


def main(year="2026", targets="targets.csv.gz"):
    base = DATA / year
    df = il.prepare(pd.read_csv(base / targets), pd.read_csv(base / "pbp_info.csv.gz"),
                    pd.read_csv(base / "pitch_context.csv.gz"))
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    # --- 1. drift: monthly medians under one model fit on the whole season ---
    model = il.fit(df, form="gain")
    ix, iz = il.predict(model, df)
    df["m_naive"] = il.miss(df, df["glove_x"].to_numpy(), df["glove_z"].to_numpy())
    df["m_inferred"] = il.miss(df, df["inferred_x_in"].to_numpy(), df["inferred_z_in"].to_numpy())
    df["m_intent"] = il.miss(df, ix, iz)

    L = [f"OpenCommand season drift and forward validity - {year} ({targets})", "",
         "1. MONTHLY MEDIAN MISS (one model fit on the whole season)", "",
         f"  {'month':>10}{'pitches':>10}{'naive':>10}{'inferred':>10}{'intent':>10}"
         f"{'gain vs inf':>13}", "  " + "-" * 63]
    for m, g in df.groupby("month"):
        L.append(f"  {m:>10}{len(g):>10}{g['m_naive'].median():>10.3f}"
                 f"{g['m_inferred'].median():>10.3f}{g['m_intent'].median():>10.3f}"
                 f"{g['m_intent'].median() - g['m_inferred'].median():>+13.3f}")
    spread = df.groupby("month")["m_intent"].median()
    L += ["", f"  month-to-month spread of the intent median: "
              f"{spread.max() - spread.min():.3f} in "
              f"(min {spread.min():.3f} {spread.idxmin()}, max {spread.max():.3f} {spread.idxmax()})", ""]

    # --- 2. forward split: fit on the first cut of the calendar, score the rest ---
    L += ["2. FORWARD SPLIT (fit on the first X% of the season by date, score the rest)", "",
          f"  {'train':>8}{'test games':>12}{'naive':>10}{'inferred':>10}{'intent':>10}"
          f"{'gain vs inf':>13}", "  " + "-" * 63]
    dates = np.sort(df["date"].unique())
    for cut in CUTS:
        split = dates[int(len(dates) * cut)]
        tr, te = df[df["date"] < split], df[df["date"] >= split]
        if len(tr) < 5000 or len(te) < 5000:
            continue
        m = il.fit(tr, form="gain")
        tx, tz = il.predict(m, te)
        # the inferred baseline refit on the same training window, for a fair race
        off = {}
        for ax in il.AXES:
            r = (tr[f"ball_{ax}"] - tr[f"glove_{ax}"])
            off[ax] = r.groupby([tr["pitcher_id"], tr["pitch_type"]]).mean()
        idx = pd.MultiIndex.from_frame(te[["pitcher_id", "pitch_type"]])
        inf_x = te["glove_x"].to_numpy() + off["x"].reindex(idx).fillna(0.0).to_numpy()
        inf_z = te["glove_z"].to_numpy() + off["z"].reindex(idx).fillna(0.0).to_numpy()
        nv = np.median(il.miss(te, te["glove_x"].to_numpy(), te["glove_z"].to_numpy()))
        inf = np.median(il.miss(te, inf_x, inf_z))
        itn = np.median(il.miss(te, tx, tz))
        L.append(f"  {f'{cut:.0%}':>8}{te['game_pk'].nunique():>12}{nv:>10.3f}{inf:>10.3f}"
                 f"{itn:>10.3f}{itn - inf:>+13.3f}")

    ART.mkdir(exist_ok=True)
    tag = "" if targets == "targets.csv.gz" else "_" + targets.split(".")[0].replace("targets_", "")
    (ART / f"season_drift_{year}{tag}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    argv = sys.argv[1:]
    year = argv[0] if argv and not argv[0].startswith("--") else "2026"
    tgt = argv[argv.index("--targets") + 1] if "--targets" in argv else "targets.csv.gz"
    main(year, tgt)
