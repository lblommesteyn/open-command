"""What should the glove gain vary over?

The per-pitcher gain was the original premise and it does not survive season scale.
This asks the other question: is the gain a property of the PITCH rather than the
man throwing it? A glove that moves for a slider may mean something different from
one that moves for a four-seam, and an arm-side target may be chased harder than a
glove-side one.

Held out by game, 5 seeds, same protocol as evaluate_intent.py.

Reads:      data/<year>/{targets,pbp_info,pitch_context}.csv.gz
Writes:     artifacts/gain_levels_<year>.txt
Run:        python src/gain_level_sweep.py [year=2025] [--targets targets.csv.gz]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intentlib as il

ROOT = Path(__file__).resolve().parents[1]
DATA, ART = ROOT / "data", ROOT / "artifacts"
SEEDS = 5
TEST_FRAC = 0.3
LEVELS = ["league", "pitcher", "pgroup", "pitch_type", "pitch_type_2k",
          "pitch_type_side", "pitch_type_cluster", "pitch_type_side_cluster",
          "pitcher_pitch_type"]


def main(year="2025", targets="targets.csv.gz"):
    base = DATA / year
    df = il.prepare(pd.read_csv(base / targets), pd.read_csv(base / "pbp_info.csv.gz"),
                    pd.read_csv(base / "pitch_context.csv.gz"))
    print(f"{len(df)} pitches, {df['pitcher_id'].nunique()} pitchers", flush=True)

    acc = {l: {"all": [], "2k": []} for l in LEVELS}
    spread = {l: [] for l in LEVELS}
    ngrp = {}
    for seed in range(SEEDS):
        games = np.array(sorted(df["game_pk"].unique()))
        rng = np.random.default_rng(seed)
        rng.shuffle(games)
        te = set(games[: int(round(len(games) * TEST_FRAC))])
        m = df["game_pk"].isin(te)
        tr, tst = df[~m], df[m]
        sub = ((tst["pgroup"].isin(["BRK", "OFF"])) & (tst["two_strike"] == 1)).to_numpy()
        for lv in LEVELS:
            model = il.fit(tr, form="gain", level=lv)
            tx, tz = il.predict(model, tst)
            mm = il.miss(tst, tx, tz)
            acc[lv]["all"].append(mm)
            acc[lv]["2k"].append(mm[sub])
            spread[lv].append(model["tau"])
            ngrp[lv] = len(model["gain"])
        print(f"  seed {seed} done", flush=True)

    L = [f"OpenCommand glove-gain level sweep - {year} ({targets})",
         f"  held out by game, {SEEDS} seeds, test_frac={TEST_FRAC}", "",
         f"  {'gain fit at':<26}{'groups':>8}{'overall':>10}{'vs league':>11}"
         f"{'2K brk/off':>12}{'tau':>8}", "  " + "-" * 75]
    base_med = np.median(np.concatenate(acc["league"]["all"]))
    for lv in LEVELS:
        a = np.median(np.concatenate(acc[lv]["all"]))
        b = np.median(np.concatenate(acc[lv]["2k"]))
        L.append(f"  {lv:<26}{ngrp[lv]:>8}{a:>10.3f}{a - base_med:>+11.3f}"
                 f"{b:>12.3f}{np.mean(spread[lv]):>8.3f}")

    ART.mkdir(exist_ok=True)
    tag = "" if targets == "targets.csv.gz" else "_" + targets.split(".")[0].replace("targets_", "")
    (ART / f"gain_levels_{year}{tag}.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    argv = sys.argv[1:]
    year = argv[0] if argv and not argv[0].startswith("--") else "2025"
    tgt = argv[argv.index("--targets") + 1] if "--targets" in argv else "targets.csv.gz"
    main(year, tgt)
