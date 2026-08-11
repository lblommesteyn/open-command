"""Step 3 of the OpenCommand pipeline.

Notes:
 - Target detection takes the glove peak
    1. Window: glove detections in [release - 2.0 s, release - 0.3 s].
    2. Peak: highest glove location whose ±0.05 s neighborhood is tight
        - neighborhood spread ≤ SUPPORT_IN inches

Reads:      data/<year>/glove_locations/<game_pk>.csv.gz +
            data/<year>/pbp_info.csv.gz (pitch type for the screen, pitcher + actual
            location for the offset)
Writes:     data/<year>/targets.csv.gz, one row per posed clip, both target pairs;
            `status` is "ok" or "no target"
Run:        python src/target_inference.py [year=2026]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

WINDOW_S = 2.0              # target search window length before release
END_BEFORE_RELEASE_S = 0.3  # window end: release - 0.3 s (catch-lock-safe)
PEAK_HALF_S = 0.05          # neighborhood half-width around a peak candidate
SUPPORT_IN = 4.0            # max neighborhood spread (inches) for a stable peak


def select_target(g):
    """Finds the targeting peak in one clip's glove-location rows (see module docstring).
    Returns a dict. status and release_s are always set; the target fields are set
    only when status is "ok"."""
    fps, release_s = float(g["fps"].iloc[0]), float(g["release_s"].iloc[0])
    out = {"release_s": release_s}
    lo = (release_s - WINDOW_S) * fps
    hi = (release_s - END_BEFORE_RELEASE_S) * fps
    # between() is False on NaN, so a glove-less clip's one NaN sentinel row drops here
    win = g[g["frame_idx"].between(lo, hi)]

    # world-space peak: highest glove first, first tight neighborhood wins
    frames = win["frame_idx"].to_numpy()
    x_in, z_in = win["x_in"].to_numpy(), win["z_in"].to_numpy()
    for i in np.argsort(-z_in):
        near = np.abs(frames - frames[i]) <= PEAK_HALF_S * fps
        if max(np.ptp(x_in[near]), np.ptp(z_in[near])) <= SUPPORT_IN:
            return {**out, "status": "ok", "target_frame": int(frames[i]),
                    "naive_x_in": float(np.median(x_in[near])),
                    "naive_z_in": float(np.median(z_in[near]))}
    return {**out, "status": "no target"}


def targets_for_game(job):
    """One game's glove-location file → per-clip target rows (see select_target)."""
    f, info_rows = job
    rows = []
    for play_id, g in pd.read_csv(f, float_precision="round_trip").groupby("play_id", sort=False):
        t = select_target(g)
        p = info_rows[play_id]
        rows.append({"game_pk": int(g["game_pk"].iloc[0]), "play_id": play_id, "park": p["home_team"],
                     "y_depth_ft": float(g["y_depth_ft"].iloc[0]),  # travels with the glove rows
                     "plate_x_in": p["plate_x"] * 12, "plate_z_in": p["plate_z"] * 12, **t})
    return rows


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2026"
    base = DATA / year
    pbp = pd.read_csv(base / "pbp_info.csv.gz")
    info = pbp.set_index("play_id")[["game_pk", "home_team", "pitch_type", "pitcher_id", "plate_x", "plate_z"]]

    fields = info[["game_pk", "home_team", "plate_x", "plate_z"]]
    by_game = {str(g): d.drop(columns="game_pk").to_dict("index")
               for g, d in fields.groupby("game_pk")}
    jobs = [(f, by_game[f.name.split(".")[0]]) for f in sorted((base / "glove_locations").glob("*.csv.gz"))]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("OC_WORKERS", max(1, os.cpu_count() - 2)))) as ex:
        rows = [r for part in ex.map(targets_for_game, jobs, chunksize=4) for r in part]
    tg = pd.DataFrame(rows)

    # plausibility screen 
    pt = tg["play_id"].map(info["pitch_type"])
    z_lo = np.where(pt.isin(["FF", "SI", "FC"]), 10, 6)
    z_hi = np.where(pt == "FF", 50, 44)
    tg["plausible"] = ((tg["status"] == "ok") & (tg["naive_x_in"].abs() <= 20)
                       & (tg["naive_z_in"] > z_lo) & (tg["naive_z_in"] < z_hi))

    # inferred targets
    offset_key = [tg["play_id"].map(info["pitcher_id"]), pt]
    for ax in ("x", "z"):
        resid = (tg[f"plate_{ax}_in"] - tg[f"naive_{ax}_in"]).where(tg["plausible"])
        offset = resid.groupby(offset_key).transform("mean")
        tg[f"inferred_{ax}_in"] = (tg[f"naive_{ax}_in"] + offset).where(tg["plausible"])

    tg.to_csv(base / "targets.csv.gz", index=False, lineterminator="\n",
              compression={"method": "gzip", "compresslevel": 6})
    ok = int((tg["status"] == "ok").sum())
    print(f"targets: {ok} ok / {int(tg['plausible'].sum())} plausible of {len(tg)} posed clips")
