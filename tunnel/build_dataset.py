"""Merge OpenCommand targets (frozen), Statcast pbp, pitch context and reconstructed trajectories."""
import sys
import numpy as np
import pandas as pd
from trajectory import CKPT_NAMES

KEEP_TYPES = ["FF", "SI", "SL", "CH", "ST", "FC", "CU", "FS", "KC", "SV"]

def build(year, targets_file="targets.csv.gz"):
    p = pd.read_csv(f"data/{year}/pbp_info.csv.gz")
    t = pd.read_csv(f"data/{year}/{targets_file}")
    c = pd.read_csv(f"data/{year}/pitch_context.csv.gz")
    r = pd.read_parquet(f"tunnel/out/traj_{year}.parquet")
    t = t[(t.status == "ok") & (t.plausible)][["game_pk", "play_id", "inferred_x_in", "inferred_z_in", "naive_x_in", "naive_z_in"]]
    c = c[["game_pk", "play_id", "inning", "half_inning", "ab_number", "pitch_number", "stand", "p_throws", "catcher", "pre_balls", "pre_strikes"]]
    d = p.merge(t, on=["game_pk", "play_id"]).merge(c, on=["game_pk", "play_id"]).merge(r, on=["game_pk", "play_id"])
    d = d[d.pitch_type.isin(KEEP_TYPES) & (d.game_type == "R")].copy()
    d["target_x"] = d.inferred_x_in
    d["target_z"] = d.inferred_z_in
    d["plate_x_in"] = d.plate_x * 12
    d["plate_z_in"] = d.plate_z * 12
    d["velo"] = np.sqrt(d.vx0**2 + d.vy0**2 + d.vz0**2) * 3600 / 5280
    d["count"] = d.pre_balls.astype(int).astype(str) + "-" + d.pre_strikes.astype(int).astype(str)
    d = d.sort_values(["game_pk", "ab_number", "pitch_number"]).reset_index(drop=True)
    return d

if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2025"
    tf = sys.argv[2] if len(sys.argv) > 2 else "targets.csv.gz"
    tag = "" if tf == "targets.csv.gz" else "_unadj"
    d = build(year, tf)
    print(year, tf, len(d), "pitches,", d.pitcher_id.nunique(), "pitchers,", d.game_pk.nunique(), "games")
    print("target - plate miss median (in):", np.median(np.hypot(d.target_x - d.plate_x_in, d.target_z - d.plate_z_in)))
    d.to_parquet(f"tunnel/out/pitches_{year}{tag}.parquet")
