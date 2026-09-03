"""Step 1: reconstruct Statcast 9-parameter trajectories at fixed distance-from-plate checkpoints.

Statcast convention: y measured from the plate apex toward the mound (ft), parameters given at
y0 = 50 ft, plate crossing defined at y = 17/12 ft (front of plate). Constant-acceleration model.
Positions are returned in INCHES to match OpenCommand's target files.
"""
import numpy as np
import pandas as pd

PLATE_Y_FT = 17.0 / 12.0
# checkpoints in feet from plate apex; 50 is the parameter origin (about 4-5 ft past release)
CHECKPOINTS_FT = [50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 23.8, 20.0, 15.0, 10.0, 5.0, PLATE_Y_FT]
CKPT_NAMES = [f"y{c:g}" if c != PLATE_Y_FT else "plate" for c in CHECKPOINTS_FT]


def time_at_y(y_target, y0, vy0, ay):
    """Smallest positive t with y0 + vy0 t + 0.5 ay t^2 = y_target (vy0 < 0, ball moving toward plate)."""
    a = 0.5 * ay
    b = vy0
    c = y0 - y_target
    disc = b * b - 4 * a * c
    disc = np.where(disc < 0, np.nan, disc)
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    # pick the smaller positive root; y is decreasing so the first crossing is what we want
    t = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
    t = np.where(np.isclose(y_target, y0), 0.0, t)
    return t


def positions_at(df, y_ft):
    """x,z (inches) of every pitch in df when the ball is at distance y_ft from the plate apex."""
    t = time_at_y(y_ft, df.y0.values, df.vy0.values, df.ay.values)
    x = df.x0.values + df.vx0.values * t + 0.5 * df.ax.values * t * t
    z = df.z0.values + df.vz0.values * t + 0.5 * df.az.values * t * t
    return x * 12.0, z * 12.0, t


def reconstruct(df, checkpoints=CHECKPOINTS_FT):
    """Return DataFrame with columns x_<ck>, z_<ck> (inches) and t_<ck> (s) per checkpoint."""
    out = {}
    for c, name in zip(checkpoints, CKPT_NAMES):
        x, z, t = positions_at(df, c)
        out[f"x_{name}"] = x
        out[f"z_{name}"] = z
        out[f"t_{name}"] = t
    return pd.DataFrame(out, index=df.index)


if __name__ == "__main__":
    import sys
    year = sys.argv[1] if len(sys.argv) > 1 else "2025"
    p = pd.read_csv(f"data/{year}/pbp_info.csv.gz")
    p = p.dropna(subset=["x0", "vy0", "ay", "plate_x", "plate_z"])
    r = reconstruct(p)
    ex = r.x_plate.values - p.plate_x.values * 12
    ez = r.z_plate.values - p.plate_z.values * 12
    err = np.hypot(ex, ez)
    lines = [f"Trajectory reconstruction validation, {year}: n={len(p)}",
             f"plate crossing error vs recorded plate_x/plate_z (inches):",
             f"  median {np.nanmedian(err):.4f}  p90 {np.nanpercentile(err,90):.4f}  p99 {np.nanpercentile(err,99):.4f}  max {np.nanmax(err):.3f}",
             f"  |dx| median {np.nanmedian(np.abs(ex)):.4f}   |dz| median {np.nanmedian(np.abs(ez)):.4f}",
             f"  rows with error > 1 in: {(err>1).sum()}  ({100*(err>1).mean():.3f}%)",
             f"  nan times: {np.isnan(r.t_plate).sum()}",
             f"time to plate median {np.nanmedian(r.t_plate):.4f}s; time at y=23.8 median {np.nanmedian(r['t_y23.8']):.4f}s"]
    # distance each checkpoint sits from release (approx 54 ft) for reference
    print("\n".join(lines))
    with open(f"tunnel/out/trajectory_validation_{year}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    r.insert(0, "play_id", p.play_id.values); r.insert(0, "game_pk", p.game_pk.values)
    r.to_parquet(f"tunnel/out/traj_{year}.parquet")
