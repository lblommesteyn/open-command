"""Step 3: consecutive-pitch pairs within a PA and tunnel scores for actual and intended trajectories.

For pair A -> B and checkpoint d:
    sep_actual(d) = |P_actual_A(d) - P_actual_B(d)|          sep_intent(d) = |P_intent_A(d) - P_intent_B(d)|
    plate separation: actual = |plate_A - plate_B|,           intent = |target_A - target_B|
    tunnel score TS(d) = log(sep_plate / sep(d))   (large = small early separation, large late separation)
Scores are computed at every checkpoint; nothing is chosen here.
"""
import sys
import numpy as np
import pandas as pd
from trajectory import CKPT_NAMES

EARLY = [c for c in CKPT_NAMES if c != "plate"]


def swing_flags(desc):
    s = desc.str.lower()
    swing = s.str.contains("swing|foul|in play|missed bunt")
    whiff = s.str.contains("swinging strike|missed bunt|foul tip")  # foul tip counts as whiff-like? keep separate below
    whiff = s.str.contains("swinging strike|missed bunt")
    called = s == "called strike"
    inplay = s.str.contains("in play")
    return swing, whiff, called, inplay


def build_pairs(d):
    d = d[d.n_train_cell > 0].copy()
    d = d.sort_values(["game_pk", "ab_number", "pitch_number"])
    nxt = d.shift(-1)
    same = (nxt.game_pk == d.game_pk) & (nxt.ab_number == d.ab_number) & (nxt.pitch_number == d.pitch_number + 1)
    A = d[same.values].reset_index(drop=True)
    B = nxt[same.values].reset_index(drop=True)
    out = pd.DataFrame({
        "game_pk": A.game_pk, "date": A.date, "play_id_A": A.play_id, "play_id_B": B.play_id,
        "pitcher_id": A.pitcher_id, "pitcher": A.pitcher, "ab_number": A.ab_number, "pitch_number_B": B.pitch_number,
        "stand": A.stand, "p_throws": A.p_throws, "count_B": B["count"], "strikes_B": B.pre_strikes.astype(int), "balls_B": B.pre_balls.astype(int),
        "type_A": A.pitch_type, "type_B": B.pitch_type, "velo_A": A.velo, "velo_B": B.velo,
        "target_xA": A.target_x, "target_zA": A.target_z, "target_xB": B.target_x, "target_zB": B.target_z,
        "plate_xA": A.plate_x_in, "plate_zA": A.plate_z_in, "plate_xB": B.plate_x_in, "plate_zB": B.plate_z_in,
        "desc_B": B.description, "sz_top_B": B.sz_top, "sz_bot_B": B.sz_bot,
    })
    out["pair_type"] = out.type_A + ">" + out.type_B
    out["same_type"] = out.type_A == out.type_B
    out["plate_sep_actual"] = np.hypot(A.plate_x_in - B.plate_x_in, A.plate_z_in - B.plate_z_in)
    out["plate_sep_intent"] = np.hypot(A.target_x - B.target_x, A.target_z - B.target_z)
    for c in EARLY:
        sa = np.hypot(A[f"x_{c}"] - B[f"x_{c}"], A[f"z_{c}"] - B[f"z_{c}"])
        si = np.hypot(A[f"ix_{c}"] - B[f"ix_{c}"], A[f"iz_{c}"] - B[f"iz_{c}"])
        out[f"sep_actual_{c}"] = sa.values
        out[f"sep_intent_{c}"] = si.values
        out[f"ts_actual_{c}"] = np.log(out.plate_sep_actual.clip(0.5) / sa.clip(0.5)).values
        out[f"ts_intent_{c}"] = np.log(out.plate_sep_intent.clip(0.5) / si.clip(0.5)).values
    sw, wh, ca, ip = swing_flags(out.desc_B)
    out["swing_B"], out["whiff_B"], out["called_strike_B"], out["inplay_B"] = sw.values, wh.values, ca.values, ip.values
    # is pitch B's actual location in the zone (rulebook width 17in + ball radius)
    out["zone_B"] = (out.plate_xB.abs() <= 9.95) & (out.plate_zB >= out.sz_bot_B * 12) & (out.plate_zB <= out.sz_top_B * 12)
    return out


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2025"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    d = pd.read_parquet(f"tunnel/out/pitches_intent_{year}{tag}.parquet")
    pr = build_pairs(d)
    print(len(pr), "pairs;", pr.pitcher_id.nunique(), "pitchers; same-type share", round(pr.same_type.mean(), 3))
    cols = ["checkpoint", "sep_actual_med", "sep_intent_med", "plate_sep_actual_med", "plate_sep_intent_med", "corr(ts_intent,ts_actual)", "corr(sep_intent,sep_actual)"]
    rows = []
    for c in EARLY:
        rows.append([c, pr[f"sep_actual_{c}"].median(), pr[f"sep_intent_{c}"].median(), pr.plate_sep_actual.median(), pr.plate_sep_intent.median(),
                     pr[f"ts_intent_{c}"].corr(pr[f"ts_actual_{c}"]), pr[f"sep_intent_{c}"].corr(pr[f"sep_actual_{c}"])])
    t = pd.DataFrame(rows, columns=cols).round(3)
    print(t.to_string(index=False))
    open(f"tunnel/out/pair_summary_{year}{tag}.txt", "w").write(t.to_string(index=False) + "\n")
    pr.to_parquet(f"tunnel/out/pairs_{year}{tag}.parquet")
