"""Visual examples: intended vs actual trajectories with targets, for designed / accidental / failed pairs,
plus the null-comparison and intent->actual lift sweeps across checkpoints."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajectory import CKPT_NAMES, CHECKPOINTS_FT

EARLY = [c for c in CKPT_NAMES if c != "plate"]
year = sys.argv[1] if len(sys.argv) > 1 else "2025"
pr = pd.read_csv(f"tunnel/out/pitch_pair_tunneling_{year}.csv.gz")
cols = ["play_id", "pitch_type", "target_x", "target_z", "plate_x_in", "plate_z_in"] + [f"{p}{a}_{c}" for p in ["", "i"] for a in ["x", "z"] for c in CKPT_NAMES]
P = pd.read_parquet(f"tunnel/out/pitches_intent_{year}.parquet", columns=cols).set_index("play_id")

def traj(row, kind):
    pre = "i" if kind == "intent" else ""
    return np.array([row[f"{pre}x_{c}"] for c in CKPT_NAMES]), np.array([row[f"{pre}z_{c}"] for c in CKPT_NAMES])

def draw_pair(ax_top, ax_side, pair, title):
    A, B = P.loc[pair.play_id_A], P.loc[pair.play_id_B]
    y = np.array(CHECKPOINTS_FT)
    for row, col, nm in [(A, "tab:blue", pair.type_A), (B, "tab:red", pair.type_B)]:
        for kind, ls in [("actual", "-"), ("intent", "--")]:
            x, z = traj(row, kind)
            ax_top.plot(y, x, ls, color=col, label=f"{nm} {kind}")
            ax_side.plot(y, z, ls, color=col)
        ax_top.plot([y[-1]], [row.target_x], "x", color=col, ms=9, mew=2)
        ax_side.plot([y[-1]], [row.target_z], "x", color=col, ms=9, mew=2)
        ax_top.plot([y[-1]], [row.plate_x_in], "o", color=col, ms=7)
        ax_side.plot([y[-1]], [row.plate_z_in], "o", color=col, ms=7)
    for ax in (ax_top, ax_side):
        ax.axvline(23.8, color="gray", lw=0.8, ls=":")
        ax.invert_xaxis()
    ax_top.set_ylabel("x (in), catcher view"); ax_side.set_ylabel("z (in)")
    ax_side.set_xlabel("distance from plate (ft)")
    ax_top.set_title(title, fontsize=9)
    ax_top.legend(fontsize=7, loc="best")

ev = pr[pr.eval_half & (pr.plate_sep_intent > 8) & (pr.plate_sep_actual > 8)]
rng = np.random.default_rng(3)
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for j, lab in enumerate(["designed", "accidental", "failed"]):
    sub = ev[(ev.label == lab) & (~ev.same_type)]
    # pick a clear example: extreme on the defining scores
    if lab == "designed":
        sub = sub.sort_values("ts_intent_y23.8", ascending=False).head(200)
    elif lab == "accidental":
        sub = sub.sort_values("ts_intent_y23.8").head(200)
    else:
        sub = sub.sort_values("ts_actual_y23.8").head(200)
    row = sub.iloc[rng.integers(len(sub))]
    t = (f"{lab.upper()}: {row.pitcher} {row.pair_type}, {row.date}, count {row.count_B}\n"
         f"score intent {row['ts_intent_y23.8']:.2f} / actual {row['ts_actual_y23.8']:.2f}; "
         f"sep @23.8ft intent {row['sep_intent_y23.8']:.1f} / actual {row['sep_actual_y23.8']:.1f} in; plate {row.plate_sep_intent:.1f} / {row.plate_sep_actual:.1f} in")
    draw_pair(axes[0, j], axes[1, j], row, t)
fig.suptitle("Solid = actual Statcast trajectory, dashed = intent-conditioned trajectory, x = OpenCommand target, o = actual plate location, dotted line = 23.8 ft", fontsize=10)
fig.tight_layout(); fig.savefig(f"tunnel/fig/examples_{year}.png", dpi=130); plt.close(fig)

# sweep figures
nt = pd.read_csv(f"tunnel/out/null_test_{year}.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
xs = [CHECKPOINTS_FT[CKPT_NAMES.index(c)] for c in EARLY]
for lev, col in [("season_type", "tab:orange"), ("game_type", "tab:green"), ("season_any", "tab:purple")]:
    s = nt[nt.level == lev].set_index("checkpoint").loc[EARLY]
    axes[0].plot(xs, s.obs_ts_intent - s.null_ts_intent, "-o", color=col, label=f"intent, null={lev}")
    axes[0].plot(xs, s.obs_ts_actual - s.null_ts_actual, "--s", color=col, alpha=0.6, label=f"actual, null={lev}")
    axes[1].plot(xs, s.obs_sep_intent / s.null_sep_intent, "-o", color=col, label=f"intent early sep, null={lev}")
    axes[1].plot(xs, s.obs_plate_intent / s.null_plate_intent, ":", color=col, label=f"intent plate sep, null={lev}")
axes[0].axhline(0, color="k", lw=0.8); axes[0].set_title("mean tunnel score: observed minus null"); axes[0].set_xlabel("checkpoint (ft from plate)"); axes[0].legend(fontsize=7); axes[0].invert_xaxis()
axes[1].axhline(1, color="k", lw=0.8); axes[1].set_title("observed / null separation (intent)"); axes[1].set_xlabel("checkpoint (ft from plate)"); axes[1].legend(fontsize=7); axes[1].invert_xaxis()
# lift sweep from classification text is easier to recompute here
evh = pr[pr.eval_half]
exh = pr[~pr.eval_half]
lift = []
for c in EARLY:
    ti = (evh[f"ts_intent_{c}"] >= exh[f"ts_intent_{c}"].quantile(0.75)) & (evh.plate_sep_intent >= 6)
    ta = (evh[f"ts_actual_{c}"] >= exh[f"ts_actual_{c}"].quantile(0.75)) & (evh.plate_sep_actual >= 6)
    lift.append((ta[ti].mean(), ta[~ti].mean()))
lift = np.array(lift)
axes[2].plot(xs, lift[:, 0], "-o", label="P(actual tunnel | intent tunnel)")
axes[2].plot(xs, lift[:, 1], "-o", label="P(actual tunnel | no intent tunnel)")
axes[2].set_title("intent -> actual, evaluation half, frozen thresholds"); axes[2].set_xlabel("checkpoint (ft from plate)"); axes[2].legend(fontsize=8); axes[2].invert_xaxis(); axes[2].set_ylim(0, 0.6)
fig.tight_layout(); fig.savefig(f"tunnel/fig/sweeps_{year}.png", dpi=130); plt.close(fig)

# scatter of intent vs actual tunnel score at primary checkpoint (hexbin)
fig, ax = plt.subplots(figsize=(5.5, 5))
hb = ax.hexbin(evh["ts_intent_y23.8"], evh["ts_actual_y23.8"], gridsize=60, bins="log", cmap="viridis")
ax.axvline(evh.thr_intent_primary.iloc[0], color="w", ls="--"); ax.axhline(evh.thr_actual_primary.iloc[0], color="w", ls="--")
ax.set_xlabel("intent tunnel score at 23.8 ft"); ax.set_ylabel("actual tunnel score at 23.8 ft"); ax.set_title(f"evaluation half, r = {evh['ts_intent_y23.8'].corr(evh['ts_actual_y23.8']):.3f}")
fig.colorbar(hb, ax=ax, label="log10 pairs"); fig.tight_layout(); fig.savefig(f"tunnel/fig/intent_vs_actual_{year}.png", dpi=130)
print("figures written")
