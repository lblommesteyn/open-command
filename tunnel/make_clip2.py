"""Annotated clip v2: one broadcast panel (pitch 1's video) with BOTH pitches overlaid as moving balls, targets and
faint intended paths, plus two live schematics (side view, top view) where the intended (dashed) vs actual (solid)
comparison is actually legible. Crop is derived from the projected geometry so it works in any park.

Usage: PYTHONPATH="tunnel;src" python tunnel/make_clip2.py tunnel/clip/<config>.json
"""
import json, sys, subprocess
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import poselib
from trajectory import CKPT_NAMES, CHECKPOINTS_FT
from intent_model import centered_ols, YCOLS

CFG = json.load(open(sys.argv[1]))
GAME, PIDS, CLIPS = CFG["game"], CFG["pids"], CFG["clips"]
BLUE, ORANGE = "#2a78d6", "#eb6834"
COL = {"A": (214, 120, 42), "B": (52, 104, 235)}          # BGR
HEX = {"A": BLUE, "B": ORANGE}
T_OFFSET = CFG["t_offset"]
PRE_S, FLIGHT_S, HOLD_S = 0.6, 0.46, 3.0
SLOW, OUT_FPS = 4, 30
W, H = 1920, 1080
HEAD, FOOT = 120, 100
VW, VH = 1180, H - HEAD - FOOT          # broadcast panel
SW, SH = W - VW, (H - HEAD - FOOT) // 2  # schematic panels
FONT = "C:/Windows/Fonts/segoeui.ttf"; FONTB = "C:/Windows/Fonts/segoeuib.ttf"

pbp = pd.read_csv("data/2025/pbp_info.csv.gz"); pbp = pbp[pbp.play_id.isin(PIDS.values())].set_index("play_id")
poses = pd.read_csv("data/2025/camera_poses.csv.gz"); poses = poses[poses.play_id.isin(PIDS.values())].set_index("play_id")
tg = pd.read_csv("data/2025/targets_unadjusted.csv.gz"); tg = tg[tg.play_id.isin(PIDS.values())].set_index("play_id")
pitcher_id = int(pbp.pitcher_id.iloc[0])
P = pd.read_parquet("tunnel/out/pitches_intent_2025.parquet", columns=["pitcher_id", "pitch_type", "game_pk", "plate_x_in", "plate_z_in"] + YCOLS)
P = P[(P.pitcher_id == pitcher_id) & (P.game_pk != GAME)]

def intended(pt, target_xz):
    g = P[P.pitch_type == pt]
    mx, my, B = centered_ols(g[["plate_x_in", "plate_z_in"]].values, g[YCOLS].values)
    v = my + (np.asarray(target_xz) - mx) @ B
    n = len(CKPT_NAMES); y = np.array(CHECKPOINTS_FT)
    yy = np.linspace(y.max(), y.min(), 80)
    return np.stack([np.interp(yy, y[::-1], v[:n][::-1]) / 12, yy, np.interp(yy, y[::-1], v[n:][::-1]) / 12], axis=1)

def actual(row, t):
    t = np.atleast_1d(t)
    return np.stack([row.x0 + row.vx0 * t + 0.5 * row.ax * t * t, row.y0 + row.vy0 * t + 0.5 * row.ay * t * t, row.z0 + row.vz0 * t + 0.5 * row.az * t * t], axis=1)

info = {}
for k, pid in PIDS.items():
    row, ps, t = pbp.loc[pid], poses.loc[pid], tg.loc[pid]
    tp = (-row.vy0 - np.sqrt(row.vy0 ** 2 - 2 * row.ay * (row.y0 - 17 / 12))) / row.ay
    info[k] = dict(row=row, intent=intended(row.pitch_type, (t.inferred_x_in, t.inferred_z_in)), target=np.array([t.inferred_x_in / 12, 17 / 12, t.inferred_z_in / 12]),
                   release_s=t.release_s, t_plate=tp, actual_full=actual(row, np.linspace(0, tp, 80)))
# one camera: pitch A's pose (the video we show)
psA = poses.loc[PIDS["A"]]; C = np.array([psA.Cx, psA.Cy, psA.Cz]); basis = poselib.camera(C, psA.pan, psA.tilt, psA.roll)
proj = lambda X: poselib.project_on(basis, C, psA.f_px, X)
# crop: bounding box of everything we draw, padded, forced to the panel aspect
allpx = np.vstack([proj(info[k]["intent"]) for k in "AB"] + [proj(info[k]["actual_full"]) for k in "AB"] + [proj(np.array([[0, 60.5, 5.5]]))])
x0, y0 = allpx.min(0) - [120, 80]; x1, y1 = allpx.max(0) + [120, 120]
cw, ch = max(x1 - x0, 640), y1 - y0
if cw / ch < VW / VH: cw = ch * VW / VH
else: ch = cw * VH / VW
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
x0, x1 = int(max(0, cx - cw / 2)), int(min(1280, cx + cw / 2)); y0, y1 = int(max(0, cy - ch / 2)), int(min(720, cy + ch / 2))
print("crop", x0, y0, x1, y1)
capA = cv2.VideoCapture(f"tunnel/clip/{CLIPS['A']}.mp4"); fps = capA.get(cv2.CAP_PROP_FPS); rel = int(round(info["A"]["release_s"] * fps))

def draw_dashed(img, pts, color, thick=1, dash=3):
    pts = pts.astype(int)
    for i in range(len(pts) - 1):
        if (i // dash) % 2 == 0:
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, thick, cv2.LINE_AA)

def draw_x(img, p, color, s=6, thick=2):
    x, y = int(p[0]), int(p[1])
    cv2.line(img, (x - s, y - s), (x + s, y + s), color, thick, cv2.LINE_AA); cv2.line(img, (x - s, y + s), (x + s, y - s), color, thick, cv2.LINE_AA)

def video_panel(frame, t_rel, final):
    img = frame.copy()
    for k in "AB":
        draw_dashed(img, proj(info[k]["intent"]), COL[k])
        draw_x(img, proj(info[k]["target"][None])[0], COL[k])
    for k in "AB":
        r = info[k]["row"]
        t_end = info[k]["t_plate"] if final else min(t_rel - T_OFFSET[k], info[k]["t_plate"])
        if t_end > 0:
            pts = proj(actual(r, np.linspace(0, t_end, 60))).astype(int)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, COL[k], 2, cv2.LINE_AA)
            if not final:
                cv2.circle(img, tuple(pts[-1]), 5, (255, 255, 255), -1, cv2.LINE_AA); cv2.circle(img, tuple(pts[-1]), 5, COL[k], 2, cv2.LINE_AA)
    return cv2.resize(img[y0:y1, x0:x1], (VW, VH), interpolation=cv2.INTER_CUBIC)

# --- schematics (matplotlib, updated per frame) ---
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 13})
fig, (ax_side, ax_top) = plt.subplots(2, 1, figsize=(SW / 100, 2 * SH / 100), dpi=100)
fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.09, hspace=0.45)
lines = {}
for ax, comp, lab in [(ax_side, 2, "height (in)"), (ax_top, 0, "left-right (in), + = catcher's right")]:
    for k in "AB":
        ip = info[k]["intent"]
        ax.plot(ip[:, 1], ip[:, comp] * 12, "--", color=HEX[k], lw=2, alpha=0.9)
        lines[(ax, k)] = ax.plot([], [], "-", color=HEX[k], lw=3)[0]
        lines[(ax, k, "ball")] = ax.plot([], [], "o", color=HEX[k], ms=9, mec="white", mew=1.5)[0]
        ax.plot([17 / 12], [info[k]["target"][comp] * 12], "x", color=HEX[k], ms=11, mew=3)
    ax.axvline(23.8, color="#bbbbbb", ls=":", lw=1)
    ax.set_xlim(52, -1); ax.set_ylabel(lab); ax.grid(color="#e6e6e6", lw=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
ax_side.set_title("side view: dashed = aimed, solid = thrown", loc="left", fontsize=13, color="#52514e")
ax_top.set_title("top view", loc="left", fontsize=13, color="#52514e"); ax_top.set_xlabel("distance from the plate (ft)")
ax_side.text(23.8, ax_side.get_ylim()[1], " 23.8 ft", va="top", fontsize=10, color="#8a8a8a")
def schematic(t_rel, final):
    for k in "AB":
        t_end = info[k]["t_plate"] if final else min(t_rel - T_OFFSET[k], info[k]["t_plate"])
        pts = actual(info[k]["row"], np.linspace(0, max(t_end, 0), 60)) if t_end > 0 else np.zeros((0, 3))
        for ax, comp in [(ax_side, 2), (ax_top, 0)]:
            lines[(ax, k)].set_data(pts[:, 1], pts[:, comp] * 12)
            lines[(ax, k, "ball")].set_data(pts[-1:, 1], pts[-1:, comp] * 12) if (len(pts) and not final) else lines[(ax, k, "ball")].set_data([], [])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return cv2.cvtColor(np.ascontiguousarray(buf), cv2.COLOR_RGB2BGR)

fontT = ImageFont.truetype(FONTB, 38); fontS = ImageFont.truetype(FONT, 24); fontF = ImageFont.truetype(FONT, 24); fontC = ImageFont.truetype(FONT, 20)
def compose(vid, sch, phase_text):
    canvas = np.full((H, W, 3), 255, np.uint8)
    canvas[HEAD:HEAD + VH, :VW] = vid
    sch = cv2.resize(sch, (SW, 2 * SH)); canvas[HEAD:HEAD + 2 * SH, VW:VW + SW] = sch
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
    d.text((28, 16), CFG["title"], font=fontT, fill=(11, 11, 11))
    d.text((28, 70), CFG["subtitle"], font=fontS, fill=(82, 81, 78))
    d.text((W - 400, 24), "OpenCommand targets + Statcast\ngithub.com/tomdoyo/open-command", font=fontC, fill=(138, 138, 138))
    d.text((28, HEAD + VH + 12), CFG["labels"]["A"], font=fontF, fill=(42, 120, 214))
    d.text((28 + 620, HEAD + VH + 12), CFG["labels"]["B"], font=fontF, fill=(235, 104, 52))
    d.text((28, HEAD + VH + 50), phase_text, font=fontF, fill=(82, 81, 78))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def get_frame(i):
    capA.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = capA.read(); return f

frames = []
n_pre, n_fl = int(PRE_S * fps), int(FLIGHT_S * fps)
for i in range(-n_pre, 0, 2):
    frames.append(compose(video_panel(get_frame(rel + i), i / fps, False), schematic(i / fps, False), CFG["pre"]))
for i in range(0, n_fl + 1):
    fr = compose(video_panel(get_frame(rel + i), i / fps, False), schematic(i / fps, False), f"1/8 speed.  {i / fps * 1000:4.0f} ms after release.  The batter has to commit by about 200 ms, when the ball is 23.8 ft away.")
    frames.extend([fr] * SLOW)
fr = compose(video_panel(get_frame(rel + n_fl), 9, True), schematic(9, True), CFG["final"])
frames.extend([fr] * int(HOLD_S * OUT_FPS))
tmp = "tunnel/clip/_raw.avi"
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), OUT_FPS, (W, H))
for f in frames: vw.write(f)
vw.release()
out = f"tunnel/thread/{CFG['out']}.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out], check=True)
cv2.imwrite("tunnel/clip/preview_flight.png", frames[len(frames) // 2]); cv2.imwrite("tunnel/clip/preview_final.png", frames[-1])
print("wrote", len(frames), "frames ->", out)
