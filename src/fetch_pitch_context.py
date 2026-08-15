"""Step 3a of the OpenCommand pipeline (optional, network).

Pulls the per-pitch *situation* that `pbp_info.csv.gz` doesn't carry: count,
batter side, pitcher hand, catcher identity, inning/outs. The intent model
conditions on these, so it is a prerequisite for `intent_inference.py`.

Source is Baseball Savant's game feed (`/gf?game_pk=`), which is keyed on the
same `play_id` UUID the rest of the pipeline uses, so the join is exact.

Reads:      data/<year>/pbp_info.csv.gz (the game list)
Writes:     data/<year>/pitch_context.csv.gz, one row per pitch
Run:        python src/fetch_pitch_context.py [year=2026] [--workers 8]

Resumable: reruns skip games already in the output file.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / "data"
FEED = "https://baseballsavant.mlb.com/gf?game_pk={}"
KEEP = ["play_id", "inning", "half_inning", "outs", "ab_number", "pitch_number",
        "batter", "stand", "pitcher", "p_throws", "catcher", "pre_balls", "pre_strikes"]
RETRIES = 3


def fetch_game(game_pk):
    """One game's pitch rows from the Savant feed. Returns [] if the feed is unusable."""
    for attempt in range(RETRIES):
        try:
            r = requests.get(FEED.format(game_pk), timeout=60)
            r.raise_for_status()
            j = r.json()
            break
        except Exception:
            if attempt == RETRIES - 1:
                return []
            time.sleep(2 ** attempt)
    rows = []
    for side in ("team_home", "team_away"):
        for p in j.get(side) or []:
            if not p.get("play_id"):
                continue
            rows.append({"game_pk": int(game_pk), **{k: p.get(k) for k in KEEP}})
    return rows


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    year = argv[0] if argv else "2026"
    workers = 8
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    base = DATA / year
    out_path = base / "pitch_context.csv.gz"
    wanted = sorted(pd.read_csv(base / "pbp_info.csv.gz", usecols=["game_pk"])["game_pk"].unique())

    have = pd.DataFrame()
    if out_path.exists():
        have = pd.read_csv(out_path)
        done = set(have["game_pk"].unique())
        wanted = [g for g in wanted if g not in done]
        print(f"resuming: {len(done)} games already fetched, {len(wanted)} to go")

    rows, empty = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, part in enumerate(ex.map(fetch_game, wanted), 1):
            if not part:
                empty += 1
            rows.extend(part)
            if i % 100 == 0:
                print(f"  {i}/{len(wanted)} games, {len(rows)} pitches, {empty} empty", flush=True)

    ctx = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if len(rows) else have
    ctx = ctx.drop_duplicates(subset="play_id")
    ctx.to_csv(out_path, index=False, lineterminator="\n",
               compression={"method": "gzip", "compresslevel": 6})
    print(f"pitch_context: {len(ctx)} pitches over {ctx['game_pk'].nunique()} games "
          f"({empty} games returned nothing)")
