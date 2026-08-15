"""Step 3b of the OpenCommand pipeline.

Turns naive glove targets into **intent targets** with the per-pitcher latent
intent model in `intentlib.py` (see that module for the model itself).

Where step 3's inferred target says "the glove is the target, up to one constant
per pitcher x pitch type", this says "the glove is *evidence* about the target,
and how much evidence it is differs by pitcher". It also splits a pitch type into
two targets when the pitcher demonstrably works both sides with it, and corrects
for who was catching.

Reads:      data/<year>/targets.csv.gz (naive + inferred targets)
            data/<year>/pbp_info.csv.gz (pitcher, pitch type, actual location)
            data/<year>/pitch_context.csv.gz (count, side, catcher; step 3a)
Writes:     data/<year>/intent_targets.csv.gz, one row per plausible clip
            data/<year>/pitcher_gain.csv, one row per pitcher
Run:        python src/intent_inference.py [year=2026] [--form gain|w]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intentlib as il

DATA = Path(__file__).resolve().parents[1] / "data"


def main(year="2026", form="gain"):
    base = DATA / year
    ctx_path = base / "pitch_context.csv.gz"
    if not ctx_path.exists():
        sys.exit(f"missing {ctx_path} - run: python src/fetch_pitch_context.py {year}")

    df = il.prepare(pd.read_csv(base / "targets.csv.gz"),
                    pd.read_csv(base / "pbp_info.csv.gz"),
                    pd.read_csv(ctx_path))
    model = il.fit(df, form=form)
    ix, iz = il.predict(model, df)

    out = df[["game_pk", "play_id", "pitcher_id", "pitcher", "pitch_type", "pgroup",
              "stand", "two_strike", "catcher", "naive_x_in", "naive_z_in",
              "inferred_x_in", "inferred_z_in", "plate_x_in", "plate_z_in"]].copy()
    out["cluster"] = il.assign_clusters(df.assign(**dict(zip(
        ["glove_x", "glove_z"], il.apply_catcher_bias(df, model["catcher_bias"])
        if model["catcher_bias"] is not None else (df["glove_x"], df["glove_z"])))),
        model["clusters"])
    out["intent_x_in"], out["intent_z_in"] = ix, iz
    out["gain_s"] = out["pitcher_id"].map(model["gain"]).fillna(model["mu"])
    out["miss_naive_in"] = il.miss(df, df["glove_x"].to_numpy(), df["glove_z"].to_numpy())
    out["miss_inferred_in"] = il.miss(df, df["inferred_x_in"].to_numpy(),
                                      df["inferred_z_in"].to_numpy())
    out["miss_intent_in"] = il.miss(df, ix, iz)
    out.to_csv(base / "intent_targets.csv.gz", index=False, lineterminator="\n",
               compression={"method": "gzip", "compresslevel": 6})

    gain = pd.DataFrame({"gain_s": model["gain"], "gain_s_raw": model["gain_raw"],
                         "gain_se": model["gain_se"], "n": model["gain_n"]})
    gain["w"] = 1 - gain["gain_s"]
    gain["pitcher"] = df.drop_duplicates("pitcher_id").set_index("pitcher_id")["pitcher"]
    gain.index.name = "pitcher_id"
    gain.sort_values("n", ascending=False).to_csv(base / "pitcher_gain.csv",
                                                 lineterminator="\n")

    print(f"intent targets: {len(out)} clips, {len(model['clusters'])} two-target groups")
    print(f"league gain s = {model['mu']:.3f}, tau = {model['tau']:.3f}")
    for col in ("miss_naive_in", "miss_inferred_in", "miss_intent_in"):
        print(f"  in-sample median {col:18s} {np.median(out[col]):.3f} in")
    print("(in-sample - see evaluate_intent.py for the held-out numbers)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    year = argv[0] if argv and not argv[0].startswith("--") else "2026"
    form = argv[argv.index("--form") + 1] if "--form" in argv else "gain"
    main(year, form)
