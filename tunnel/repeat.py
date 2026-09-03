"""Split-half repeatability of tunnel excess (observed minus same-game null) by pitcher and by pair type.
Halves = each pitcher's games split by date (first half of their games vs second half)."""
import sys
import numpy as np
import pandas as pd
from null_test import run

def half_filter(h):
    def f(pr):
        g = pr.groupby("pitcher_id").game_pk.transform(lambda s: s.rank(method="dense"))
        m = pr.groupby("pitcher_id").game_pk.transform("nunique")
        return (g <= m / 2) if h == 1 else (g > m / 2)
    return f

def wcorr(a, b, w):
    w = w / w.sum(); ma, mb = (w * a).sum(), (w * b).sum()
    return ((w * (a - ma) * (b - mb)).sum()) / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())

def spearman_brown(r):
    return 2 * r / (1 + r)

if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2025"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    level = sys.argv[3] if len(sys.argv) > 3 else "game_type"
    lines = [f"Split-half repeatability, {year}{tag}, null level = {level}, halves = pitcher's games by date"]
    for group_col in ["pitcher_id", "pair_type"]:
        halves = []
        for h in (1, 2):
            _, pp = run(year, tag, levels=(level,), R=50, seed=10 + h, pair_filter=half_filter(h), group_col=group_col)
            halves.append(pp.set_index(group_col))
        j = halves[0].join(halves[1], lsuffix="_1", rsuffix="_2", how="inner")
        j.to_csv(f"tunnel/out/repeat_{group_col}_{year}{tag}_{level}.csv")
        for minn in ([100, 300] if group_col == "pitcher_id" else [500, 2000]):
            k = j[(j.n_1 >= minn) & (j.n_2 >= minn)]
            w = np.minimum(k.n_1, k.n_2).values.astype(float)
            lines.append(f"\n[{group_col}] min pairs per half {minn}: {len(k)} groups")
            for m in ["ti", "ta", "ti40", "ta40"]:
                r_raw = np.corrcoef(k[f"{m}_1"], k[f"{m}_2"])[0, 1]
                r_ex = np.corrcoef(k[f"excess_{m}_1"], k[f"excess_{m}_2"])[0, 1]
                r_exw = wcorr(k[f"excess_{m}_1"].values, k[f"excess_{m}_2"].values, w)
                lines.append(f"  {m:5s} raw score r={r_raw:6.3f} | excess over null r={r_ex:6.3f} (weighted {r_exw:6.3f}, Spearman-Brown full-season {spearman_brown(r_ex):5.3f}) | mean excess h1 {k[f'excess_{m}_1'].mean():+.4f} h2 {k[f'excess_{m}_2'].mean():+.4f}")
    txt = "\n".join(lines)
    print(txt)
    open(f"tunnel/out/repeat_{year}{tag}_{level}.txt", "w").write(txt + "\n")
