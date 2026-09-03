"""Cross-season repeatability of per-pitcher and per-pair-type excess tunneling (same-game null)."""
import sys, numpy as np, pandas as pd
a, b = sys.argv[1], sys.argv[2]
lines = [f"Cross-season repeatability {a} vs {b}, excess over same-game null at 23.8 ft (ti) and 40 ft (ti40)"]
for lev in ["game_type"]:
    A = pd.read_csv(f"tunnel/out/pitcher_excess_{a}.csv"); B = pd.read_csv(f"tunnel/out/pitcher_excess_{b}.csv")
    A, B = A[A.level == lev], B[B.level == lev]
    j = A.merge(B, on="pitcher_id", suffixes=("_a", "_b"))
    for minn in [100, 300]:
        k = j[(j.n_a >= minn) & (j.n_b >= minn)]
        lines.append(f"[pitcher, {lev}] min pairs {minn}: {len(k)} pitchers")
        for m in [m for m in ["ti", "ta", "ti40", "ta40"] if m+"_a" in k and m+"_b" in k]:
            lines.append(f"  {m:5s} raw r={np.corrcoef(k[m+'_a'], k[m+'_b'])[0,1]:6.3f} | excess r={np.corrcoef(k['excess_'+m+'_a'], k['excess_'+m+'_b'])[0,1]:6.3f} | mean excess {a} {k['excess_'+m+'_a'].mean():+.4f} {b} {k['excess_'+m+'_b'].mean():+.4f}")
txt = "\n".join(lines); print(txt); open(f"tunnel/out/cross_season_{a}_{b}.txt", "w").write(txt + "\n")
