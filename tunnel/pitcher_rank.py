"""Per-pitcher tunnel profile and its split-half reliability. Only metrics that repeat get ranked."""
import numpy as np, pandas as pd
pr = pd.read_csv("tunnel/out/pitch_pair_tunneling_2025.csv.gz", usecols=["pitcher_id","pitcher","game_pk","label","same_type","type_A","type_B","swing_B","whiff_B"], low_memory=False)
pr = pr[~pr.same_type]   # different-pitch-type sequences only
pr["designed"] = pr.label == "designed"; pr["intent"] = pr.label.isin(["designed","failed"]); pr["actual"] = pr.label.isin(["designed","accidental"])
pr["failed"] = pr.label == "failed"; pr["accidental"] = pr.label == "accidental"
def prof(d):
    n = len(d); i = d.intent.sum(); a = d.actual.sum()
    return pd.Series(dict(n=n, intent_share=i/n, actual_share=a/n, designed_share=d.designed.sum()/n, accidental_share=d.accidental.sum()/n,
                          conversion=d.designed.sum()/max(i,1), acc_given_actual=d.accidental.sum()/max(a,1)))
full = pr.groupby(["pitcher_id","pitcher"]).apply(prof).reset_index()
g = pr.groupby("pitcher_id").game_pk.transform(lambda s: s.rank(method="dense")); m = pr.groupby("pitcher_id").game_pk.transform("nunique")
h1 = pr[g <= m/2].groupby("pitcher_id").apply(prof); h2 = pr[g > m/2].groupby("pitcher_id").apply(prof)
j = h1.join(h2, lsuffix="_1", rsuffix="_2"); j = j[(j.n_1 >= 150) & (j.n_2 >= 150)]
lines = [f"Split-half reliability (different-type pairs, >=150 per half, {len(j)} pitchers):"]
for mtr in ["intent_share","actual_share","designed_share","accidental_share","conversion","acc_given_actual"]:
    r = np.corrcoef(j[mtr+"_1"], j[mtr+"_2"])[0,1]; lines.append(f"  {mtr:18s} r = {r:.3f}   (Spearman-Brown {2*r/(1+r):.3f})")
full = full[full.n >= 400].copy()
lines.append(f"\nLeague (>=400 pairs, {len(full)} pitchers): intent {full.intent_share.mean():.3f} actual {full.actual_share.mean():.3f} designed {full.designed_share.mean():.3f} conversion {full.conversion.mean():.3f}")
for mtr, asc in [("designed_share", False), ("intent_share", False), ("conversion", False), ("conversion", True), ("accidental_share", False)]:
    t = full.sort_values(mtr, ascending=asc).head(12)
    lines.append(f"\n{'BOTTOM' if asc else 'TOP'} {mtr}:"); lines.append(t[["pitcher","n","intent_share","actual_share","designed_share","accidental_share","conversion"]].round(3).to_string(index=False))
w = full[full.pitcher.str.contains("Wheeler")]; lines.append("\nWheeler:"); lines.append(w.round(3).to_string(index=False))
for mtr in ["designed_share","intent_share","conversion","accidental_share"]:
    lines.append(f"  Wheeler rank on {mtr}: {int((full[mtr] > w[mtr].iloc[0]).sum())+1} of {len(full)}")
txt = "\n".join(lines); print(txt); open("tunnel/out/pitcher_rank_2025.txt","w").write(txt+"\n"); full.to_csv("tunnel/out/pitcher_profile_2025.csv", index=False)
