"""Step 5: do consecutive-pitch targets tunnel more than chance?

Null: keep pitch A, replace pitch B by a random OTHER pitch of the same pitcher drawn from a stratum that
preserves the marginals, then recompute the intent (and actual) tunnel scores with the donor's trajectories.
Strata (levels):
  season_type : same pitcher, pitch type of B, batter side, strikes            -> targets given pitch selection
  game_type   : same pitcher, SAME GAME, pitch type of B, batter side           -> also removes per-game target/camera bias
  season_any  : same pitcher, batter side, strikes, ANY pitch type              -> pitch selection + target
Statistic per checkpoint d: mean tunnel score ts(d) and mean early separation sep(d) over pairs.
"""
import sys
import numpy as np
import pandas as pd
from trajectory import CKPT_NAMES

EARLY = [c for c in CKPT_NAMES if c != "plate"]
R = 200


def stratum_sampler(pitches, keys, rng):
    """returns function(pair_idx_of_B_rows, own_rows) -> donor row indices (or -1)."""
    order = pitches.sort_values(keys).index.values
    key = pitches.loc[order, keys].astype(str).agg("|".join, axis=1).values
    starts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1]
    lens = np.diff(np.r_[starts, len(key)])
    kmap = dict(zip(key[starts], zip(starts, lens)))

    def draw(bkeys, own, own2=None):
        st = np.array([kmap.get(k, (0, 0)) for k in bkeys])
        s, n = st[:, 0], st[:, 1]
        donor = np.full(len(bkeys), -1)
        valid = n >= 3
        for _ in range(8):
            u = rng.random(len(bkeys))
            cand = order[(s + np.floor(u * np.maximum(n, 1)).astype(int)).clip(0, len(order) - 1)]
            take = valid & (donor < 0) & (cand != own)
            if own2 is not None:
                take &= cand != own2
            donor[take] = cand[take]
        donor[~valid] = -1
        return donor
    return draw


def sep_between(IA, IB):
    """IA, IB: (n, 2*ncp) arrays [x..., z...]. returns (n, ncp) separations."""
    m = IA.shape[1] // 2
    return np.hypot(IA[:, :m] - IB[:, :m], IA[:, m:] - IB[:, m:])


def scores(sep, plate_sep):
    return np.log(np.clip(plate_sep, 0.5, None)[:, None] / np.clip(sep, 0.5, None))


def run(year, tag, levels=("season_type", "game_type", "season_any"), R=R, seed=0, per_pitcher_out=True, pair_filter=None, group_col="pitcher_id"):
    rng = np.random.default_rng(seed)
    P = pd.read_parquet(f"tunnel/out/pitches_intent_{year}{tag}.parquet")
    P = P[P.n_train_cell > 0].reset_index(drop=True)
    P["strikes"] = P.pre_strikes.astype(int)
    pr = pd.read_parquet(f"tunnel/out/pairs_{year}{tag}.parquet")
    if pair_filter is not None:
        pr = pr[pair_filter(pr)].reset_index(drop=True)
    rowmap = pd.Series(P.index.values, index=P.play_id.values)
    ia, ib = rowmap[pr.play_id_A].values, rowmap[pr.play_id_B].values
    icols = [f"ix_{c}" for c in CKPT_NAMES] + [f"iz_{c}" for c in CKPT_NAMES]
    acols = [f"x_{c}" for c in CKPT_NAMES] + [f"z_{c}" for c in CKPT_NAMES]
    I, A = P[icols].values, P[acols].values
    ncp = len(CKPT_NAMES)  # last checkpoint is plate
    T = P[["target_x", "target_z"]].values
    PL = P[["plate_x_in", "plate_z_in"]].values
    strata = {"season_type": ["pitcher_id", "pitch_type", "stand", "strikes"],
              "pa_any": ["pitcher_id", "game_pk", "ab_number"],
              "game_any": ["pitcher_id", "game_pk", "stand"],
              "game_type": ["pitcher_id", "game_pk", "pitch_type", "stand"],
              "season_any": ["pitcher_id", "stand", "strikes"]}
    results, pp_rows = [], []
    for lev in levels:
        keys = strata[lev]
        draw = stratum_sampler(P, keys, rng)
        bkeys = P.loc[ib, keys].astype(str).agg("|".join, axis=1).values
        # observed
        obs_si = sep_between(I[ia], I[ib]); obs_sa = sep_between(A[ia], A[ib])
        obs_ti = scores(obs_si[:, :-1], obs_si[:, -1]); obs_ta = scores(obs_sa[:, :-1], obs_sa[:, -1])
        null_ti = np.zeros((R, ncp - 1)); null_ta = np.zeros((R, ncp - 1)); null_si = np.zeros((R, ncp - 1)); null_sa = np.zeros((R, ncp - 1))
        null_pl_i = np.zeros(R); null_pl_a = np.zeros(R)
        valid_any = None
        pp_obs, pp_null = None, []
        for r in range(R):
            donor = draw(bkeys, ib, ia)
            v = donor >= 0
            if valid_any is None:
                valid_any = v
            dn = np.where(v, donor, ib)
            si = sep_between(I[ia], I[dn]); sa = sep_between(A[ia], A[dn])
            ti = scores(si[:, :-1], si[:, -1]); ta = scores(sa[:, :-1], sa[:, -1])
            null_ti[r] = ti[v].mean(0); null_ta[r] = ta[v].mean(0)
            null_si[r] = si[v, :-1].mean(0); null_sa[r] = sa[v, :-1].mean(0)
            null_pl_i[r] = si[v, -1].mean(); null_pl_a[r] = sa[v, -1].mean()
            if per_pitcher_out and r < 50:
                pp_null.append(pd.DataFrame({"g": pr[group_col].values[v], "ti": ti[v, CKPT_NAMES.index("y23.8")], "ta": ta[v, CKPT_NAMES.index("y23.8")], "ti40": ti[v, CKPT_NAMES.index("y40")], "ta40": ta[v, CKPT_NAMES.index("y40")]}).groupby("g").mean())
        v = valid_any
        n = v.sum()
        for j, c in enumerate(EARLY):
            o_ti, o_ta = obs_ti[v, j].mean(), obs_ta[v, j].mean()
            results.append(dict(level=lev, checkpoint=c, n_pairs=n,
                                obs_ts_intent=o_ti, null_ts_intent=null_ti[:, j].mean(), null_sd=null_ti[:, j].std(),
                                z_intent=(o_ti - null_ti[:, j].mean()) / null_ti[:, j].std(),
                                p_intent=(np.sum(null_ti[:, j] >= o_ti) + 1) / (R + 1),
                                obs_sep_intent=obs_si[v, j].mean(), null_sep_intent=null_si[:, j].mean(),
                                obs_plate_intent=obs_si[v, -1].mean(), null_plate_intent=null_pl_i.mean(),
                                obs_ts_actual=o_ta, null_ts_actual=null_ta[:, j].mean(),
                                z_actual=(o_ta - null_ta[:, j].mean()) / null_ta[:, j].std(),
                                obs_sep_actual=obs_sa[v, j].mean(), null_sep_actual=null_sa[:, j].mean(),
                                obs_plate_actual=obs_sa[v, -1].mean(), null_plate_actual=null_pl_a.mean()))
        if per_pitcher_out:
            j = CKPT_NAMES.index("y23.8")
            j40 = CKPT_NAMES.index("y40")
            o = pd.DataFrame({"g": pr[group_col].values[v], "ti": obs_ti[v, j], "ta": obs_ta[v, j], "ti40": obs_ti[v, j40], "ta40": obs_ta[v, j40]}).groupby("g").agg(ti=("ti", "mean"), ta=("ta", "mean"), ti40=("ti40", "mean"), ta40=("ta40", "mean"), n=("ti", "size"))
            nl = pd.concat(pp_null).groupby("g").mean()
            for k in ["ti", "ta", "ti40", "ta40"]:
                o[f"null_{k}"] = nl[k]; o[f"excess_{k}"] = o[k] - nl[k]
            o.index.name = group_col
            o["level"] = lev
            pp_rows.append(o.reset_index())
    res = pd.DataFrame(results)
    return res, (pd.concat(pp_rows) if pp_rows else None)


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2025"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    levels = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("season_type", "game_type", "season_any")
    suffix = sys.argv[4] if len(sys.argv) > 4 else ""
    res, pp = run(year, tag, levels=levels)
    tag = tag + suffix
    pd.set_option("display.width", 250)
    txt = res.round(4).to_string(index=False)
    print(txt)
    open(f"tunnel/out/null_test_{year}{tag}.txt", "w").write(txt + "\n")
    res.to_csv(f"tunnel/out/null_test_{year}{tag}.csv", index=False)
    pp.to_csv(f"tunnel/out/pitcher_excess_{year}{tag}.csv", index=False)
