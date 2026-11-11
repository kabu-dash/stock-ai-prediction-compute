"""Stage 3 全銘柄ベースライン ランナー (フォールド単位ジョブ)。

事前登録 v1.0.0 §2〜§5 の実装。分割方式: 1 ジョブ = 1 フォールド、全シャードを
読んで適格銘柄全体で LightGBM をプール学習する (学習母集団がシャード分割に
依存しないため統計的同一性が自明)。--shards / --max-codes はパイロット・
スモーク用の部分実行で、マニフェストに partial=true を記録する。

使い方:
  python run_stage3.py --fold 0 --out-dir runs/full
  python run_stage3.py --fold 0 --shards 0 --max-codes 30 --out-dir runs/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as s3  # noqa: E402

from poc import config as poc_config  # noqa: E402
from poc.evaluation import (  # noqa: E402
    apply_calibration, blend_quantiles, fit_cqr, search_blend_weight,
)
from poc.features import FEATURE_NAMES, apply_clip, estimate_clip_bounds  # noqa: E402
from poc.models import (  # noqa: E402
    garch_fixed_per_stock, train_direction_model, train_quantile_models,
)

BASE = Path(__file__).resolve().parents[1]
PANEL_DIR = BASE / "phase3" / "data" / "full"
PRICES_DIR = BASE / "full_universe_prices"
MASTER_CSV = BASE / "universe_master_20260930.csv"
UNIVERSE500_CSV = BASE / "phase1" / "data" / "phase2" / "universe500.csv"
PHASE2_OOS = BASE / "phase1" / "data" / "phase2" / "oos_predictions.parquet"

PANEL_COLS = (["trade_date", "date_pos", "code"] + FEATURE_NAMES
              + ["sigma", "valid_bars_250", "med_turnover_250"]
              + [f"y_{h}" for h in s3.CHECKPOINTS]
              + [f"ytilde_{h}" for h in s3.CHECKPOINTS])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_master() -> dict[str, str]:
    df = pd.read_csv(MASTER_CSV, dtype={"code": str})
    return {c: s3.normalize_market(m) for c, m in zip(df["code"], df["market"])}


def load_universe500() -> set[str]:
    df = pd.read_csv(UNIVERSE500_CSV, dtype={"code": str})
    return set(df["code"].str.zfill(4))


def check_phase2_boundaries(folds) -> dict:
    """§2 の機械照合: Phase 2 OOS のフォールド別テスト範囲と比較する。"""
    if not PHASE2_OOS.exists():
        return {"checked": False, "reason": "phase2 oos_predictions.parquet not found"}
    o = pd.read_parquet(PHASE2_OOS, columns=["fold", "date_pos", "h"])
    o = o[o["h"] == 1]
    result = {"checked": True, "match": True, "folds": {}}
    for k, (_tr, te) in enumerate(folds):
        sub = o[o["fold"] == k]
        p2 = [int(sub["date_pos"].min()), int(sub["date_pos"].max())]
        mine = [int(te[0]), int(te[-1])]
        ok = p2 == mine
        result["match"] = result["match"] and ok
        result["folds"][str(k)] = {"stage3_test": mine, "phase2_test": p2, "match": ok}
    return result


def select_origins(cd: s3.CodeData, positions: np.ndarray, h: int,
                   phase: str, apply_origin_filter: bool,
                   excl_rows: list) -> np.ndarray:
    """origin フィルタ (§3) とガード (§4) を適用し、通過した date_pos を返す。
    除外は全件 excl_rows へ記録する (code, trade_date, date_pos, h, y_h,
    sigma_t, reason, phase)。ytilde が NaN の行は構造的に対象外 (件数のみ)。"""
    positions = np.asarray(positions, dtype=int)
    if len(positions) == 0:
        return positions
    has = cd.has_row(positions)
    cand = positions[has]
    if len(cand) == 0:
        return cand
    trade_dates = cd.col_at(cand, "trade_date")
    sigma = cd.col_at(cand, "sigma")
    y_h = cd.col_at(cand, f"y_{h}")
    ytilde = cd.col_at(cand, f"ytilde_{h}")
    vb = cd.col_at(cand, "valid_bars_250")
    keep = np.ones(len(cand), dtype=bool)

    def drop(mask: np.ndarray, reason: str) -> None:
        idx = np.where(mask & keep)[0]
        for i in idx:
            excl_rows.append({
                "code": cd.code, "trade_date": trade_dates[i],
                "date_pos": int(cand[i]), "h": h,
                "y_h": float(y_h[i]), "sigma_t": float(sigma[i]),
                "reason": reason, "phase": phase,
            })
        keep[idx] = False

    if apply_origin_filter:
        drop(vb < s3.VALID_BARS_MIN, "origin_valid_bars_250")
    reasons = cd.guard_reasons(cand, h, y_h, sigma)
    for reason in ("guard_main_window", "guard_backward_ret", "guard_sigma",
                   "safety_valve_abs_y"):
        drop(reasons == reason, reason)
    # 構造的除外: ターゲット未確定 (ytilde NaN)。§4 の記録対象ではないため
    # 行ログには入れず、呼び出し側で件数を数える
    struct = keep & np.isnan(ytilde)
    n_struct = int(struct.sum())
    keep[struct] = False
    if n_struct:
        excl_rows.append({"__struct_nan__": n_struct, "code": cd.code,
                          "h": h, "phase": phase})
    return cand[keep]


def frame_for(cd: s3.CodeData, positions: np.ndarray) -> pd.DataFrame:
    """パネル行を date_pos 順に取り出した DataFrame (特徴量 + 補助列)。"""
    idx = [cd._rowidx[int(p)] for p in positions]
    return cd.panel.iloc[idx]


# --- GARCH の銘柄並列化 ---
# 子プロセスは spawn で起動し、銘柄データを pickle で受け取る (fork は
# LightGBM の OpenMP スレッドと干渉して固まるため使わない: 2026-11-11 実測)。
# シードは (SEED, code, date_pos) の決定論関数のため、並列化しても
# 直列実行と数値は完全一致する。
def _garch_worker(task):
    code, r_pct, panel_code, origins, fit_end, h = task
    origins = np.asarray(origins, dtype=int)
    sigma_map = dict(zip(panel_code["date_pos"].to_numpy(dtype=int),
                         panel_code["sigma"].to_numpy(dtype=float)))
    sigma_o = np.array([sigma_map.get(int(p), np.nan) for p in origins])
    res = garch_fixed_per_stock(
        r_pct, sigma_o, origins, fit_end_pos=fit_end,
        checkpoints=(h,), taus=s3.TAUS,
        n_sims=poc_config.GARCH_N_SIMS_OOS, seed=s3.code_seed_base(code),
    )
    return code, (res[0] if res is not None
                  else np.full((len(origins), 5), np.nan))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--shards", default=None,
                    help="カンマ区切り (例 0,1)。未指定は全8シャード")
    ap.add_argument("--max-codes", type=int, default=None,
                    help="適格銘柄を先頭 N 件に制限 (スモーク/パイロット用)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-499", action="store_true",
                    help="共通499 (b) 用の予測を省略する")
    ap.add_argument("--garch-workers", type=int, default=1,
                    help="GARCH の銘柄並列数 (結果は直列と一致する)")
    args = ap.parse_args()

    t_all = time.time()
    shard_ids = ([int(x) for x in args.shards.split(",")]
                 if args.shards else list(range(8)))
    partial = len(shard_ids) < 8 or args.max_codes is not None
    out_dir = Path(args.out_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    market_map = load_master()
    codes500 = load_universe500()

    # ---------- 入力ロード ----------
    t0 = time.time()
    input_hashes: dict[str, str] = {}
    panels, prices = [], []
    for si in shard_ids:
        pp = PANEL_DIR / f"panel_full_shard_{si}.parquet"
        rp = PRICES_DIR / f"shard_{si}" / "prices" / f"shard_{si}.parquet"
        input_hashes[f"panel_full_shard_{si}.parquet"] = s3.sha256_file(pp)
        input_hashes[f"prices_shard_{si}.parquet"] = s3.sha256_file(rp)
        panels.append(pd.read_parquet(pp, columns=PANEL_COLS))
        prices.append(pd.read_parquet(rp))
    panel = pd.concat(panels, ignore_index=True)
    prices_all = pd.concat(prices, ignore_index=True)
    del panels, prices
    load_sec = time.time() - t0

    cal = (panel.loc[panel["date_pos"] <= 0].head(0))
    date_pairs = (panel[["date_pos", "trade_date"]].drop_duplicates()
                  .sort_values("date_pos"))
    n_total = int(date_pairs["date_pos"].max() + 1)
    date_to_pos = {pd.Timestamp(d): int(p) for p, d in
                   zip(date_pairs["date_pos"], date_pairs["trade_date"])}
    pos_to_date = {int(p): pd.Timestamp(d) for p, d in
                   zip(date_pairs["date_pos"], date_pairs["trade_date"])}
    prices_all["date"] = pd.to_datetime(prices_all["date"])
    log(f"loaded: panel rows={len(panel)} codes={panel['code'].nunique()} "
        f"n_total={n_total} in {load_sec:.0f}s")

    folds = s3.compute_folds(n_total)
    boundary_check = check_phase2_boundaries(folds)
    k = args.fold
    test_positions = folds[k][1]
    test_start = int(test_positions[0])

    # ---------- 銘柄データと適格判定 (§3 銘柄レベル) ----------
    t0 = time.time()
    codes_data: dict[str, s3.CodeData] = {}
    price_groups = {c: g for c, g in prices_all.groupby("code", sort=False)}
    empty_prices = prices_all.iloc[0:0]
    del prices_all
    eligible: list[str] = []
    ineligible: dict[str, str] = {}
    for code, pc in panel.groupby("code", sort=False):
        cd = s3.CodeData(code, pc, price_groups.get(code, empty_prices),
                         date_to_pos)
        codes_data[code] = cd
        row = cd.row_at(test_start)
        if row is None:
            ineligible[code] = "no_row_at_test_start"
        elif not (float(row["valid_bars_250"]) >= s3.VALID_BARS_MIN):
            ineligible[code] = "valid_bars_250"
        elif (pd.isna(row["log_turnover_20"])
              or float(row["log_turnover_20"]) < s3.LOG_TURNOVER_MIN):
            ineligible[code] = "log_turnover_20"
        else:
            eligible.append(code)
    eligible.sort()
    if args.max_codes is not None:
        eligible = eligible[:args.max_codes]
    elig_set = set(eligible)
    prep_sec = time.time() - t0
    log(f"fold{k}: eligible={len(eligible)} ineligible={len(ineligible)} "
        f"(prep {prep_sec:.0f}s)")

    (out_dir / "eligible_codes.json").write_text(json.dumps({
        "fold": k, "partial": partial,
        "eligible": eligible,
        "ineligible": ineligible,
        "test_start": test_start,
    }, ensure_ascii=False, indent=1))

    # 共通499: このジョブが読んだシャードに存在する 500 銘柄版コード
    common499 = sorted(c for c in codes500 if c in codes_data)

    # ---------- ホライズン別処理 (§5) ----------
    excl_rows: list = []
    pred_frames: list[pd.DataFrame] = []
    pred499_frames: list[pd.DataFrame] = []
    calib_info: dict[str, dict] = {}
    garch_stats: dict[str, dict] = {}
    timing: dict[str, float] = {"load": load_sec, "prep": prep_sec}

    for h in s3.CHECKPOINTS:
        th = time.time()
        b = s3.fold_boundaries(folds, k, h)
        subtrain_pos = np.arange(b["subtrain_start"], b["subtrain_end"] + 1)
        calib_pos = np.arange(b["calib_start"], b["calib_end"] + 1)

        # --- Sub-train / Calib の origin 選別 (適格銘柄のみ) ---
        train_sel: dict[str, np.ndarray] = {}
        calib_sel: dict[str, np.ndarray] = {}
        for code in eligible:
            cd = codes_data[code]
            train_sel[code] = select_origins(
                cd, subtrain_pos, h, "train", True, excl_rows)
            calib_sel[code] = select_origins(
                cd, calib_pos, h, "calib", True, excl_rows)
        # テスト origin: 適格銘柄 (主集合) と共通499 (ガードのみ)
        test_sel: dict[str, np.ndarray] = {}
        for code in eligible:
            test_sel[code] = select_origins(
                codes_data[code], test_positions, h, "test", True, excl_rows)
        test499_sel: dict[str, np.ndarray] = {}
        if not args.skip_499:
            # 共通499 の全銘柄 (適格・不適格を問わず) を、ガードのみの条件で選別
            for code in common499:
                test499_sel[code] = select_origins(
                    codes_data[code], test_positions, h, "test499", False,
                    excl_rows)

        def pooled(sel: dict[str, np.ndarray]) -> pd.DataFrame:
            frames = [frame_for(codes_data[c], sel[c]) for c in sel if len(sel[c])]
            return (pd.concat(frames, ignore_index=True) if frames
                    else panel.iloc[0:0].copy())

        train_df = pooled(train_sel)
        calib_df = pooled(calib_sel)
        if len(train_df) == 0 or len(calib_df) == 0:
            raise RuntimeError(f"fold{k} h={h}: train/calib が空 "
                               f"(train={len(train_df)}, calib={len(calib_df)})")

        # --- 学習 (Sub-train のみからクリップ境界を推定) ---
        clip_b = estimate_clip_bounds(train_df)
        X_tr = apply_clip(train_df[FEATURE_NAMES], clip_b)
        y_tr = train_df[f"ytilde_{h}"]
        pos_tr = train_df["date_pos"].to_numpy()
        qm = train_quantile_models(X_tr, y_tr, pos_tr, h)
        dm = train_direction_model(X_tr, (train_df[f"y_{h}"] > 0).astype(int),
                                   pos_tr, h)

        # --- GARCH (銘柄別・フォールド×ホライズン固定 fit) ---
        def garch_for(sel_by_code: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            tasks = []
            for code, origins in sel_by_code.items():
                if not len(origins):
                    continue
                cd = codes_data[code]
                tasks.append((code, cd.r_pct,
                              cd.panel[["date_pos", "sigma"]],
                              np.asarray(origins, dtype=int).tolist(),
                              b["subtrain_end"], h))
            if args.garch_workers > 1 and len(tasks) > 1:
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=args.garch_workers,
                                         mp_context=ctx) as ex:
                    return dict(ex.map(_garch_worker, tasks))
            return dict(_garch_worker(t) for t in tasks)

        calib_codes = {c: calib_sel[c] for c in eligible}
        test_codes = {c: test_sel[c] for c in eligible}
        g_cal = garch_for(calib_codes)
        g_test = garch_for(test_codes)
        g_499 = garch_for(test499_sel) if not args.skip_499 else {}
        n_fit_ok = sum(1 for c in eligible
                       if c in g_test and not np.isnan(g_test[c][0, 0])
                       if len(test_sel[c]))
        codes_with_test = [c for c in eligible if len(test_sel[c])]
        garch_stats[str(h)] = {
            "fit_end_pos": b["subtrain_end"],
            "codes_with_test_rows": len(codes_with_test),
            "codes_garch_ok_among_them": int(n_fit_ok),
            "garch_convergence_rate": (float(n_fit_ok / len(codes_with_test))
                                       if codes_with_test else None),
        }

        # --- Calib でブレンド重み + CQR + Isotonic を推定 ---
        def predict_block(df: pd.DataFrame, g_map: dict[str, np.ndarray],
                          sel: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            X = apply_clip(df[FEATURE_NAMES], clip_b)
            q_l = qm.predict(X)
            p_raw = dm.predict_proba(X)
            q_g = np.full((len(df), 5), np.nan)
            off = 0
            for code in sel:
                n = len(sel[code])
                if n:
                    q_g[off:off + n] = g_map[code]
                    off += n
            return q_l, p_raw, q_g

        ql_cal, pcal_raw, qg_cal = predict_block(calib_df, g_cal, calib_sel)
        y_cal = calib_df[f"ytilde_{h}"].to_numpy(dtype=np.float64)
        w = search_blend_weight(y_cal, qg_cal, ql_cal)
        qens_cal = blend_quantiles(qg_cal, ql_cal, w)
        calib = fit_cqr(y_cal, qens_cal)
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        ok_iso = ~np.isnan(pcal_raw)
        iso.fit(pcal_raw[ok_iso], (calib_df[f"y_{h}"].to_numpy()[ok_iso] > 0).astype(int))

        def build_output(df: pd.DataFrame, sel: dict[str, np.ndarray],
                         g_map: dict[str, np.ndarray]) -> pd.DataFrame:
            q_l, p_raw, q_g = predict_block(df, g_map, sel)
            q_ens = blend_quantiles(q_g, q_l, w)
            q_cal = apply_calibration(q_ens, h, calib)
            sigma = df["sigma"].to_numpy(dtype=np.float64)
            out = pd.DataFrame({
                "fold": k, "h": h, "code": df["code"].to_numpy(),
                "date_pos": df["date_pos"].to_numpy(),
                "trade_date": df["trade_date"].to_numpy(),
                "market": [market_map.get(c, "") for c in df["code"]],
                "med_turnover_250": df["med_turnover_250"].to_numpy(),
                "y": df[f"y_{h}"].to_numpy(),
                "ytilde": df[f"ytilde_{h}"].to_numpy(),
                "sigma": sigma,
                "p_cls_raw": p_raw,
                "p_cls": iso.predict(np.where(np.isnan(p_raw), 0.5, p_raw)),
            })
            for qi in range(5):
                out[f"qg_{qi}"] = q_g[:, qi]
                out[f"ql_{qi}"] = q_l[:, qi]
                out[f"qe_{qi}"] = q_ens[:, qi]
                out[f"qc_{qi}"] = q_cal[:, qi]
            out["p_q"] = s3.p_q_from_quantiles(q_cal, sigma, h)
            out["prob_up"] = 0.5 * out["p_cls"] + 0.5 * out["p_q"]
            return out

        test_df = pooled(test_sel)
        pred_frames.append(build_output(test_df, test_sel, g_test))
        if not args.skip_499:
            df499 = pooled(test499_sel)
            if len(df499):
                pred499_frames.append(build_output(df499, test499_sel, g_499))

        # 診断 (m-3): ソート前の生 LightGBM 分位点の交差率 (テスト主集合)
        if len(test_df):
            X_te = apply_clip(test_df[FEATURE_NAMES], clip_b)
            raw = np.column_stack([qm.models[float(tau)].predict(X_te[FEATURE_NAMES])
                                   for tau in s3.TAUS])
            cross_rate = float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))
        else:
            cross_rate = None
        n_te_fb = 0
        if len(test_df):
            qg_all = np.column_stack([pred_frames[-1][f"qg_{qi}"] for qi in range(5)])
            n_te_fb = int(np.sum(np.isnan(qg_all[:, 0]) & ~np.isnan(
                pred_frames[-1]["ql_0"].to_numpy())))
        calib_info[str(h)] = {
            **b, "w_lgbm": float(w), "d90": float(calib["d90"]),
            "d50": float(calib["d50"]),
            "n_train": int(len(train_df)), "n_calib": int(len(calib_df)),
            "n_test": int(len(test_df)),
            "best_iterations": qm.best_iterations,
            "direction_best_iteration": dm.best_iteration,
            "clip_bounds": {kk: list(vv) for kk, vv in clip_b.items()},
            "raw_cross_rate_lgbm_test": cross_rate,
            "n_test_rows_garch_fallback_lgbm_only": n_te_fb,
        }
        timing[f"h{h}"] = time.time() - th
        log(f"fold{k} h={h}: train={len(train_df)} calib={len(calib_df)} "
            f"test={len(test_df)} w={w} calib={calib} ({timing[f'h{h}']:.0f}s)")

    # ---------- 出力 ----------
    preds = pd.concat(pred_frames, ignore_index=True)
    atomic_to_parquet(preds, out_dir / "test_predictions.parquet")
    if pred499_frames:
        atomic_to_parquet(pd.concat(pred499_frames, ignore_index=True),
                          out_dir / "common499_predictions.parquet")
    struct_counts: dict[str, int] = {}
    clean_excl = []
    for r in excl_rows:
        if "__struct_nan__" in r:
            key = f"{r['phase']}_h{r['h']}"
            struct_counts[key] = struct_counts.get(key, 0) + r["__struct_nan__"]
        else:
            r = dict(r)
            r["trade_date"] = pos_to_date[int(r["date_pos"])]
            clean_excl.append(r)
    excl_df = pd.DataFrame(clean_excl, columns=[
        "code", "trade_date", "date_pos", "h", "y_h", "sigma_t", "reason",
        "phase", "fold"])
    if len(excl_df):
        excl_df["fold"] = k
    atomic_to_parquet(excl_df, out_dir / "exclusion_log.parquet")
    (out_dir / "calib_params.json").write_text(json.dumps(
        {"fold": k, "per_h": calib_info}, ensure_ascii=False, indent=1))

    import platform
    manifest = {
        "stage": "stage3_full_baseline",
        "preregistration": "STAGE3_FULL_BASELINE_PREREGISTRATION_20261111_v1.0.0",
        "fold": k, "shards": shard_ids, "partial": partial,
        "max_codes": args.max_codes,
        "n_total_dates": n_total,
        "fold_boundary_check_vs_phase2": boundary_check,
        "input_sha256": input_hashes,
        "seed_rule": "garch seed = md5(f'{SEED}:{code}')[:8] mod 2^31, "
                     "+ date_pos inside garch_fixed_per_stock",
        "poc_seed": s3.SEED,
        "n_eligible": len(eligible), "n_ineligible": len(ineligible),
        "n_test_prediction_rows": int(len(preds)),
        "struct_nan_counts": struct_counts,
        "garch_stats": garch_stats,
        "timing_sec": {**timing, "total": time.time() - t_all},
        "env": {"python": platform.python_version(),
                "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out_dir / "manifest_fragment.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=1, default=str))
    log(f"fold{k} DONE: preds={len(preds)} excl={len(excl_df)} "
        f"total={time.time() - t_all:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
