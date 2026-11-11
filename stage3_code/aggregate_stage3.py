"""Stage 3 全銘柄ベースライン 集計スクリプト。

run_stage3.py のフォールド別出力を統合し、事前登録 v1.0.0 §6 の指標と
判定表 S3-G1〜G4、日次平均損失系列、有効評価 origin マスク (SHA-256)、
市場区分別・流動性三分位別の内訳、共通499 の2通り集計 (§7)、
条件 C の判定記録 (§10) を生成する。集計は float64 で行う。

使い方: python aggregate_stage3.py --runs-dir runs/full --out-dir runs/full/aggregate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s3common as s3  # noqa: E402
from poc.evaluation import pinball_loss  # noqa: E402

QC = [f"qc_{i}" for i in range(5)]


def core_metrics(df: pd.DataFrame, h: int) -> dict:
    """§6 の指標一式 (規格化空間が主、対数リターン空間は参考)。"""
    sub = df[df["h"] == h]
    ok = (~np.isnan(sub["ytilde"].to_numpy(dtype=np.float64))
          & ~np.isnan(sub[QC].to_numpy(dtype=np.float64)).any(axis=1))
    sub = sub[ok]
    n = len(sub)
    if n == 0:
        return {"n": 0}
    y = sub["ytilde"].to_numpy(dtype=np.float64)
    yl = sub["y"].to_numpy(dtype=np.float64)
    sigma = sub["sigma"].to_numpy(dtype=np.float64)
    q = sub[QC].to_numpy(dtype=np.float64)
    q_b0 = np.tile(sstats.norm.ppf(s3.TAUS), (n, 1))
    loss = pinball_loss(y, q)
    loss_b0 = pinball_loss(y, q_b0)
    pin = float(loss.mean())
    pin_b0 = float(loss_b0.mean())
    # 対数リターン空間 (参考): q_lr = q_tilde * sigma * sqrt(h)
    q_lr = q * (sigma * np.sqrt(h))[:, None]
    pin_lr = float(pinball_loss(yl, q_lr).mean())
    pin_lr_b0 = float(pinball_loss(yl, q_b0 * (sigma * np.sqrt(h))[:, None]).mean())
    # 方向指標: ラベル I(y > 0)
    y_up = (yl > 0).astype(float)
    p_q = sub["p_q"].to_numpy(dtype=np.float64)
    p_h = sub["prob_up"].to_numpy(dtype=np.float64)
    return {
        "n": int(n),
        "pinball_norm": pin,
        "pinball_b0_norm": pin_b0,
        "skill_vs_b0": float(1.0 - pin / pin_b0) if pin_b0 > 0 else None,
        "crps_norm": 2.0 * pin,
        "coverage90": float(np.mean((y >= q[:, 0]) & (y <= q[:, 4]))),
        "coverage50": float(np.mean((y >= q[:, 1]) & (y <= q[:, 3]))),
        "winkler90_norm": float(s3.winkler_score(y, q[:, 0], q[:, 4], 0.10).mean()),
        "winkler50_norm": float(s3.winkler_score(y, q[:, 1], q[:, 3], 0.50).mean()),
        "pinball_logret_ref": pin_lr,
        "skill_vs_b0_logret_ref": (float(1.0 - pin_lr / pin_lr_b0)
                                   if pin_lr_b0 > 0 else None),
        "direction_accuracy_pq": float(np.mean((p_q > 0.5) == (y_up > 0))),
        "brier_pq": float(np.mean((p_q - y_up) ** 2)),
        "direction_accuracy_hybrid": float(np.mean((p_h > 0.5) == (y_up > 0))),
        "brier_hybrid": float(np.mean((p_h - y_up) ** 2)),
        "base_rate_up": float(np.mean(y_up)),
        "interval_collapse_iqr_rate": float(np.mean(q[:, 3] == q[:, 1])),
        "interval_collapse_90_rate": float(np.mean(q[:, 4] == q[:, 0])),
        "monotonic_violation_rate": float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1))),
    }


def daily_losses(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, dp), grp in df.groupby(["h", "date_pos"]):
        y = grp["ytilde"].to_numpy(dtype=np.float64)
        q = grp[QC].to_numpy(dtype=np.float64)
        q_b0 = np.tile(sstats.norm.ppf(s3.TAUS), (len(grp), 1))
        ok = ~np.isnan(y) & ~np.isnan(q).any(axis=1)
        if ok.sum() == 0:
            continue
        rows.append({
            "h": int(h), "date_pos": int(dp),
            "trade_date": grp["trade_date"].iloc[0],
            "n_codes": int(ok.sum()),
            "loss_baseline_a": float(pinball_loss(y[ok], q[ok]).mean()),
            "loss_b0": float(pinball_loss(y[ok], q_b0[ok]).mean()),
        })
    return pd.DataFrame(rows).sort_values(["h", "date_pos"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    runs = Path(args.runs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(runs.glob("fold*"))
    preds, preds499, fragments, calib_all = [], [], {}, {}
    eligible_by_fold: dict[str, list] = {}
    excl_frames = []
    for fd in fold_dirs:
        k = int(fd.name.replace("fold", ""))
        preds.append(pd.read_parquet(fd / "test_predictions.parquet"))
        f499 = fd / "common499_predictions.parquet"
        if f499.exists():
            preds499.append(pd.read_parquet(f499))
        fragments[str(k)] = json.loads((fd / "manifest_fragment.json").read_text())
        calib_all[str(k)] = json.loads((fd / "calib_params.json").read_text())
        ej = json.loads((fd / "eligible_codes.json").read_text())
        eligible_by_fold[str(k)] = ej["eligible"]
        excl_frames.append(pd.read_parquet(fd / "exclusion_log.parquet"))
    preds = pd.concat(preds, ignore_index=True)
    preds499 = (pd.concat(preds499, ignore_index=True) if preds499
                else pd.DataFrame(columns=preds.columns))
    excl = pd.concat(excl_frames, ignore_index=True) if excl_frames else pd.DataFrame()
    folds_done = sorted(int(k) for k in fragments)
    complete = folds_done == list(range(6))

    # ---------- origin マスク (m-2: 保存 + SHA-256) ----------
    mask = (preds[["fold", "code", "date_pos", "h"]]
            .sort_values(["fold", "h", "code", "date_pos"]).reset_index(drop=True))
    mask_path = out / "origin_mask.parquet"
    mask.to_parquet(mask_path, index=False)
    mask_sha = s3.sha256_file(mask_path)

    # ---------- 日次平均損失系列と DM (S3-G4) ----------
    dl = daily_losses(preds)
    dl.to_parquet(out / "daily_losses_stage3_baseline.parquet", index=False)
    dm: dict[str, dict] = {}
    for h in s3.CHECKPOINTS:
        sub = dl[dl["h"] == h].sort_values("date_pos")
        d = (sub["loss_baseline_a"] - sub["loss_b0"]).to_numpy(dtype=np.float64)
        entry = {"lag_h_minus_1": s3.dm_test(d, h)}
        if h == 1:
            entry["lag_5_sensitivity"] = s3.dm_test(d, h, lag_override=5)
        dm[str(h)] = entry

    # ---------- 主指標と判定表 ----------
    metrics = {str(h): core_metrics(preds, h) for h in s3.CHECKPOINTS}
    g1 = {str(h): bool(0.86 <= metrics[str(h)]["coverage90"] <= 0.94
                       and 0.44 <= metrics[str(h)]["coverage50"] <= 0.56)
          for h in s3.CHECKPOINTS}
    g3_ok = all(metrics[str(h)].get("monotonic_violation_rate", 1.0) == 0.0
                for h in s3.CHECKPOINTS)
    gates = {
        "S3-G1_coverage": {"per_h_pass": g1, "pass": bool(all(g1.values()))},
        "S3-G2_skill_vs_b0": {"per_h": {str(h): metrics[str(h)]["skill_vs_b0"]
                                        for h in s3.CHECKPOINTS},
                              "note": "参照値の記録 (合否ゲートではない)"},
        "S3-G3_monotonic": {"pass": bool(g3_ok),
                            "violation_rate_per_h": {
                                str(h): metrics[str(h)]["monotonic_violation_rate"]
                                for h in s3.CHECKPOINTS}},
        "S3-G4_dm_reference": dm,
    }

    # ---------- 内訳 ----------
    by_fold = {str(k): {str(h): core_metrics(preds[preds["fold"] == k], h)
                        for h in s3.CHECKPOINTS} for k in folds_done}
    by_market = {}
    for seg in sorted(preds["market"].unique()):
        by_market[seg] = {str(h): core_metrics(preds[preds["market"] == seg], h)
                          for h in s3.CHECKPOINTS}
    # 流動性三分位: フォールドごとに適格銘柄の med_turnover_250 (テスト開始行)
    # で三分位を切り、(fold, code) へラベル付けする
    ts_rows = preds.drop_duplicates(["fold", "code"])[
        ["fold", "code", "med_turnover_250"]]
    labels = {}
    for k, grp in ts_rows.groupby("fold"):
        g = grp.dropna(subset=["med_turnover_250"])
        if len(g) >= 3:
            qs = g["med_turnover_250"].quantile([1 / 3, 2 / 3]).to_numpy()
            for code, v in zip(g["code"], g["med_turnover_250"]):
                labels[(k, code)] = ("low" if v <= qs[0]
                                      else "mid" if v <= qs[1] else "high")
    preds_l = preds.copy()
    preds_l["liquidity_tercile"] = [labels.get((f, c)) for f, c in
                                    zip(preds_l["fold"], preds_l["code"])]
    by_tercile = {}
    for ter in ("low", "mid", "high"):
        by_tercile[ter] = {str(h): core_metrics(
            preds_l[preds_l["liquidity_tercile"] == ter], h)
            for h in s3.CHECKPOINTS}
    # 全フォールド適格な固定部分集合 (感度指標)
    if eligible_by_fold:
        sets = [set(v) for v in eligible_by_fold.values()]
        fixed_codes = set.intersection(*sets) if sets else set()
    else:
        fixed_codes = set()
    fixed_metrics = {str(h): core_metrics(preds[preds["code"].isin(fixed_codes)], h)
                     for h in s3.CHECKPOINTS}

    # ---------- 共通499 (§7) ----------
    codes500 = set(preds499["code"].unique()) if len(preds499) else set()
    main499 = preds[preds["code"].isin(codes500)]
    common499_a = {str(h): core_metrics(main499, h) for h in s3.CHECKPOINTS}
    common499_b = ({str(h): core_metrics(preds499, h) for h in s3.CHECKPOINTS}
                   if len(preds499) else {})

    excl_breakdown = {}
    if len(excl):
        excl_breakdown = (excl.groupby(["phase", "reason"]).size()
                          .astype(int).to_dict())
        excl_breakdown = {f"{k[0]}|{k[1]}": v for k, v in excl_breakdown.items()}

    # ---------- 条件 C (§10) ----------
    cond_c = {
        "all_6_folds_present": complete,
        "folds_present": folds_done,
        "predictions_rows": int(len(preds)),
        "gate_table_produced": True,
        "exclusion_breakdown_recorded": True,
        "condition_C_complete": bool(complete and len(preds) > 0),
        "note": "関門の合否は条件 C の成否とは別に扱う (事前登録 §10)",
    }

    metrics_doc = {
        "pooled": metrics,
        "gates": gates,
        "by_fold": by_fold,
        "by_market_segment": by_market,
        "by_liquidity_tercile": by_tercile,
        "fixed_universe_all_folds": {"n_codes": len(fixed_codes),
                                     "metrics": fixed_metrics},
        "common499": {"n_codes": len(codes500),
                      "a_eligible_and_guarded": common499_a,
                      "b_guard_only_all": common499_b},
        "n_rows": {"main": int(len(preds)), "common499_b": int(len(preds499)),
                   "exclusions_logged": int(len(excl))},
        "exclusion_breakdown": excl_breakdown,
        "condition_C": cond_c,
    }
    (out / "metrics_stage3.json").write_text(json.dumps(
        metrics_doc, ensure_ascii=False, indent=1, default=str))
    (out / "eligible_codes_by_fold.json").write_text(json.dumps(
        eligible_by_fold, ensure_ascii=False, indent=1))
    manifest = {
        "stage": "stage3_full_baseline_aggregate",
        "origin_mask_file": "origin_mask.parquet",
        "origin_mask_sha256": mask_sha,
        "origin_mask_rows": int(len(mask)),
        "folds": fragments,
        "calib_params": calib_all,
        "condition_C": cond_c,
    }
    (out / "run_manifest.json").write_text(json.dumps(
        manifest, ensure_ascii=False, indent=1, default=str))
    print(f"aggregate done: folds={folds_done} rows={len(preds)} "
          f"mask_sha={mask_sha[:16]}... condC={cond_c['condition_C_complete']}")


if __name__ == "__main__":
    main()
