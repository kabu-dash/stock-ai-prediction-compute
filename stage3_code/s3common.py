"""Stage 3 全銘柄ベースライン共通部品。

事前登録 v1.0.0 (STAGE3_FULL_BASELINE_PREREGISTRATION_20261111_v1.0.0.md) の
§2〜§6 のデータ側・評価側の定義を、phase0/src/poc の既存機械を再利用して実装する。
poc 本体には一切手を入れない。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# phase0/src/poc を import 可能にする (phase1 backend の app.ml.poc と実質同一である
# ことを diff で確認済み: 差分は evaluation.py の docstring 文言のみ)
POC_SRC = Path(__file__).resolve().parents[1] / "phase0" / "src"
if str(POC_SRC) not in sys.path:
    sys.path.insert(0, str(POC_SRC))

from poc import config as poc_config  # noqa: E402
from poc.evaluation import pinball_loss, purged_walk_forward  # noqa: E402

CHECKPOINTS = poc_config.CHECKPOINTS  # (1, 5, 10, 20)
TAUS = poc_config.TAUS
SEED = poc_config.SEED
EMBARGO = poc_config.EMBARGO  # 5
N_FOLDS = poc_config.N_FOLDS  # 6
TEST_LEN = poc_config.TEST_LEN  # 60
CALIB_LEN = 60  # 事前登録 §5: Calib は末尾60営業日
LOG_TURNOVER_MIN = float(np.log(1e7))  # §3: ln(1e7) ≒ 16.11809565
VALID_BARS_MIN = 200  # §3
SIGMA_GUARD_MAX = 0.50  # §4 後方 σ 汚染ガード
ABS_RET_GUARD = 2.0  # §4 後方ガード / 安全弁の閾値
BACKWARD_WINDOW = 250  # §4 後方ガードの窓

PANEL_COLS_BASE = ["trade_date", "date_pos", "code", "sigma",
                   "valid_bars_250", "med_turnover_250"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def code_seed_base(code: str) -> int:
    """m-1 処分: GARCH 乱数シードを (SEED, code) の決定論ハッシュにする。

    garch_fixed_per_stock は内部で seed + date_pos を使うため、ここで返す
    base を seed 引数に渡すことで実効シードは (SEED, code, date_pos) の
    決定論関数になる。シャード分割・実行順序に依存しない。
    """
    digest = hashlib.md5(f"{SEED}:{code}".encode()).hexdigest()
    return int(digest[:8], 16) % (2 ** 31)


def compute_folds(n_total: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """500銘柄版と同一の境界生成 (purged_walk_forward、末尾アンカー)。"""
    n_valid = n_total - poc_config.MAX_H
    folds = purged_walk_forward(n_valid, h=poc_config.MAX_H)
    assert len(folds) == N_FOLDS, f"folds={len(folds)}"
    return folds


def fold_boundaries(folds, k: int, h: int) -> dict:
    """事前登録 §5 の境界式 (フォールド k・ホライズン h)。

    T_tr_max = test_start_k - h - 5 - 1
    Calib    = [T_tr_max - 59, T_tr_max]          (60営業日)
    Sub-train= [0, T_tr_max - 60 - h - 5]
    """
    test = folds[k][1]
    test_start = int(test[0])
    t_tr_max = test_start - h - EMBARGO - 1
    return {
        "test_start": test_start,
        "test_end": int(test[-1]),
        "t_tr_max": t_tr_max,
        "calib_start": t_tr_max - (CALIB_LEN - 1),
        "calib_end": t_tr_max,
        "subtrain_start": 0,
        "subtrain_end": t_tr_max - CALIB_LEN - h - EMBARGO,
    }


def normalize_market(raw: str) -> str:
    for prefix in ("プライム", "スタンダード", "グロース"):
        if str(raw).startswith(prefix):
            return prefix
    return str(raw)


class CodeData:
    """1銘柄分のカレンダー整列済み系列とパネル行。"""

    def __init__(self, code: str, panel_code: pd.DataFrame,
                 prices_code: pd.DataFrame, date_to_pos: dict):
        self.code = code
        pc = panel_code.sort_values("date_pos")
        self.panel = pc
        self.positions = pc["date_pos"].to_numpy(dtype=int)
        self._rowidx = {int(p): i for i, p in enumerate(self.positions)}

        n_total = len(date_to_pos)
        # --- 価格シャードから XTKS カレンダー上の系列を構築 ---
        adj = np.full(n_total, np.nan)
        v_flag = np.zeros(n_total, dtype=float)
        if len(prices_code):
            pos = prices_code["date"].map(date_to_pos)
            keep = pos.notna()
            ppos = pos[keep].to_numpy(dtype=int)
            adj[ppos] = prices_code.loc[keep, "adj_close"].to_numpy(dtype=float)
            vol = prices_code.loc[keep, "volume"].to_numpy(dtype=float)
            v_flag[ppos] = ((~np.isnan(vol)) & (vol > 0)).astype(float)
        self.v_flag = v_flag
        self.v_cum = np.concatenate([[0.0], np.cumsum(v_flag)])
        # build_stock_features と同一: reindex 後に ffill(limit=2) した adj で
        # リターンを作る (Phase 2 の garch_inputs と同じ作り方)
        adj_ff = pd.Series(adj).ffill(limit=2).to_numpy()
        self.adj_ff = adj_ff
        r_pct = np.full(n_total, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            la = np.log(adj_ff)
        r_pct[1:] = np.diff(la) * 100.0
        self.r_pct = r_pct
        r_log = r_pct / 100.0
        bad = ((np.abs(r_log) > ABS_RET_GUARD) & ~np.isnan(r_log)).astype(float)
        self.bad_ret_cum = np.concatenate([[0.0], np.cumsum(bad)])

    # --- パネル行へのアクセス ---
    def row_at(self, pos: int):
        i = self._rowidx.get(int(pos))
        return None if i is None else self.panel.iloc[i]

    def col_at(self, positions: np.ndarray, col: str) -> np.ndarray:
        idx = np.array([self._rowidx.get(int(p), -1) for p in positions])
        vals = self.panel[col].to_numpy(dtype=float)
        out = np.full(len(positions), np.nan)
        ok = idx >= 0
        out[ok] = vals[idx[ok]]
        return out

    def has_row(self, positions: np.ndarray) -> np.ndarray:
        return np.array([int(p) in self._rowidx for p in positions])

    # --- ガード (§4) ---
    def guard_reasons(self, positions: np.ndarray, h: int,
                      y_h: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """各 origin の除外理由 (None = 通過) を返す。判定順は固定:
        main (閉区間) -> backward_ret -> sigma -> safety_valve。"""
        t = np.asarray(positions, dtype=int)
        reasons = np.full(len(t), None, dtype=object)
        # 主ガード: 閉区間 [t, t+h] の全 h+1 日で v == 1
        win = self.v_cum[t + h + 1] - self.v_cum[t]
        reasons[win < (h + 1)] = "guard_main_window"
        # 後方ガード1: 過去250日 [t-249, t] に |r_s| > 2.0
        lo = np.maximum(0, t - (BACKWARD_WINDOW - 1))
        badn = self.bad_ret_cum[t + 1] - self.bad_ret_cum[lo]
        m = (reasons == None) & (badn > 0)  # noqa: E711
        reasons[m] = "guard_backward_ret"
        # 後方ガード2: sigma_t > 0.50
        m = (reasons == None) & (sigma > SIGMA_GUARD_MAX)  # noqa: E711
        reasons[m] = "guard_sigma"
        # 安全弁: |y_h| > 2.0
        m = (reasons == None) & (np.abs(y_h) > ABS_RET_GUARD)  # noqa: E711
        reasons[m] = "safety_valve_abs_y"
        return reasons


def p_q_from_quantiles(q_tilde: np.ndarray, sigma: np.ndarray, h: int) -> np.ndarray:
    """pipeline.build_summary と同一式のベクトル化: 較正済み規格化分位点を
    対数リターン空間へ戻し、F(0) の線形補間から p_q = clip(1 - F(0), .05, .95)。"""
    q_ret = np.sort(q_tilde * (sigma * np.sqrt(h))[:, None], axis=1)
    n = len(q_ret)
    cdf0 = np.empty(n)
    lo_edge = q_ret[:, 0] >= 0.0
    hi_edge = q_ret[:, -1] <= 0.0
    cdf0[lo_edge] = 0.05
    cdf0[hi_edge] = 0.95
    mid = ~(lo_edge | hi_edge)
    if mid.any():
        qm = q_ret[mid]
        # np.interp(0, q_row, TAUS) の行単位版
        idx = np.array([np.searchsorted(row, 0.0) for row in qm])
        idx = np.clip(idx, 1, len(TAUS) - 1)
        x0 = qm[np.arange(len(qm)), idx - 1]
        x1 = qm[np.arange(len(qm)), idx]
        y0 = TAUS[idx - 1]
        y1 = TAUS[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(x1 > x0, (0.0 - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
        cdf0[mid] = y0 + frac * (y1 - y0)
    return np.clip(1.0 - cdf0, 0.05, 0.95)


def winkler_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    """区間 [lo, hi] (名目 1-alpha) の Winkler スコア (行単位)。"""
    width = hi - lo
    return (width
            + (2.0 / alpha) * (lo - y) * (y < lo)
            + (2.0 / alpha) * (y - hi) * (y > hi))


def dm_test(d: np.ndarray, h: int, lag_override: int | None = None) -> dict:
    """日次損失差系列 d に対する DM 検定 (Newey-West Bartlett + HLN 補正)。

    M-4 処分の定式化: gamma_m = (1/T) * sum_t (d_t - dbar)(d_{t-m} - dbar)、
    Var(dbar) = (1/T) [gamma_0 + 2 * sum_{m=1..L} (1 - m/(L+1)) gamma_m]、
    L = h - 1。HLN: DM * sqrt((T + 1 - 2h + h(h-1)/T) / T)、p は t(T-1)。
    """
    from scipy import stats as sstats

    d = np.asarray(d, dtype=np.float64)
    d = d[~np.isnan(d)]
    T = len(d)
    L = (h - 1) if lag_override is None else lag_override
    if T < 10:
        return {"T": int(T), "lag": int(L), "mean_diff": float(np.mean(d)) if T else None,
                "dm_stat": None, "p_two_sided": None, "p_one_sided_a_better": None,
                "dm_hln": None, "p_hln_two_sided": None}
    dbar = float(np.mean(d))
    dev = d - dbar
    var = float(np.dot(dev, dev) / T)
    for m in range(1, L + 1):
        gamma_m = float(np.dot(dev[m:], dev[:-m]) / T)
        var += 2.0 * (1.0 - m / (L + 1)) * gamma_m
    var_dbar = var / T
    if var_dbar <= 0:
        return {"T": int(T), "lag": int(L), "mean_diff": dbar,
                "dm_stat": None, "p_two_sided": None,
                "p_one_sided_a_better": None, "dm_hln": None,
                "p_hln_two_sided": None}
    dm = dbar / np.sqrt(var_dbar)
    p_two = float(2.0 * (1.0 - sstats.norm.cdf(abs(dm))))
    p_one = float(sstats.norm.cdf(dm))  # H1: d < 0 (A の損失が小さい)
    adj = (T + 1 - 2 * h + h * (h - 1) / T) / T
    dm_hln = float(dm * np.sqrt(adj)) if adj > 0 else None
    p_hln = (float(2.0 * (1.0 - sstats.t.cdf(abs(dm_hln), df=T - 1)))
             if dm_hln is not None else None)
    return {"T": int(T), "lag": int(L), "mean_diff": dbar,
            "dm_stat": float(dm), "p_two_sided": p_two,
            "p_one_sided_a_better": p_one,
            "dm_hln": dm_hln, "p_hln_two_sided": p_hln}
