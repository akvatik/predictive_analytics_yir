"""Этап 5 — слой 2: индикатор «здоровья» вала H(t) (EWMA + CUSUM).

Вход — остаток режимной модели r(t)=F1−f_reg (``modero_residuals.parquet``).
Идея (Вариант C): устойчивый дрейф остатка = смещение связи режим→F1 =
ранний признак потери жёсткости. Слой 2 извлекает медленный дрейф из шума и
поднимает тревогу с контролируемой частотой ложных срабатываний.

Ключевые методологические решения (согласовано):
* **Пороги — по OUT-OF-SAMPLE здоровому окну** (блочная CV на train,
  ``resid_oos_train``), не по in-sample остаткам train (иначе занижение σ →
  узкие пороги → ложные тревоги). Опорное окно — раннее, held-out, in-envelope.
* **Флаг ``in_envelope``** — ключевые драйверы (Tcnd1/2, Pa) в пределах
  train-огибающей. Тревоги засчитываются ТОЛЬКО на in-envelope часах: дрейф вне
  огибающей = экстраполяция/зажим деревьев, артефакт, а не «здоровье».
* **Сезон не вычитаем** (риск убрать настоящий дрейф) — ограничиваемся
  in-envelope тревогами; сезонная чувствительность зафиксирована как ограничение.

Запуск:  ``python -m src.health``
Артефакты: ``outputs/health_series.parquet``, ``outputs/health_indicator.png``.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval import SPLIT_BOUNDS

# Консоль Windows (cp1251) → UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RANDOM_STATE = 42
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
RESIDUALS_PARQUET = os.path.join(OUTPUTS_DIR, "modero_residuals.parquet")
CLEAN_CSV = os.path.join(OUTPUTS_DIR, "modero_clean.csv")
HEALTH_PARQUET = os.path.join(OUTPUTS_DIR, "health_series.parquet")
HEALTH_PNG = os.path.join(OUTPUTS_DIR, "health_indicator.png")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("health")

# --- Параметры индикатора --------------------------------------------------
ENVELOPE_COLS = ["Tcnd1", "Tcnd2", "Pa"]  # ключевые драйверы для огибающей
EWMA_LAMBDA = 0.03      # сглаживание EWMA (halflife ≈ сутки) — медленный дрейф
CUSUM_K_SIGMA = 0.5     # «slack» CUSUM = 0.5σ (порог обнаружения сдвига ≈1σ)
# Пороги калибруются ЭМПИРИЧЕСКИ по здоровому окну под целевую частоту ложных
# тревог (классические L·σ-пределы неверны: остаток сильно автокоррелирован).
TARGET_FA = 0.01        # целевая доля ложных тревог на опорном здоровом окне
TRAIN_END = SPLIT_BOUNDS["train"][1]  # граница train (для огибающей и опоры)


def load_inputs() -> pd.DataFrame:
    """Загрузить остатки + драйверы огибающей, отсортировать по времени."""
    # Остатки слоя 1 (operational + OOS) и температуры/мощность для огибающей.
    res = pd.read_parquet(RESIDUALS_PARQUET)
    cln = pd.read_csv(CLEAN_CSV, parse_dates=["time"])
    df = res.merge(cln[["time", *ENVELOPE_COLS]], on="time", how="left")
    return df.sort_values("time").reset_index(drop=True)


def add_envelope(df: pd.DataFrame) -> pd.DataFrame:
    """Флаг in_envelope: ключевые драйверы в пределах [min,max] train."""
    df = df.copy()
    # Границы считаем ТОЛЬКО по train (раннее опорное здоровое поведение).
    tr = df["time"] <= pd.Timestamp(TRAIN_END)
    inside = pd.Series(True, index=df.index)
    for c in ENVELOPE_COLS:
        lo, hi = df.loc[tr, c].min(), df.loc[tr, c].max()
        inside &= df[c].between(lo, hi)
    df["in_envelope"] = inside
    log.info("in_envelope: %.1f%% часов внутри train-огибающей (%s)",
             100 * inside.mean(), ENVELOPE_COLS)
    return df


def reference_sigma(df: pd.DataFrame) -> tuple[float, float]:
    """Опорные μ0 и σ по OOS здоровому окну (train, in-envelope)."""
    # Здоровое окно: train, out-of-sample остатки, только in-envelope.
    ref = df.loc[df["resid_oos_train"].notna() & df["in_envelope"], "resid_oos_train"]
    mu0, sigma = float(ref.mean()), float(ref.std())
    log.info("Опора (OOS train, in-env): n=%d  μ0=%+.5f  σ=%.5f", len(ref), mu0, sigma)
    return mu0, sigma


def _ewma(resid: pd.Series) -> pd.Series:
    """EWMA остатка — сглаживание, проявляющее медленный сдвиг среднего."""
    return resid.ewm(alpha=EWMA_LAMBDA, adjust=False).mean()


def _cusum(resid: pd.Series, mu0: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Двусторонний CUSUM: накопление отклонений от μ0 со slack k."""
    # Рекуррентно копим положительные/отрицательные сдвиги (без сброса по сегментам).
    x = resid.to_numpy()
    cpos = np.zeros(len(x)); cneg = np.zeros(len(x))
    for i in range(1, len(x)):
        d = x[i] - mu0
        cpos[i] = max(0.0, cpos[i - 1] + d - k)
        cneg[i] = max(0.0, cneg[i - 1] - d - k)
    return cpos, cneg


def calibrate(df: pd.DataFrame, mu0: float, sigma: float) -> dict:
    """Эмпирические пороги по OOS здоровому окну под целевую частоту ложных тревог.

    Классические L·σ-пределы неприменимы (остаток автокоррелирован), поэтому
    порог EWMA и порог CUSUM берём как высокие квантили их РАСПРЕДЕЛЕНИЯ на
    здоровом ряду — так доля ложных тревог на опоре ≈ TARGET_FA по построению.
    """
    # Здоровый ряд: train, in-envelope, во времени; индикаторы на OOS-остатке.
    ref = df.loc[df["resid_oos_train"].notna() & df["in_envelope"]].sort_values("time")
    rr = ref["resid_oos_train"]
    h_ref = _ewma(rr)
    k = CUSUM_K_SIGMA * sigma
    cpos_ref, cneg_ref = _cusum(rr, mu0, k)
    cmax_ref = np.maximum(cpos_ref, cneg_ref)

    # Порог EWMA — двусторонний квантиль |H−μ0|; порог CUSUM — квантиль max(C+,C−).
    ewma_limit = float(np.quantile(np.abs(h_ref - mu0), 1 - TARGET_FA))
    cusum_h = float(np.quantile(cmax_ref, 1 - TARGET_FA))
    log.info("Калибровка (OOS, in-env, FA=%.0f%%): EWMA-предел=±%.5f, CUSUM h=%.5f",
             100 * TARGET_FA, ewma_limit, cusum_h)
    return {"mu0": mu0, "sigma": sigma, "k": k,
            "ewma_limit": ewma_limit, "cusum_h": cusum_h}


def build_health(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Собрать индикатор H(t): EWMA + CUSUM, эмпирические пороги, тревоги in-envelope."""
    # Опора μ0/σ и эмпирические пороги из OOS здорового окна.
    mu0, sigma = reference_sigma(df)
    cal = calibrate(df, mu0, sigma)
    out = df.sort_values("time").reset_index(drop=True).copy()

    # Индикаторы по operational остатку (на val/test это честный OOS).
    out["H_ewma"] = _ewma(out["resid_gbm"])
    cpos, cneg = _cusum(out["resid_gbm"], mu0, cal["k"])
    out["cusum_pos"], out["cusum_neg"] = cpos, cneg
    # Пределы — константы из калибровки (для рисунка/сравнения).
    out["ewma_lo"], out["ewma_hi"] = mu0 - cal["ewma_limit"], mu0 + cal["ewma_limit"]
    out["cusum_h"] = cal["cusum_h"]

    # Сырое срабатывание (EWMA вне предела ИЛИ CUSUM выше порога).
    raw_alarm = ((out["H_ewma"] > out["ewma_hi"]) | (out["H_ewma"] < out["ewma_lo"])
                 | (out["cusum_pos"] > cal["cusum_h"]) | (out["cusum_neg"] > cal["cusum_h"]))
    # Тревога — ТОЛЬКО на in-envelope часах (вне огибающей = артефакт экстраполяции).
    out["alarm"] = raw_alarm & out["in_envelope"]

    # Факт. доля ложных тревог на опорном окне (контроль калибровки).
    ref_mask = (out["time"] <= pd.Timestamp(TRAIN_END)) & out["in_envelope"]
    info = {**cal, "fa_rate_ref": float(out.loc[ref_mask, "alarm"].mean())}
    return out, info


def _first_alarm(out: pd.DataFrame, lo: str, hi: str) -> pd.Timestamp | None:
    """Время первой тревоги в интервале [lo, hi] (или None)."""
    # Ищем самый ранний час с тревогой в заданном окне.
    m = (out["time"] >= pd.Timestamp(lo)) & (out["time"] <= pd.Timestamp(hi)) & out["alarm"]
    return out.loc[m, "time"].min() if m.any() else None


def _plot(out: pd.DataFrame, info: dict, path: str) -> None:
    """Двухпанельный рисунок: EWMA H(t) с пределами и CUSUM с порогом."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    # Панель 1: EWMA-индикатор и контрольные пределы.
    ax1.plot(out["time"], out["H_ewma"], color="#1f6feb", lw=0.9, label="H(t) = EWMA остатка")
    ax1.axhline(info["mu0"], color="k", lw=0.5)
    ax1.plot(out["time"], out["ewma_hi"], "r--", lw=0.8, label="±3σ предел EWMA")
    ax1.plot(out["time"], out["ewma_lo"], "r--", lw=0.8)
    # Отмечаем out-of-envelope часы (артефактная зона) и тревоги.
    oe = out.loc[~out["in_envelope"]]
    ax1.scatter(oe["time"], oe["H_ewma"], s=8, color="orange", label="вне огибающей")
    al = out.loc[out["alarm"]]
    ax1.scatter(al["time"], al["H_ewma"], s=10, color="red", zorder=5, label="тревога (in-env)")
    ax1.set_ylabel("H(t), Гц"); ax1.set_title("Слой 2: индикатор здоровья H(t) (EWMA остатка f_reg)")
    ax1.legend(loc="upper left", fontsize=8)

    # Панель 2: CUSUM (+/−) и порог решения.
    ax2.plot(out["time"], out["cusum_pos"], color="#2a9d8f", lw=0.8, label="CUSUM+")
    ax2.plot(out["time"], out["cusum_neg"], color="#8a2be2", lw=0.8, label="CUSUM−")
    ax2.axhline(info["cusum_h"], color="r", ls="--", lw=0.8, label="порог h=5σ")
    ax2.set_ylabel("CUSUM, Гц"); ax2.set_xlabel("время"); ax2.legend(loc="upper left", fontsize=8)

    # Границы val/test на обеих панелях.
    for ax in (ax1, ax2):
        for b in ("val", "test"):
            ax.axvline(pd.Timestamp(SPLIT_BOUNDS[b][0]), color="gray", ls=":", lw=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log.info("Рисунок сохранён: %s", path)


def main() -> None:
    """CLI: построить индикатор H(t), сохранить ряд/рисунок, напечатать сводку."""
    # Загрузка, огибающая, индикаторы и тревоги.
    df = add_envelope(load_inputs())
    out, info = build_health(df)

    # Сохраняем ряд индикатора для отчёта/слоя 3.
    cols = ["time", "seg", "transient", "in_envelope", "resid_gbm",
            "H_ewma", "ewma_lo", "ewma_hi", "cusum_pos", "cusum_neg", "cusum_h", "alarm"]
    out[cols].to_parquet(HEALTH_PARQUET, index=False)
    _plot(out, info, HEALTH_PNG)

    # Сводка: пороги, ложные тревоги на опоре, первые тревоги по периодам.
    print("\n============== СЛОЙ 2 — индикатор H(t) ==============")
    print(f"Опора (OOS train, in-env): μ0={info['mu0']:+.5f}  σ={info['sigma']:.5f} Гц")
    print(f"Пределы EWMA: ±{info['ewma_limit']:.5f} Гц | порог CUSUM h={info['cusum_h']:.5f} Гц")
    print(f"Ложные тревоги на опорном окне (train,in-env): {100*info['fa_rate_ref']:.2f}%")
    for name, (lo, hi) in SPLIT_BOUNDS.items():
        m = (out["time"] >= pd.Timestamp(lo)) & (out["time"] <= pd.Timestamp(hi))
        fa = _first_alarm(out, lo, hi)
        print(f"  {name}: тревог {int(out.loc[m,'alarm'].sum()):4d}/{int(m.sum()):4d} "
              f"({100*out.loc[m,'alarm'].mean():5.1f}%)  первая: {fa}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
