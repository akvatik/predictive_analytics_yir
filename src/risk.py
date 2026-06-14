"""Предиктивный слой: time-to-threshold (TTT) и уровни риска.

Переход от «аномалия уже есть» к «при текущей динамике порог риска будет
достигнут через ~N дней». Работает поверх УЖЕ посчитанного индикатора H(t)
(EWMA из ``health.py``) и УЖЕ откалиброванных порогов (эмпирический предел
EWMA и порог CUSUM из ``health_series.parquet``) — новых порогов не вводим.

ВАЖНО (честность):
* TTT выдаётся ТОЛЬКО при статистически значимом ОТРИЦАТЕЛЬНОМ тренде H
  (верхняя 95% односторонняя граница наклона < 0); иначе — «нет значимого тренда».
* Деградация по физике = снижение F1 = отрицательный остаток → ориентир —
  НИЖНИЙ контрольный предел EWMA (``ewma_lo``).
* TTT демонстрируется в основном на синтетике; реальный эпизод один и спорный.
  Срабатывание ⇒ «стоит присмотреться / запланировать осмотр», не «вал сломан».

Уровни риска (привязаны к уже посчитанным величинам):
* **Green**  — H в норме И значимого отрицательного тренда нет.
* **Yellow** — есть значимый отрицательный тренд (14/28 дн), но H ещё не пробил
  порог тревоги (наблюдать).
* **Orange** — устойчивый дрейф подтверждён: CUSUM защёлкнут (> эмпирич. порога)
  И час in-envelope (рекомендуется внеплановый контроль/осмотр).
* **Red**    — сильный устойчивый дрейф; ВНЕ области применимости проекта
  (нет полного набора физических признаков для подтверждения аварии). Метка
  только ДОКУМЕНТИРУЕТ границу, модель не утверждает Red как состояние.

Запуск:  ``python -m src.risk``
Артефакты: ``outputs/time_to_threshold.csv``, ``risk_timeline.csv``,
``time_to_threshold.png``.
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

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
HEALTH_PARQUET = os.path.join(OUTPUTS_DIR, "health_series.parquet")
TTT_CSV = os.path.join(OUTPUTS_DIR, "time_to_threshold.csv")
RISK_CSV = os.path.join(OUTPUTS_DIR, "risk_timeline.csv")
TTT_PNG = os.path.join(OUTPUTS_DIR, "time_to_threshold.png")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("risk")

# --- Параметры (пороги — из health_series, не вводим новых) ----------------
TREND_WINDOWS_D = {"14d": 14, "28d": 28}   # трейлинг-окна тренда H
SIG_Z = 1.65          # односторонняя 95% граница значимости наклона
CI_Z = 1.96           # 95% диапазон для границ TTT
MIN_PTS = 48          # минимум точек в окне для регрессии
RED_FACTOR = 3.0      # граница Red: CUSUM > RED_FACTOR×порога (только документируем)


def trailing_ols(t_hours: np.ndarray, y: np.ndarray, window_h: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Наклон H по времени и его SE на трейлинг-окне (OLS), для каждой точки."""
    n = len(t_hours)
    slope = np.full(n, np.nan); se = np.full(n, np.nan)
    # Для каждой точки берём предшествующее окно [t-window, t] и считаем OLS.
    for i in range(n):
        lo = np.searchsorted(t_hours, t_hours[i] - window_h, side="left")
        xs, ys = t_hours[lo:i + 1], y[lo:i + 1]
        if len(xs) < MIN_PTS:
            continue
        # Центрируем x; наклон b=Sxy/Sxx, SE_b=sqrt(SSE/(n-2)/Sxx).
        xc = xs - xs.mean()
        sxx = np.dot(xc, xc)
        if sxx <= 0:
            continue
        b = np.dot(xc, ys - ys.mean()) / sxx
        resid = ys - (ys.mean() + b * xc)
        s2 = np.dot(resid, resid) / max(len(xs) - 2, 1)
        slope[i] = b
        se[i] = np.sqrt(s2 / sxx)
    return slope, se


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Посчитать тренды H, TTT до нижнего порога и уровни риска по времени."""
    df = df.sort_values("time").reset_index(drop=True).copy()
    # Время в часах от начала и пороги из health_series (уже откалиброваны).
    t_h = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    thr_lo = float(df["ewma_lo"].iloc[0])      # нижний предел EWMA = ориентир TTT
    cusum_h = float(df["cusum_h"].iloc[0])

    # Тренды H на трейлинг-окнах 14 и 28 дней (наклон + SE).
    for name, days in TREND_WINDOWS_D.items():
        b, se = trailing_ols(t_h, df["H_ewma"].to_numpy(), days * 24)
        df[f"slope_{name}"] = b              # Гц/час
        df[f"se_{name}"] = se
        # Значимо отрицательный = верхняя односторонняя 95% граница < 0.
        df[f"signeg_{name}"] = (b + SIG_Z * se < 0)

    # TTT по 14-дневному наклону: только при значимом отриц. тренде и H выше порога.
    b14, se14 = df["slope_14d"].to_numpy(), df["se_14d"].to_numpy()
    gate = df["signeg_14d"].to_numpy() & (df["H_ewma"].to_numpy() > thr_lo)
    dist = df["H_ewma"].to_numpy() - thr_lo            # запас до нижнего порога (>0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Точка и диапазон из наклона b±1.96·SE (часы → дни).
        ttt = np.where(gate, dist / np.abs(b14) / 24.0, np.nan)
        b_lo = b14 - CI_Z * se14   # круче (быстрее) → нижняя граница TTT
        b_hi = b14 + CI_Z * se14   # положе; если ≥0 — верхней границы нет
        ttt_low = np.where(gate, dist / np.abs(b_lo) / 24.0, np.nan)
        ttt_high = np.where(gate & (b_hi < 0), dist / np.abs(b_hi) / 24.0, np.nan)
    df["TTT_dni"], df["TTT_low"], df["TTT_high"] = ttt, ttt_low, ttt_high

    # CUSUM защёлкнут (любая сторона) и сила дрейфа — для уровней риска.
    cmax = np.maximum(df["cusum_pos"], df["cusum_neg"]).to_numpy()
    latched = cmax > cusum_h
    in_env = df["in_envelope"].to_numpy().astype(bool)
    sig_neg = (df["signeg_14d"] | df["signeg_28d"]).to_numpy()
    in_alarm = (df["H_ewma"].to_numpy() < thr_lo) | (df["H_ewma"].to_numpy() > df["ewma_hi"].iloc[0])

    # Приоритет уровней: Red(граница) > Orange > Yellow > Green.
    level = np.full(len(df), "Green", dtype=object)
    level[sig_neg & ~in_alarm] = "Yellow"                 # значимый тренд, ещё без тревоги
    level[latched & in_env] = "Orange"                    # дрейф подтверждён
    level[(cmax > RED_FACTOR * cusum_h) & in_env] = "Red"  # граница вне области применимости
    df["risk_level"] = level
    df.attrs["thr_lo"] = thr_lo
    df.attrs["cusum_h"] = cusum_h
    return df


def _plot(df: pd.DataFrame, path: str) -> None:
    """Рисунок: H(t) с нижним порогом и раскраской риска + TTT, где определён."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    colors = {"Green": "#2a9d8f", "Yellow": "#e9c46a", "Orange": "#f4a261", "Red": "#e63946"}
    # Панель 1: H(t), нижний порог EWMA, точки риска по уровням.
    ax1.plot(df["time"], df["H_ewma"], color="#444", lw=0.6, label="H(t)")
    ax1.axhline(df.attrs["thr_lo"], color="r", ls="--", lw=0.8, label="нижний порог EWMA")
    for lvl, c in colors.items():
        m = df["risk_level"] == lvl
        if m.any():
            ax1.scatter(df.loc[m, "time"], df.loc[m, "H_ewma"], s=5, color=c, label=lvl)
    for b in ("val", "test"):
        ax1.axvline(pd.Timestamp(SPLIT_BOUNDS[b][0]), color="gray", ls=":", lw=0.9)
    ax1.set_ylabel("H(t), Гц"); ax1.legend(fontsize=7, ncol=3)
    ax1.set_title("Уровни риска по H(t) (Green/Yellow/Orange/Red — Red лишь документирует границу)")

    # Панель 2: TTT (дни) с диапазоном, где тренд значимо отрицателен.
    ax2.plot(df["time"], df["TTT_dni"], color="#1f6feb", lw=0.8, label="TTT, дни")
    ax2.fill_between(df["time"], df["TTT_low"], df["TTT_high"], color="#1f6feb", alpha=0.2,
                     label="диапазон 95%")
    ax2.set_ylabel("TTT, дни"); ax2.set_xlabel("время"); ax2.legend(fontsize=8)
    ax2.set_title("Time-to-threshold (только при значимом отрицательном тренде)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log.info("Рисунок сохранён: %s", path)


def main() -> None:
    """CLI: посчитать TTT и уровни риска, сохранить артефакты, печать сводки."""
    df = pd.read_parquet(HEALTH_PARQUET)
    out = compute(df)

    # Сохраняем TTT-ряд и риск-таймлайн (новые файлы, старое не трогаем).
    ttt_cols = ["time", "H_ewma", "slope_14d", "slope_28d",
                "TTT_dni", "TTT_low", "TTT_high", "risk_level"]
    out[ttt_cols].to_csv(TTT_CSV, index=False)
    out[["time", "H_ewma", "signeg_14d", "signeg_28d", "in_envelope",
         "cusum_pos", "cusum_neg", "risk_level"]].to_csv(RISK_CSV, index=False)
    _plot(out, TTT_PNG)

    # --- Сводка ------------------------------------------------------------
    counts = out["risk_level"].value_counts()
    print("\n========== TTT + уровни риска ==========")
    print(f"Нижний порог EWMA (из health): {out.attrs['thr_lo']:+.5f} Гц | "
          f"CUSUM-порог: {out.attrs['cusum_h']:.4f}")
    print("Распределение уровней риска (часы):")
    for lvl in ("Green", "Yellow", "Orange", "Red"):
        print(f"  {lvl:7s}: {int(counts.get(lvl, 0)):5d} "
              f"({100*counts.get(lvl,0)/len(out):.1f}%)")
    ttt_valid = out["TTT_dni"].notna()
    print(f"\nЧасов со значимым отриц. трендом (есть TTT): {int(ttt_valid.sum())}")
    if ttt_valid.any():
        v = out.loc[ttt_valid, "TTT_dni"]
        print(f"  TTT, дни: медиана={v.median():.1f}  диапазон=[{v.min():.1f}, {v.max():.1f}]")
    print("Примечание: TTT/риск демонстрируются гл. обр. на синтетике; реальный")
    print("эпизод один и спорный; дрейф ≠ доказанная деградация.")
    print("=======================================\n")


if __name__ == "__main__":
    main()
