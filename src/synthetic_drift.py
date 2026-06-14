"""Этап 7 — валидация детектора впрыском синтетического дрейфа.

Размеченных отказов нет → детектор слоя 2 валидируем контролируемым дрейфом,
впрыснутым в ЗДОРОВЫЕ окна (Вариант C, раздел 7 брифа). Это основной источник
чисел о детекторе: наблюдаемый эпизод авг–сен 2016 даёт лишь ОДНУ OOS-точку
детекции и конфаундится сезоном (лето 2015 тоже даёт +0.0033 Гц).

Методика:
* **Пул несущих окон** — скользящие 8-недельные окна по здоровым in-envelope
  данным train, БЕЗ лета (июль–сентябрь): достаточно окон для оценки FA и
  РАСПРЕДЕЛЕНИЯ задержки. Несущая — честный OOS-остаток (block-CV),
  центрированный к нулю (изолируем отклик на впрыснутый дрейф от сезонного смещения).
* **Сценарии** (F1 падает = потеря жёсткости):
    - линейный creep: −0.0005/−0.001/−0.002/−0.005 Гц/неделя (имитация
      медленного деградационного процесса);
    - ступень: −0.005/−0.01/−0.015 Гц (имитация резкого изменения состояния).
* **Пайплайн не обходим:** дрейф добавляется к F1 → остаток = (F1+дрейф) − f_reg
  = baseline-остаток + дрейф (f_reg зависит от режима x, не от F1).
* **Устойчивый алярм:** срабатывание = ``N_CONSEC`` часов подряд выше порога
  (убирает ложное срабатывание на одиночном шуме, особенно у EWMA).
* **Порог под оконный FA=1%** подбирается из ROC (а не глобальный непрерывный
  порог health.py, который консервативнее на оконной шкале).

Метрики: задержка vs скорость/амплитуда; ROC (FA vs power); минимальный
обнаружимый дрейф при FA≤1%; распределение задержек; EWMA vs CUSUM.

Запуск:  ``python -m src.synthetic_drift``
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

from src.health import (CUSUM_K_SIGMA, OUTPUTS_DIR, add_envelope, calibrate,
                        load_inputs, reference_sigma, _cusum, _ewma)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RANDOM_STATE = 42
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("synthetic_drift")

# --- Конфигурация эксперимента --------------------------------------------
WINDOW_WEEKS = 8
WINDOW_HOURS = WINDOW_WEEKS * 7 * 24
STEP_HOURS = 72                # шаг скользящего окна (для пула несущих)
COVERAGE_MIN = 0.7            # мин. покрытие окна валидными часами
N_CONSEC = 6                  # устойчивый алярм: часов подряд выше порога
SUMMER_MONTHS = {7, 8, 9}    # лето исключаем (сезонное перекрытие)
TRAIN_END = "2016-06-30 23:59:59"

CREEP_RATES_WK = [-0.0005, -0.001, -0.002, -0.005]   # Гц/неделя
STEP_AMPS = [-0.005, -0.01, -0.015]                  # Гц
TARGET_FA = 0.01
ROC_FACTORS = np.linspace(0.05, 2.5, 30)             # множители порога для ROC

DELAYS_CSV = os.path.join(OUTPUTS_DIR, "synthetic_drift_delays.csv")
ROC_CSV = os.path.join(OUTPUTS_DIR, "synthetic_drift_roc.csv")
DELAY_PNG = os.path.join(OUTPUTS_DIR, "synthetic_delay_vs_drift.png")
ROC_PNG = os.path.join(OUTPUTS_DIR, "synthetic_roc.png")


def make_pool(df: pd.DataFrame) -> list[dict]:
    """Пул здоровых несущих: скользящие несезонные in-envelope окна, центр. к 0."""
    # Здоровые часы: train, in-envelope, валидный OOS-остаток, не лето.
    base = df[(df["time"] <= pd.Timestamp(TRAIN_END)) & df["in_envelope"]
              & df["resid_oos_train"].notna()
              & ~df["time"].dt.month.isin(SUMMER_MONTHS)].sort_values("time")
    starts = pd.date_range(base["time"].min(), base["time"].max() - pd.Timedelta(hours=WINDOW_HOURS),
                           freq=f"{STEP_HOURS}h")
    pool = []
    # Для каждого старта собираем окно; отбрасываем дырявые и летние/смешанные.
    for t0 in starts:
        w = base[(base["time"] >= t0) & (base["time"] < t0 + pd.Timedelta(hours=WINDOW_HOURS))]
        if len(w) < WINDOW_HOURS * COVERAGE_MIN or w["time"].dt.month.isin(SUMMER_MONTHS).any():
            continue
        carrier = (w["resid_oos_train"] - w["resid_oos_train"].mean()).to_numpy()
        elapsed = (w["time"] - w["time"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        pool.append({"start": t0, "carrier": carrier, "elapsed": elapsed})
    log.info("Пул несущих окон: %d (по %d нед, шаг %dч, без лета)", len(pool), WINDOW_WEEKS, STEP_HOURS)
    return pool


def drift_signal(elapsed_h: np.ndarray, kind: str, magnitude: float) -> np.ndarray:
    """Сигнал дрейфа d(t): creep (линейный, Гц/нед) или step (ступень с t0)."""
    if kind == "creep":
        return magnitude * (elapsed_h / (7 * 24.0))
    if kind == "step":
        return np.full_like(elapsed_h, magnitude, dtype=float)
    raise ValueError(kind)


def _first_sustained(over: np.ndarray) -> int | None:
    """Индекс первого устойчивого алярма: N_CONSEC часов подряд выше порога."""
    # Считаем длину текущей серии превышений; срабатывание при достижении N_CONSEC.
    run = 0
    for i, v in enumerate(over):
        run = run + 1 if v else 0
        if run >= N_CONSEC:
            return i
    return None


def _alarm_stat(resid: np.ndarray, cal: dict, detector: str) -> np.ndarray:
    """Булев ряд превышения сырого порога (до требования устойчивости)."""
    s = pd.Series(resid)
    if detector == "cusum":
        # Дрейф отрицательный → растёт CUSUM− (накопление отрицательных сдвигов).
        _, cneg = _cusum(s, cal["mu0"], cal["k"])
        return cneg, cal["cusum_h"]
    if detector == "ewma":
        h = _ewma(s).to_numpy()
        return (cal["mu0"] - h), cal["ewma_limit"]   # насколько H ниже нижнего предела
    raise ValueError(detector)


def detect_delay(resid: np.ndarray, elapsed_h: np.ndarray, cal: dict,
                 detector: str, thr_factor: float) -> float:
    """Задержка (ч) до первого устойчивого алярма; NaN если не обнаружен."""
    stat, base_thr = _alarm_stat(resid, cal, detector)
    over = stat > base_thr * thr_factor
    idx = _first_sustained(over)
    return float(elapsed_h[idx]) if idx is not None else np.nan


def fa_rate(pool: list[dict], cal: dict, detector: str, thr_factor: float) -> float:
    """Оконный FA: доля здоровых окон с устойчивым ложным алярмом (без дрейфа)."""
    # Прогоняем детектор по НЕдрейфовым несущим и считаем долю сработавших окон.
    hits = [np.isfinite(detect_delay(w["carrier"], w["elapsed"], cal, detector, thr_factor))
            for w in pool]
    return float(np.mean(hits))


def calibrate_window_threshold(pool: list[dict], cal: dict, detector: str) -> float:
    """Найти множитель порога, дающий оконный FA ≤ TARGET_FA (самый чувствительный)."""
    # Идём от низкого порога (чувствительный) к высокому; берём первый с FA≤цели.
    best = ROC_FACTORS[-1]
    for f in ROC_FACTORS:
        if fa_rate(pool, cal, detector, f) <= TARGET_FA:
            best = f
            break
    log.info("[%s] порог под FA≤%.0f%%: ×%.3f (FA=%.3f)",
             detector, 100 * TARGET_FA, best, fa_rate(pool, cal, detector, best))
    return best


def run_delays(pool: list[dict], cal: dict, thr: dict) -> pd.DataFrame:
    """Задержки/обнаружения по всем окнам × сценариям × детекторам при FA=1% порогах."""
    rows = []
    scenarios = [("creep", r) for r in CREEP_RATES_WK] + [("step", a) for a in STEP_AMPS]
    # В каждом окне инжектим сценарий и детектим обоими правилами при их FA=1% пороге.
    for kind, mag in scenarios:
        for w in pool:
            d = drift_signal(w["elapsed"], kind, mag)
            m = w["carrier"] + d
            for det in ("cusum", "ewma"):
                delay = detect_delay(m, w["elapsed"], cal, det, thr[det])
                rows.append({"kind": kind, "magnitude": mag, "start": str(w["start"].date()),
                             "detector": det, "delay_h": delay, "detected": bool(np.isfinite(delay))})
    return pd.DataFrame(rows)


def run_roc(pool: list[dict], cal: dict, kind: str, mag: float) -> pd.DataFrame:
    """ROC: при варьировании порога — оконный FA vs power для CUSUM и EWMA."""
    rows = []
    for det in ("cusum", "ewma"):
        for f in ROC_FACTORS:
            fa = fa_rate(pool, cal, det, f)
            power = np.mean([np.isfinite(detect_delay(w["carrier"] + drift_signal(w["elapsed"], kind, mag),
                                                      w["elapsed"], cal, det, f)) for w in pool])
            rows.append({"detector": det, "h_factor": float(f), "FA": fa, "power": float(power)})
    return pd.DataFrame(rows)


def _nanq(x: pd.Series, q: float) -> float:
    """Квантиль по обнаруженным окнам (NaN если не обнаружено ни одного)."""
    # Только конечные задержки; пустой срез → NaN без предупреждения numpy.
    v = x[np.isfinite(x)]
    return float(np.percentile(v, q)) if len(v) else np.nan


def summarize(delays: pd.DataFrame) -> pd.DataFrame:
    """Power и квантили задержки по детектору/сценарию/магнитуде (при FA=1% пороге)."""
    return (delays.groupby(["detector", "kind", "magnitude"])
            .agg(power=("detected", "mean"),
                 median_delay_h=("delay_h", lambda x: _nanq(x, 50)),
                 p90_delay_h=("delay_h", lambda x: _nanq(x, 90)))
            .reset_index())


def _plot_delay(delays: pd.DataFrame, path: str) -> None:
    """Рисунок: задержка обнаружения vs магнитуда (разброс по окнам), CUSUM vs EWMA."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, kind, xlabel in [(axes[0], "creep", "|скорость creep|, Гц/нед"),
                             (axes[1], "step", "|амплитуда ступени|, Гц")]:
        sub = delays[delays["kind"] == kind]
        for det, color in [("cusum", "#2a9d8f"), ("ewma", "#1f6feb")]:
            d = sub[sub["detector"] == det]
            ax.scatter(d["magnitude"].abs(), d["delay_h"], s=10, color=color, alpha=0.3)
            med = d.groupby("magnitude")["delay_h"].median()
            ax.plot(med.index.to_series().abs(), med.values, "-o", color=color, lw=1.3, label=det)
        ax.set_xlabel(xlabel); ax.set_ylabel("задержка обнаружения, ч"); ax.legend(fontsize=8)
        ax.set_title(f"Задержка vs {kind} (медиана + разброс окон)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log.info("Рисунок сохранён: %s", path)


def _plot_roc(roc: pd.DataFrame, path: str, scen: str) -> None:
    """Рисунок ROC: оконный FA vs power для CUSUM и EWMA."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for det, color in [("cusum", "#2a9d8f"), ("ewma", "#1f6feb")]:
        d = roc[roc["detector"] == det].sort_values("FA")
        ax.plot(d["FA"], d["power"], "-o", ms=3, color=color, label=det)
    ax.axvline(TARGET_FA, color="r", ls="--", lw=0.8, label="FA=1%")
    ax.set_xlabel("оконный FA"); ax.set_ylabel("power (доля обнаружений)")
    ax.set_title(f"ROC детектора ({scen})"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log.info("Рисунок сохранён: %s", path)


def main() -> None:
    """CLI: впрыск дрейфа, метрики детектора, рисунки и таблицы."""
    # Данные, огибающая, калибровка σ/порогов (как слой 2) и пул несущих окон.
    df = add_envelope(load_inputs())
    mu0, sigma = reference_sigma(df)
    cal = calibrate(df, mu0, sigma)
    pool = make_pool(df)

    # Пороги под оконный FA=1% для каждого детектора (честная шкала эксперимента).
    thr = {det: calibrate_window_threshold(pool, cal, det) for det in ("cusum", "ewma")}

    # Задержки/power по сценариям + ROC для представительного медленного creep.
    delays = run_delays(pool, cal, thr)
    delays.to_csv(DELAYS_CSV, index=False)
    summary = summarize(delays)
    roc = run_roc(pool, cal, "creep", -0.001)
    roc.to_csv(ROC_CSV, index=False)
    _plot_delay(delays, DELAY_PNG)
    _plot_roc(roc, ROC_PNG, "creep −0.001 Гц/нед")

    # --- Печать сводки -----------------------------------------------------
    print("\n========== ЭТАП 7 — валидация детектора синтетикой ==========")
    print(f"Пул несущих окон: {len(pool)} (по {WINDOW_WEEKS} нед, без лета) | σ={sigma:.5f}")
    print(f"Устойчивый алярм: {N_CONSEC} ч подряд | пороги под оконный FA=1%: "
          f"CUSUM ×{thr['cusum']:.3f}, EWMA ×{thr['ewma']:.3f}")
    print("\nPower (доля обнаруживших окон) и задержка (медиана / p90) при FA=1%:")
    for det in ("cusum", "ewma"):
        print(f"\n  [{det.upper()}]")
        s = summary[summary["detector"] == det]
        for kind in ("creep", "step"):
            for _, r in s[s["kind"] == kind].sort_values("magnitude", ascending=False).iterrows():
                md = f"{r['median_delay_h']:.0f}" if np.isfinite(r["median_delay_h"]) else "—"
                p9 = f"{r['p90_delay_h']:.0f}" if np.isfinite(r["p90_delay_h"]) else "—"
                unit = "Гц/нед" if kind == "creep" else "Гц"
                print(f"    {kind:5s} {r['magnitude']:+.4f} {unit}: power={r['power']:.2f}  "
                      f"задержка медз={md} ч / p90={p9} ч")
    print("=============================================================\n")


if __name__ == "__main__":
    main()
