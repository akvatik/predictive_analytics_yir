"""Sensitivity по шуму σ: как качество остатка влияет на задержку обнаружения.

What-if поверх СУЩЕСТВУЮЩЕЙ синтетической обвязки (``synthetic_drift.py``):
масштабируем шум здорового остатка к целевому σ (центрированный остаток ×
σ_target/σ_current), амплитуды впрыскиваемого дрейфа ДЕРЖИМ фиксированными и
пересчитываем задержку/power. Показывает: чище остаток (меньше σ) ⇒ раньше
и надёжнее обнаружение того же дрейфа.

Заземление: σ≈0.005 ≈ уровень промышленного Prism; σ=0.003 — гипотетически
лучше Prism. Это **what-if** — фактический σ проекта не меняется.

Пороги масштабируются вместе с σ (всё линейно), поэтому оконный FA=1%
сохраняется, а меняется только отношение сигнал/шум.

Запуск:  ``python -m src.sensitivity_noise``
Артефакты: ``outputs/sensitivity_noise.csv``, ``outputs/sensitivity_noise.png``.
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

from src.synthetic_drift import (CREEP_RATES_WK, OUTPUTS_DIR, STEP_AMPS,
                                 add_envelope, calibrate, calibrate_window_threshold,
                                 detect_delay, drift_signal, load_inputs, make_pool,
                                 reference_sigma)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("sensitivity_noise")

SIGMA_TARGETS = [0.008, 0.005, 0.003]   # + фактический добавим первым
SENS_CSV = os.path.join(OUTPUTS_DIR, "sensitivity_noise.csv")
SENS_PNG = os.path.join(OUTPUTS_DIR, "sensitivity_noise.png")


def _scaled_pool(pool: list[dict], s: float) -> list[dict]:
    """Масштабировать шум несущих окон к целевому σ (дрейф добавим отдельно)."""
    # Центрированная несущая × s меняет σ, сохраняя форму автокорреляции.
    return [{"start": w["start"], "elapsed": w["elapsed"], "carrier": w["carrier"] * s}
            for w in pool]


def _scaled_cal(cal: dict, s: float) -> dict:
    """Масштабировать опорные μ0/σ и пороги вместе с σ (FA=1% инвариантен)."""
    # Все величины линейны по σ → умножаем на тот же коэффициент s.
    return {"mu0": cal["mu0"] * s, "sigma": cal["sigma"] * s, "k": cal["k"] * s,
            "ewma_limit": cal["ewma_limit"] * s, "cusum_h": cal["cusum_h"] * s}


def run(pool: list[dict], cal: dict, thr: dict, sigma_now: float) -> pd.DataFrame:
    """Power и медианная задержка по σ × сценариям × детекторам (дрейф фиксирован)."""
    rows = []
    scenarios = [("creep", r) for r in CREEP_RATES_WK] + [("step", a) for a in STEP_AMPS]
    sigmas = [sigma_now] + SIGMA_TARGETS
    # Для каждого целевого σ масштабируем шум и пороги, дрейф оставляем как есть.
    for sigma_t in sigmas:
        s = sigma_t / sigma_now
        sp, sc = _scaled_pool(pool, s), _scaled_cal(cal, s)
        for kind, mag in scenarios:
            for det in ("cusum", "ewma"):
                # detect_delay при том же FA=1% факторе порога (FA инвариантен к s).
                delays = [detect_delay(w["carrier"] + drift_signal(w["elapsed"], kind, mag),
                                       w["elapsed"], sc, det, thr[det]) for w in sp]
                delays = np.array(delays, dtype=float)
                det_mask = np.isfinite(delays)
                rows.append({"sigma": round(sigma_t, 4), "detector": det, "kind": kind,
                             "magnitude": mag, "power": float(det_mask.mean()),
                             "median_delay_h": float(np.median(delays[det_mask])) if det_mask.any() else np.nan})
    return pd.DataFrame(rows)


def min_detectable(sens: pd.DataFrame, detector="cusum", power_min=0.7) -> pd.DataFrame:
    """Минимальный обнаружимый |дрейф| (power≥порога) по σ для creep и ступени."""
    rows = []
    # Для каждого σ и типа сценария берём наименьшую |магнитуду| с достаточным power.
    for sigma in sorted(sens["sigma"].unique()):
        for kind in ("creep", "step"):
            d = sens[(sens.sigma == sigma) & (sens.detector == detector) & (sens.kind == kind)]
            ok = d[d["power"] >= power_min]
            mind = ok["magnitude"].abs().min() if len(ok) else np.nan
            rows.append({"sigma": sigma, "kind": kind, "min_detectable_abs": mind})
    return pd.DataFrame(rows)


def _plot(sens: pd.DataFrame, mind: pd.DataFrame, path: str) -> None:
    """Рисунок: (A) задержка vs σ (репрезентативные сценарии); (B) мин. дрейф vs σ."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4))
    # Панель A: медианная задержка CUSUM от σ для пары наглядных сценариев.
    for (kind, mag), color in [(("step", -0.01), "#e76f51"), (("creep", -0.002), "#2a9d8f")]:
        d = sens[(sens.detector == "cusum") & (sens.kind == kind) & (sens.magnitude == mag)].sort_values("sigma")
        axA.plot(d["sigma"], d["median_delay_h"], "-o", color=color, label=f"{kind} {mag:+.3f}")
    axA.set_xlabel("σ остатка, Гц"); axA.set_ylabel("медианная задержка, ч (CUSUM)")
    axA.set_title("Задержка падает с уменьшением σ"); axA.legend(fontsize=8); axA.grid(alpha=0.3)

    # Панель B: минимальный обнаружимый дрейф (power≥0.7) от σ.
    for kind, color in [("creep", "#2a9d8f"), ("step", "#1f6feb")]:
        d = mind[mind.kind == kind].sort_values("sigma")
        axB.plot(d["sigma"], d["min_detectable_abs"], "-o", color=color, label=kind)
    axB.set_xlabel("σ остатка, Гц"); axB.set_ylabel("|мин. обнаружимый дрейф|")
    axB.set_title("Чувствительность растёт с уменьшением σ"); axB.legend(fontsize=8); axB.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log.info("Рисунок сохранён: %s", path)


def main() -> None:
    """CLI: what-if по σ, сохранить таблицу/рисунок, печать сводки."""
    # Данные/огибающая/калибровка и пул несущих (как в синтетике), пороги при FA=1%.
    df = add_envelope(load_inputs())
    mu0, sigma = reference_sigma(df)
    cal = calibrate(df, mu0, sigma)
    pool = make_pool(df)
    thr = {det: calibrate_window_threshold(pool, cal, det) for det in ("cusum", "ewma")}

    sens = run(pool, cal, thr, sigma)
    sens.to_csv(SENS_CSV, index=False)
    mind = min_detectable(sens)
    _plot(sens, mind, SENS_PNG)

    # --- Сводка ------------------------------------------------------------
    print("\n========== Sensitivity по шуму σ (what-if) ==========")
    print(f"Фактический σ ≈ {sigma:.4f} Гц | целевые: {SIGMA_TARGETS} "
          f"(σ≈0.005 ≈ Prism, σ=0.003 — гипотетически лучше)")
    print("\nCUSUM, медианная задержка (ч) по σ:")
    piv = (sens[sens.detector == "cusum"]
           .assign(scn=lambda d: d.kind + " " + d.magnitude.map(lambda m: f"{m:+.4f}"))
           .pivot_table(index="scn", columns="sigma", values="median_delay_h"))
    print(piv.round(0).to_string())
    print("\nМинимальный обнаружимый |дрейф| (power≥0.7, CUSUM) по σ:")
    print(mind.pivot(index="kind", columns="sigma", values="min_detectable_abs").round(4).to_string())
    print("Оговорка: what-if; фактический σ проекта не меняется.")
    print("=====================================================\n")


if __name__ == "__main__":
    main()
