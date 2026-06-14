"""Таблица интерпретации синтетических сценариев (для отчёта).

Числа берутся ТОЛЬКО из существующих артефактов синтетики
(``synthetic_drift_delays.csv`` — задержки/обнаружения при оконном FA=1%),
не из памяти. Добавляется физический смысл сценария и качественная
обнаружимость. Детектор — CUSUM (основное правило, см. этап 7).

Запуск:  ``python -m src.scenarios_table``
Артефакт: ``outputs/synthetic_scenarios_table.csv``.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
DELAYS_CSV = os.path.join(OUTPUTS_DIR, "synthetic_drift_delays.csv")
OUT_CSV = os.path.join(OUTPUTS_DIR, "synthetic_scenarios_table.csv")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("scenarios_table")

# Физический смысл по строкам (как в задании, §3.5).
MEANING = {
    ("step", -0.005): "очень слабый сдвиг, у границы обнаружимости",
    ("step", -0.010): "ранняя деградация",
    ("step", -0.015): "заметный, но не аварийный сдвиг",
    ("creep", -0.0005): "сверхслабый дрейф, ниже уверенной обнаружимости",
    ("creep", -0.001): "очень медленный, на грани",
    ("creep", -0.002): "медленная деградация (минимально обнаружимая)",
    ("creep", -0.005): "значимый негативный тренд",
}


def detectability(power: float) -> str:
    """Качественная обнаружимость по power при FA=1%."""
    # Грубые градации — для читаемости таблицы в ПЗ.
    if power >= 0.8:
        return "высокая"
    if power >= 0.5:
        return "средняя"
    if power >= 0.2:
        return "низкая (у границы)"
    return "практически не ловится"


def build() -> pd.DataFrame:
    """Свернуть delays.csv (CUSUM) в таблицу сценариев с физическим смыслом."""
    # Берём CUSUM и считаем power и медианную задержку по сценариям.
    d = pd.read_csv(DELAYS_CSV)
    d = d[d["detector"] == "cusum"]
    g = (d.groupby(["kind", "magnitude"])
         .agg(power=("detected", "mean"),
              median_delay_h=("delay_h", lambda x: np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else np.nan))
         .reset_index())

    # Добавляем физический смысл, единицы и обнаружимость; упорядочиваем.
    rows = []
    for _, r in g.iterrows():
        key = (r["kind"], round(r["magnitude"], 4))
        unit = "Гц/нед" if r["kind"] == "creep" else "Гц"
        delay_d = r["median_delay_h"] / 24 if np.isfinite(r["median_delay_h"]) else np.nan
        rows.append({
            "сценарий": r["kind"],
            "амплитуда": f"{r['magnitude']:+.4f} {unit}",
            "физический_смысл": MEANING.get(key, "—"),
            "power_FA1": round(r["power"], 2),
            "медианная_задержка_ч": round(r["median_delay_h"], 0) if np.isfinite(r["median_delay_h"]) else np.nan,
            "медианная_задержка_дн": round(delay_d, 1) if np.isfinite(delay_d) else np.nan,
            "обнаружимость": detectability(r["power"]),
        })
    out = pd.DataFrame(rows)
    # Сортировка: ступени, затем creep; по возрастанию |амплитуды|.
    out["_k"] = out["сценарий"].map({"step": 0, "creep": 1})
    return out.sort_values(["_k", "амплитуда"]).drop(columns="_k").reset_index(drop=True)


def main() -> None:
    """CLI: построить и сохранить таблицу сценариев, напечатать."""
    out = build()
    out.to_csv(OUT_CSV, index=False)
    log.info("Таблица сценариев сохранена: %s", OUT_CSV)
    print("\n========== Таблица интерпретации синтетических сценариев (CUSUM, FA=1%) ==========")
    print(out.to_string(index=False))
    print("Источник чисел: outputs/synthetic_drift_delays.csv (не из памяти).")
    print("Оговорка: валидация на синтетике + 1 спорный реальный эпизод; дрейф ≠ доказанная деградация.")
    print("=================================================================================\n")


if __name__ == "__main__":
    main()
