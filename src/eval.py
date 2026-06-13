"""Метрики и сравнение моделей (МоДеРо) — общий модуль для слоёв 1 и 3.

Содержит:
* ``time_split`` — разбиение по времени строго из раздела 5 брифа (без shuffle);
* ``FEATURE_COLS`` — отбор предикторов из таблицы признаков;
* ``regression_metrics`` — RMSE / MAE / R²;
* ``evaluate_predictions`` — метрики по сплитам и в разбивке steady/transient;
* эталоны: **persistence** (везде) и **Prism** (на подмножестве N≈1219 ч, где
  модель посчитана). Колонка «ДП1» — масштабированная копия Prism, как
  самостоятельный baseline НЕ используется (см. README, раздел data hygiene);
* сохранение таблицы метрик и базовых рисунков в ``outputs/``.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Консоль Windows (cp1251) → UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("eval")

# --- Разбиение по времени (раздел 5 брифа, без перемешивания) -------------
# Границы фиксированы по датам: обучение / валидация / тест.
SPLIT_BOUNDS = {
    "train": ("2015-03-01", "2016-06-30 23:59:59"),
    "val":   ("2016-07-01", "2016-10-31 23:59:59"),
    "test":  ("2016-11-01", "2017-01-31 23:59:59"),
}
TARGET = "F1"
# Колонки, НЕ являющиеся предикторами (цель, идентификаторы, метки, эталоны).
NON_FEATURES = {"time", "F1", "seg", "transient", "resid_prism", "dPa"}


def time_split(df: pd.DataFrame) -> dict[str, pd.Index]:
    """Вернуть индексы строк для train/val/test по датам раздела 5 (без shuffle)."""
    # Для каждого сплита отбираем строки, попавшие в его временной интервал.
    out = {}
    for name, (lo, hi) in SPLIT_BOUNDS.items():
        mask = (df["time"] >= pd.Timestamp(lo)) & (df["time"] <= pd.Timestamp(hi))
        out[name] = df.index[mask]
    # Логируем объёмы — контроль, что границы покрыли весь ряд без пересечений.
    sizes = {k: len(v) for k, v in out.items()}
    log.info("Сплит по времени (раздел 5): %s; вне сплитов: %d",
             sizes, len(df) - sum(sizes.values()))
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Полный список предикторов (всё, кроме цели/идентификаторов/меток).

    Для слоя 3 (прогноз F1(t+h)) допускает признаки от прошлой F1.
    """
    # Берём все числовые колонки, исключая служебные.
    return [c for c in df.columns if c not in NON_FEATURES]


def regime_feature_columns(df: pd.DataFrame) -> list[str]:
    """Только РЕЖИМНЫЕ предикторы для f_reg (слой 1): без признаков от F1.

    Бриф (раздел 3): f_reg(x(t)) — F1 как функция нагрузки/температур, а
    остаток r=F1−f_reg — индикатор здоровья. Лаги/скользящие F1 в f_reg
    впитали бы медленный дрейф F1 и «спрятали» его из остатка — поэтому
    исключаем все признаки, производные от F1 (``F1_*``).
    """
    # Из полного набора убираем всё, что вычислено из самой F1.
    return [c for c in feature_columns(df) if not c.startswith("F1_")]


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """RMSE / MAE / R² по совпадающим непустым парам (y_true, y_pred)."""
    # Считаем только там, где обе величины определены (NaN-прогнозы исключаем).
    m = y_true.notna() & y_pred.notna()
    if m.sum() == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
    yt, yp = y_true[m], y_pred[m]
    # RMSE через MSE (совместимо со старыми версиями sklearn), MAE и R².
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {"n": int(m.sum()), "RMSE": rmse,
            "MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp))}


def _subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    """Булева маска для подмножества: all / steady / transient."""
    # steady = НЕ переходный час; transient = переходный; all = всё.
    if subset == "all":
        return pd.Series(True, index=df.index)
    if subset == "steady":
        return df["transient"] == False  # noqa: E712  (явное сравнение с булевым)
    if subset == "transient":
        return df["transient"] == True   # noqa: E712
    raise ValueError(subset)


def evaluate_predictions(df: pd.DataFrame, pred_col: str, splits: dict[str, pd.Index],
                         model_name: str) -> pd.DataFrame:
    """Метрики модели по каждому сплиту × {all, steady, transient}."""
    rows = []
    # Перебираем сплиты и режимные подмножества, считаем метрики на пересечении.
    for split_name, idx in splits.items():
        sub = df.loc[idx]
        for subset in ("all", "steady", "transient"):
            mask = _subset_mask(sub, subset)
            mt = regression_metrics(sub.loc[mask, TARGET], sub.loc[mask, pred_col])
            rows.append({"model": model_name, "split": split_name,
                         "subset": subset, **mt})
    return pd.DataFrame(rows)


def persistence_predictions(df: pd.DataFrame) -> pd.Series:
    """Эталон persistence для слоя 1: F1̂(t) = F1(t-1) (= признак F1_lag1)."""
    # 1-шаговый persistence — сильный наивный ориентир (внутри сегмента).
    return df["F1_lag1"]


def prism_reference(df: pd.DataFrame, splits: dict[str, pd.Index]) -> pd.DataFrame:
    """Эталон Prism: RMSE остатка Prism на подмножестве N, где он посчитан."""
    rows = []
    # Остаток Prism = (F1 − прогноз Prism); его RMSE = ошибка модели Prism.
    for split_name, idx in splits.items():
        sub = df.loc[idx]
        for subset in ("all", "steady", "transient"):
            mask = _subset_mask(sub, subset) & sub["resid_prism"].notna()
            r = sub.loc[mask, "resid_prism"]
            # n здесь — число часов, где Prism реально посчитал прогноз.
            rmse = float(np.sqrt((r ** 2).mean())) if len(r) else np.nan
            mae = float(r.abs().mean()) if len(r) else np.nan
            rows.append({"model": "Prism (N посчит.)", "split": split_name,
                         "subset": subset, "n": int(len(r)),
                         "RMSE": rmse, "MAE": mae, "R2": np.nan})
    return pd.DataFrame(rows)


def save_metrics_table(table: pd.DataFrame, name: str) -> str:
    """Сохранить таблицу метрик в outputs/ (CSV) и вернуть путь."""
    # Округляем для читаемости в ПЗ; RMSE/MAE в Гц нужны с запасом знаков.
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, name)
    table.round({"RMSE": 5, "MAE": 5, "R2": 4}).to_csv(path, index=False)
    log.info("Таблица метрик сохранена: %s", path)
    return path
