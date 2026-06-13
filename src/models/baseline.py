"""Слой 1 — baseline: линейная регрессия режимной модели F1 ≈ f_reg(x).

Простая объяснимая модель-ориентир для слоя 1. Pipeline:
импутация пропусков (медиана, обученная на train) → стандартизация →
``LinearRegression``. Обучается ТОЛЬКО на train-сплите (раздел 5), затем
даёт прогноз на всём ряду (для метрик на val/test и расчёта остатка).

Запуск:  ``python -m src.models.baseline`` — печатает метрики по сплитам.
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Общие утилиты метрик/сплита (запуск: python -m src.models.baseline).
from src.eval import (OUTPUTS_DIR, TARGET, evaluate_predictions,
                      persistence_predictions, regime_feature_columns, time_split)

# Консоль Windows (cp1251) → UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RANDOM_STATE = 42  # LinearRegression детерминирован; константа для единообразия.
FEATURES_PARQUET = os.path.join(OUTPUTS_DIR, "modero_features.parquet")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("baseline")


def build_pipeline() -> Pipeline:
    """Собрать pipeline: импутация медианой → стандартизация → линейная регрессия."""
    # Импутация нужна из-за NaN на стартах сегментов (лаги/окна); медиана с train.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("linreg", LinearRegression()),
    ])


def train_predict(df: pd.DataFrame, features: list[str] | None = None
                  ) -> tuple[Pipeline, pd.Series]:
    """Обучить на train и вернуть (модель, прогноз f_reg на всём ряду)."""
    # Режимные предикторы (без F1_*) и train-сплит для обучения (без утечки).
    features = features or regime_feature_columns(df)
    splits = time_split(df)
    Xtr, ytr = df.loc[splits["train"], features], df.loc[splits["train"], TARGET]

    # Обучаем pipeline только на train; затем прогноз по всему ряду.
    model = build_pipeline()
    model.fit(Xtr, ytr)
    pred = pd.Series(model.predict(df[features]), index=df.index, name="pred_baseline")
    log.info("Baseline обучен на %d строках train, %d признаков", len(Xtr), len(features))
    return model, pred


def main() -> None:
    """CLI: обучить baseline и напечатать метрики по сплитам × steady/transient."""
    # Загружаем признаки, обучаем, считаем метрики модели и persistence.
    df = pd.read_parquet(FEATURES_PARQUET)
    _, pred = train_predict(df)
    df = df.assign(pred_baseline=pred, pred_persist=persistence_predictions(df))

    # Сводные метрики: наша модель vs persistence (ориентир).
    splits = time_split(df)
    tbl = pd.concat([
        evaluate_predictions(df, "pred_baseline", splits, "LinReg (baseline)"),
        evaluate_predictions(df, "pred_persist", splits, "persistence"),
    ], ignore_index=True)
    print("\n===== Слой 1 — baseline (RMSE/MAE/R²) =====")
    print(tbl.round({"RMSE": 5, "MAE": 5, "R2": 4}).to_string(index=False))


if __name__ == "__main__":
    main()
