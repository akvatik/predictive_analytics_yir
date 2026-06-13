"""Слой 1 — основная модель: градиентный бустинг (CatBoost) для f_reg.

Режимная модель F1 ≈ f_reg(x): CatBoostRegressor (нативно работает с NaN на
стартах сегментов, не требует масштабирования). Обучение ТОЛЬКО на train,
ранняя остановка по val (раздел 5, без shuffle).

``__main__`` запускает полный прогон слоя 1:
* обучает baseline (LinReg) и CatBoost;
* строит сравнительную таблицу метрик (модели + persistence + Prism);
* сохраняет таблицу, остатки r(t)=F1−f_reg (артефакт для слоя 2) и рисунки.

Запуск:  ``python -m src.models.gbm``
LightGBM в окружении отсутствует — основной моделью взят CatBoost (бриф
допускает CatBoost/LightGBM).
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # без дисплея — только сохранение PNG для ПЗ.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.eval import (OUTPUTS_DIR, SPLIT_BOUNDS, TARGET, evaluate_predictions,
                      persistence_predictions, prism_reference,
                      regime_feature_columns, save_metrics_table, time_split)
from src.models.baseline import train_predict as baseline_train_predict

# Консоль Windows (cp1251) → UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RANDOM_STATE = 42  # фиксируем random_seed CatBoost — воспроизводимость.
FEATURES_PARQUET = os.path.join(OUTPUTS_DIR, "modero_features.parquet")
RESIDUALS_PARQUET = os.path.join(OUTPUTS_DIR, "modero_residuals.parquet")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("gbm")

# Параметры CatBoost: умеренная модель, RMSE-лосс, ранняя остановка по val.
CATBOOST_PARAMS = dict(
    loss_function="RMSE",
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_STATE,
    early_stopping_rounds=100,
    verbose=False,
)


def train_predict_gbm(df: pd.DataFrame, features: list[str] | None = None
                      ) -> tuple[CatBoostRegressor, pd.Series]:
    """Обучить CatBoost на train (ранняя остановка по val) и спрогнозировать весь ряд."""
    # Режимные предикторы (без F1_*); CatBoost принимает NaN, импутация не нужна.
    features = features or regime_feature_columns(df)
    splits = time_split(df)
    train_pool = Pool(df.loc[splits["train"], features], df.loc[splits["train"], TARGET])
    val_pool = Pool(df.loc[splits["val"], features], df.loc[splits["val"], TARGET])

    # Обучаем с ранней остановкой по валидации (контроль переобучения).
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool)
    pred = pd.Series(model.predict(df[features]), index=df.index, name="pred_gbm")
    log.info("CatBoost: лучшая итерация %s, train=%d, val=%d",
             model.get_best_iteration(), len(splits["train"]), len(splits["val"]))
    return model, pred


def oos_train_residuals(df: pd.DataFrame, features: list[str], n_blocks: int = 5
                        ) -> pd.Series:
    """Out-of-sample остатки на train через блочную CV (для калибровки порогов).

    In-sample остатки train занижены (модель видела эти строки) → узкие пороги
    и ложные тревоги на test. Делим train на ``n_blocks`` непрерывных временных
    блоков; каждый блок прогнозируем моделью, обученной на остальных → честный
    масштаб шума здорового остатка.
    """
    # Берём только train-сплит и режем его на непрерывные блоки по времени.
    train_idx = time_split(df)["train"]
    blocks = np.array_split(np.asarray(train_idx), n_blocks)
    oos = pd.Series(np.nan, index=df.index, dtype=float)

    # Для каждого блока обучаемся на остальных блоках и прогнозируем сам блок.
    for blk in blocks:
        blk = pd.Index(blk)
        others = train_idx.difference(blk)
        # Фиксированное число итераций (без early stopping) — модель того же класса.
        m = CatBoostRegressor(loss_function="RMSE", iterations=200, learning_rate=0.03,
                              depth=6, l2_leaf_reg=3.0, random_seed=RANDOM_STATE, verbose=False)
        m.fit(df.loc[others, features], df.loc[others, TARGET])
        oos.loc[blk] = df.loc[blk, TARGET].values - m.predict(df.loc[blk, features])
    log.info("OOS-остатки train (блочная CV, %d блоков): n=%d", n_blocks, int(oos.notna().sum()))
    return oos


def _plot_pred_vs_actual(df: pd.DataFrame, test_idx: pd.Index, path: str) -> None:
    """Рисунок: прогноз CatBoost против факта F1 на тесте (диагностика смещения)."""
    # Точечная диаграмма факт↔прогноз; идеал — на диагонали y=x.
    sub = df.loc[test_idx]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sub[TARGET], sub["pred_gbm"], s=6, alpha=0.3, color="#1f6feb")
    lims = [sub[TARGET].min(), sub[TARGET].max()]
    ax.plot(lims, lims, "r--", lw=1, label="y = x")
    ax.set_xlabel("F1 факт, Гц"); ax.set_ylabel("F1 прогноз (CatBoost), Гц")
    ax.set_title("Слой 1: прогноз vs факт (тест)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log.info("Рисунок сохранён: %s", path)


def _plot_residual_timeseries(df: pd.DataFrame, path: str) -> None:
    """Рисунок: остаток r(t)=F1−f_reg во времени с разметкой сплитов."""
    # Линия остатка + вертикали границ val/test — вход для слоя 2 (индикатор).
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(df["time"], df["resid_gbm"], lw=0.4, color="#444")
    ax.axhline(0, color="k", lw=0.6)
    for b in ("val", "test"):
        ax.axvline(pd.Timestamp(SPLIT_BOUNDS[b][0]), color="r", ls="--", lw=0.8)
    ax.set_xlabel("время"); ax.set_ylabel("r(t) = F1 − f_reg, Гц")
    ax.set_title("Слой 1: остаток режимной модели (вход для индикатора здоровья)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log.info("Рисунок сохранён: %s", path)


def run_layer1() -> pd.DataFrame:
    """Полный прогон слоя 1: обучение, сравнение, сохранение артефактов."""
    # Загружаем признаки; f_reg использует только режимные предикторы (без F1_*).
    df = pd.read_parquet(FEATURES_PARQUET)
    features = regime_feature_columns(df)
    log.info("Слой 1: %d строк, %d режимных признаков (F1_* исключены)", len(df), len(features))

    # Обучаем обе модели на одинаковых train-данных (честное сравнение).
    _, pred_base = baseline_train_predict(df, features)
    _, pred_gbm = train_predict_gbm(df, features)
    df = df.assign(pred_baseline=pred_base, pred_gbm=pred_gbm,
                   pred_persist=persistence_predictions(df))
    # Остаток основной модели — ключевой артефакт для слоя 2 (индикатор H).
    df["resid_gbm"] = df[TARGET] - df["pred_gbm"]
    # OOS-остатки на train (блочная CV) — честный масштаб шума для порогов слоя 2.
    df["resid_oos_train"] = oos_train_residuals(df, features)

    # Сводная таблица: CatBoost, baseline, persistence, Prism (на N посчит.).
    splits = time_split(df)
    table = pd.concat([
        evaluate_predictions(df, "pred_gbm", splits, "CatBoost (f_reg)"),
        evaluate_predictions(df, "pred_baseline", splits, "LinReg (baseline)"),
        evaluate_predictions(df, "pred_persist", splits, "persistence"),
        prism_reference(df, splits),
    ], ignore_index=True)
    save_metrics_table(table, "layer1_metrics.csv")

    # Сохраняем остатки (time/seg/transient/F1/pred/resid) для слоёв 2–3.
    resid_cols = ["time", "seg", "transient", TARGET, "pred_gbm", "resid_gbm",
                  "resid_oos_train", "resid_prism"]
    df[resid_cols].to_parquet(RESIDUALS_PARQUET, index=False)
    log.info("Остатки сохранены: %s", RESIDUALS_PARQUET)

    # Рисунки для практического раздела ПЗ.
    _plot_pred_vs_actual(df, splits["test"], os.path.join(OUTPUTS_DIR, "layer1_pred_vs_actual.png"))
    _plot_residual_timeseries(df, os.path.join(OUTPUTS_DIR, "layer1_residual_timeseries.png"))
    return table


def _print_table(table: pd.DataFrame) -> None:
    """Печать сводной таблицы метрик слоя 1 (компактно, для контроля)."""
    # Фокус на 'all' по сплитам + разбивка steady/transient на тесте.
    show = table.round({"RMSE": 5, "MAE": 5, "R2": 4})
    print("\n===== Слой 1 — сводка (subset=all) =====")
    print(show[show["subset"] == "all"].to_string(index=False))
    print("\n===== Тест: steady vs transient =====")
    print(show[(show["split"] == "test") & (show["subset"] != "all")].to_string(index=False))


def main() -> None:
    """CLI-точка входа: прогнать слой 1 и напечатать сводку метрик."""
    table = run_layer1()
    _print_table(table)


if __name__ == "__main__":
    main()
