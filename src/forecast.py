"""Этап 6 — слой 3: прогноз индикатора H(t+h) и F1(t+h) на горизонт.

Декомпозиция Варианта C даёт прогноз F1 без «лобовой» экстраполяции:
    F1̂(t+h) = f_reg(x(t+h)) + Ĥ(t+h),
где f_reg(x(t+h)) — режимная часть по ПЛАНУ нагрузки на t+h (в бэктесте — по
фактическому будущему режиму; деплой требует диспетчерского плана), а Ĥ(t+h) —
прогноз индикатора здоровья (медленный остаток).

Горизонты: основные **24 ч и 72 ч**; короткие **1/6 ч** — как точки кривой
ошибки (раздел 8 брифа). Бэктест — **rolling-origin** на OOS-периоде (val+test,
с 07.2016): каждый час периода = независимый ориентир прогноза.

Сравниваемые модели F1̂(t+h):
* **persistence** — F1̂ = F1(t) (сильный наивный ориентир: F1 почти постоянна);
* **f_reg_only** — F1̂ = f_reg(x(t+h)) (режим без здоровья, Ĥ=0);
* **dec_resid** — f_reg(x(t+h)) + r(t) (здоровье = сырой остаток на t);
* **dec_ewma** — f_reg(x(t+h)) + H(t) (здоровье = сглаженный индикатор).

Сдвиги во времени — строго ВНУТРИ сегментов (прогноз через разрыв бессмыслен).

Запуск:  ``python -m src.forecast``
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

from src.eval import SPLIT_BOUNDS, regression_metrics

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RANDOM_STATE = 42
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
RESIDUALS_PARQUET = os.path.join(OUTPUTS_DIR, "modero_residuals.parquet")
HEALTH_PARQUET = os.path.join(OUTPUTS_DIR, "health_series.parquet")
F1_METRICS_CSV = os.path.join(OUTPUTS_DIR, "layer3_forecast_metrics.csv")
H_METRICS_CSV = os.path.join(OUTPUTS_DIR, "layer3_H_forecast.csv")
RMSE_PNG = os.path.join(OUTPUTS_DIR, "layer3_rmse_vs_horizon.png")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("forecast")

HORIZONS = [1, 6, 24, 72]               # 24/72 — основные; 1/6 — контекст
BACKTEST_START = SPLIT_BOUNDS["val"][0]  # OOS-период: val+test (с 07.2016)
F1_MODELS = ["persistence", "f_reg_only", "dec_resid", "dec_ewma"]


def load_inputs() -> pd.DataFrame:
    """Слить остатки слоя 1 и индикатор H слоя 2 по времени, отсортировать."""
    # Нужны F1, pred_gbm, resid_gbm (слой 1) и H_ewma (слой 2).
    res = pd.read_parquet(RESIDUALS_PARQUET)
    hea = pd.read_parquet(HEALTH_PARQUET)[["time", "H_ewma", "in_envelope"]]
    df = res.merge(hea, on="time", how="left").sort_values(["seg", "time"]).reset_index(drop=True)
    return df


def forecast_horizon(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Собрать прогнозы всех моделей F1̂(t+h) и факт F1(t+h), сдвиг внутри сегмента."""
    g = df.groupby("seg")
    # Будущее (t+h) внутри сегмента: NaN на хвосте сегмента (через разрыв не тянем).
    fut_F1 = g["F1"].shift(-h)
    fut_pred = g["pred_gbm"].shift(-h)
    fut_transient = g["transient"].shift(-h)

    # Четыре прогноза: persistence и три варианта декомпозиции (Вариант C).
    out = pd.DataFrame({
        "time": df["time"], "horizon": h, "y_true": fut_F1,
        "persistence": df["F1"],                       # F1̂ = F1(t)
        "f_reg_only": fut_pred,                         # f_reg(x(t+h)), Ĥ=0
        "dec_resid": fut_pred + df["resid_gbm"],        # + сырой остаток r(t)
        "dec_ewma": fut_pred + df["H_ewma"],            # + сглаженный H(t)
        "transient_fut": fut_transient,
    })
    # Ориентиры только OOS (val+test) и с валидным будущим фактом.
    keep = (df["time"] >= pd.Timestamp(BACKTEST_START)) & out["y_true"].notna()
    return out[keep]


def evaluate_f1(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE/MAE по горизонтам × моделям, с разбивкой steady/transient (24/72ч)."""
    rows = []
    # Для каждого горизонта собираем прогнозы и считаем метрики по подмножествам.
    for h in HORIZONS:
        fc = forecast_horizon(df, h)
        for model in F1_MODELS:
            # subset=all всегда; steady/transient — для основных горизонтов.
            subsets = {"all": pd.Series(True, index=fc.index)}
            if h in (24, 72):
                subsets["steady"] = fc["transient_fut"] == False   # noqa: E712
                subsets["transient"] = fc["transient_fut"] == True  # noqa: E712
            for name, mask in subsets.items():
                mt = regression_metrics(fc.loc[mask, "y_true"], fc.loc[mask, model])
                rows.append({"target": "F1", "model": model, "horizon": h,
                             "subset": name, **mt})
    return pd.DataFrame(rows)


def evaluate_H(df: pd.DataFrame) -> pd.DataFrame:
    """Прогноз индикатора H(t+h)=H(t) (persistence) — RMSE по горизонтам (упреждение)."""
    rows = []
    g = df.groupby("seg")
    # Насколько хорошо текущий H предсказывает будущий H — это и есть упреждение.
    for h in HORIZONS:
        fut_H = g["H_ewma"].shift(-h)
        keep = (df["time"] >= pd.Timestamp(BACKTEST_START)) & fut_H.notna()
        mt = regression_metrics(fut_H[keep], df.loc[keep, "H_ewma"])
        rows.append({"target": "H", "model": "persistence", "horizon": h, "subset": "all", **mt})
    return pd.DataFrame(rows)


def _plot_rmse(f1_metrics: pd.DataFrame, path: str) -> None:
    """Кривая ошибки: RMSE F1̂ vs горизонт для всех моделей (subset=all)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = f1_metrics[f1_metrics["subset"] == "all"]
    # Линия RMSE(h) для каждой модели — наглядно, где декомпозиция бьёт persistence.
    for model, color in zip(F1_MODELS, ["#e76f51", "#264653", "#2a9d8f", "#1f6feb"]):
        d = sub[sub["model"] == model].sort_values("horizon")
        ax.plot(d["horizon"], d["RMSE"], "-o", color=color, label=model)
    ax.set_xlabel("горизонт h, ч"); ax.set_ylabel("RMSE F1̂, Гц")
    ax.set_title("Слой 3: кривая ошибки прогноза F1 по горизонтам")
    ax.set_xticks(HORIZONS); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    log.info("Рисунок сохранён: %s", path)


def main() -> None:
    """CLI: rolling-origin бэктест прогноза F1(t+h) и H(t+h), метрики и рисунок."""
    # Данные слоёв 1–2 и оценка по горизонтам.
    df = load_inputs()
    log.info("Бэктест с %s (OOS), горизонты %s", BACKTEST_START, HORIZONS)
    f1_metrics = evaluate_f1(df)
    h_metrics = evaluate_H(df)
    f1_metrics.round({"RMSE": 5, "MAE": 5, "R2": 4}).to_csv(F1_METRICS_CSV, index=False)
    h_metrics.round({"RMSE": 5, "MAE": 5, "R2": 4}).to_csv(H_METRICS_CSV, index=False)
    _plot_rmse(f1_metrics, RMSE_PNG)

    # --- Печать сводки -----------------------------------------------------
    print("\n========== ЭТАП 6 — прогноз F1(t+h) (RMSE, Гц, subset=all) ==========")
    piv = (f1_metrics[f1_metrics["subset"] == "all"]
           .pivot(index="model", columns="horizon", values="RMSE").reindex(F1_MODELS))
    print(piv.round(5).to_string())
    print("\n--- Тест на переходные часы (t+h transient), горизонты 24/72 ---")
    tr = f1_metrics[(f1_metrics["subset"] == "transient")]
    if len(tr):
        print(tr.pivot(index="model", columns="horizon", values="RMSE").reindex(F1_MODELS).round(5).to_string())
    print("\n========== Прогноз H(t+h)=H(t): RMSE по горизонтам (упреждение) ==========")
    print(h_metrics[["horizon", "n", "RMSE", "MAE"]].round(5).to_string(index=False))
    print("===================================================================\n")


if __name__ == "__main__":
    main()
