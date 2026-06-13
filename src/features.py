"""Этап 3 — конструирование признаков (МоДеРо).

Из чистой почасовой таблицы ``outputs/modero_clean.csv`` строит расширенный
набор признаков для слоя 1 (режимная модель ``f_reg``):

* **режимные** — мгновенные значения нагрузки/температур (passthrough);
* **рампы** — скорость изменения мощности ``|ΔPa|`` на горизонтах 1/3/6 ч;
* **лаги** — F1 и драйверы режима с задержкой (структура автокорреляции
  F1: 0.87 @1ч, 0.37 @6ч, 0.49 @24ч — суточный отклик);
* **скользящие** — среднее/СКО драйверов и F1 в окнах 3/6/24 ч;
* **календарные** — час суток (циклически) и день недели.

ЖЁСТКОЕ ПРАВИЛО (раздел 5 брифа): все лаги, рампы и скользящие считаются
**ТОЛЬКО внутри сегментов** (``groupby('seg')``). Через разрыв ряда (новый
``seg``) разности/окна не протягиваются — иначе утечка через «дыру».
Дополнительно: признаки от F1 берутся со сдвигом (только прошлое, ``t-1`` и
ранее), чтобы не было утечки целевой переменной в момент ``t``.

Запуск:  ``python -m src.features``
Результат: ``outputs/modero_features.parquet``
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

# Консоль Windows (cp1251) — переводим вывод в UTF-8 (стрелки/кириллица).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# --- Воспроизводимость и пути ---------------------------------------------
RANDOM_STATE = 42  # фиксируется во всех модулях (этап детерминирован).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
CLEAN_CSV = os.path.join(OUTPUTS_DIR, "modero_clean.csv")
FEATURES_PARQUET = os.path.join(OUTPUTS_DIR, "modero_features.parquet")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("features")

# --- Конфигурация набора признаков ----------------------------------------
# Режимные признаки (passthrough из чистой таблицы) — известны в момент t.
REGIME_COLS = [
    "Pa", "Pr", "Td", "Tg", "Tst", "Tcnd1", "Tcnd2",
    "Tgpz_a", "Tgpz_b", "Tgpz_v", "Tgpz_g",
    "Tturb_a", "Tturb_b", "Tturb_v", "Tturb_g", "Fr",
]
# Главные драйверы режима — их лагируем и сглаживаем (коррелируют с F1).
DRIVERS = ["Pa", "Pr", "Td", "Tg", "Tst", "Fr"]

F1_LAGS = [1, 6, 24]        # лаги цели (по структуре автокорреляции)
DRIVER_LAGS = [1, 24]       # лаги драйверов: ближний + суточный
RAMP_COLS = ["Pa", "Pr"]    # для каких величин считаем рампы
RAMP_HORIZONS = [1, 3, 6]   # горизонты рампы, ч
ROLL_COLS = ["Pa", "Pr", "Tst"]   # драйверы для скользящих
ROLL_WINDOWS = [3, 6, 24]   # окна скользящих, ч
F1_ROLL_WINDOWS = [6, 24]   # окна скользящих по F1 (на прошлом, со сдвигом)

# --- Тепловая динамика режима (итерация усиления f_reg, без признаков F1) ---
# F1 реагирует на тепловое состояние с ЗАДЕРЖКОЙ (инерция металла/среды), а не
# мгновенно — добавляем динамику и «накопленное» тепло температурных драйверов.
# Агрегаты Т ГПЗ / Т турб по группам А–Г (среднее) — компактный тепловой признак.
TGPZ_GROUP = ["Tgpz_a", "Tgpz_b", "Tgpz_v", "Tgpz_g"]
TTURB_GROUP = ["Tturb_a", "Tturb_b", "Tturb_v", "Tturb_g"]
THERMAL_COLS = ["Td", "Tg", "Tst", "Tcnd1", "Tcnd2", "Tgpz_mean", "Tturb_mean"]
THERMAL_LAGS = [6, 24]         # задержка теплового отклика
THERMAL_ROLL_WINDOWS = [6, 24]  # сглаживание теплового состояния
THERMAL_EWMA_HALFLIFE = 24     # экспоненциальная инерция, ч
THERMAL_ACCUM_WINDOW = 72      # «накопленное» тепло за ~3 суток, ч


def _roll_within_seg(df: pd.DataFrame, col: str, window: int, stat: str,
                     shift: int = 0) -> pd.Series:
    """Скользящая статистика по ``col`` ВНУТРИ сегмента (полное окно, без частичных)."""
    # Группируем по seg и при необходимости сдвигаем (shift>0 — только прошлое).
    s = df.groupby("seg")[col].shift(shift) if shift else df[col]
    # min_periods=window → никаких частичных окон: старт сегмента даёт NaN.
    grouped = s.groupby(df["seg"])
    roller = grouped.transform(lambda x: getattr(x.rolling(window, min_periods=window), stat)())
    return roller


def add_ramps(df: pd.DataFrame) -> pd.DataFrame:
    """Рампы мощности: ΔX и |ΔX| на горизонтах 1/3/6 ч, внутри сегментов."""
    df = df.copy()
    # Для каждой величины и горизонта берём разность внутри сегмента.
    for col in RAMP_COLS:
        for h in RAMP_HORIZONS:
            d = df.groupby("seg")[col].diff(h)
            df[f"d{col}_{h}h"] = d
            # Модуль рампы — ключевой признак переходности (|ΔPa|).
            df[f"abs_d{col}_{h}h"] = d.abs()
    log.info("Рампы: %d признаков (cols=%s, horizons=%s)",
             len(RAMP_COLS) * len(RAMP_HORIZONS) * 2, RAMP_COLS, RAMP_HORIZONS)
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Лаги F1 и драйверов режима, сдвиг ТОЛЬКО внутри сегмента."""
    df = df.copy()
    # Лаги цели: прошлые значения F1 (утечки нет — это t-1, t-6, t-24).
    for lag in F1_LAGS:
        df[f"F1_lag{lag}"] = df.groupby("seg")["F1"].shift(lag)
    # Лаги драйверов режима — ближний и суточный отклик.
    for col in DRIVERS:
        for lag in DRIVER_LAGS:
            df[f"{col}_lag{lag}"] = df.groupby("seg")[col].shift(lag)
    log.info("Лаги: F1=%s, драйверы=%s × %s", F1_LAGS, DRIVERS, DRIVER_LAGS)
    return df


def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Скользящие среднее/СКО: драйверы (вкл. t) и F1 (только прошлое, сдвиг 1)."""
    df = df.copy()
    # Драйверы: режим в момент t известен → окно включает текущее значение.
    for col in ROLL_COLS:
        for w in ROLL_WINDOWS:
            df[f"{col}_rollmean_{w}"] = _roll_within_seg(df, col, w, "mean")
            df[f"{col}_rollstd_{w}"] = _roll_within_seg(df, col, w, "std")
    # F1: чтобы НЕ утекла цель, скользящие считаем по прошлому (shift=1).
    for w in F1_ROLL_WINDOWS:
        df[f"F1_rollmean_{w}"] = _roll_within_seg(df, "F1", w, "mean", shift=1)
        df[f"F1_rollstd_{w}"] = _roll_within_seg(df, "F1", w, "std", shift=1)
    log.info("Скользящие: драйверы %s × окна %s; F1 × окна %s (сдвиг 1)",
             ROLL_COLS, ROLL_WINDOWS, F1_ROLL_WINDOWS)
    return df


def add_thermal(df: pd.DataFrame) -> pd.DataFrame:
    """Тепловая динамика: агрегаты температур + лаги/скользящие/EWMA/накопление.

    Все операции — ВНУТРИ сегментов; признаки производны только от температур
    режима (не от F1). Цель — дать f_reg задержанный тепловой отклик.
    """
    df = df.copy()
    # Агрегаты групп Т ГПЗ и Т турб (среднее по А–Г) — компактное тепловое состояние.
    df["Tgpz_mean"] = df[TGPZ_GROUP].mean(axis=1)
    df["Tturb_mean"] = df[TTURB_GROUP].mean(axis=1)

    # Лаги и скользящие средние температур — задержка и сглаживание отклика.
    for col in THERMAL_COLS:
        for lag in THERMAL_LAGS:
            df[f"{col}_lag{lag}"] = df.groupby("seg")[col].shift(lag)
        for w in THERMAL_ROLL_WINDOWS:
            df[f"{col}_rollmean_{w}"] = _roll_within_seg(df, col, w, "mean")

    # Тепловая инерция: EWMA (экспоненциальное «затухающее» прошлое) и
    # накопленное среднее за ~3 суток — медленное тепловое состояние металла.
    for col in THERMAL_COLS:
        ewma = df.groupby("seg")[col].transform(
            lambda s: s.ewm(halflife=THERMAL_EWMA_HALFLIFE, min_periods=1).mean())
        df[f"{col}_ewma{THERMAL_EWMA_HALFLIFE}"] = ewma
        df[f"{col}_accum{THERMAL_ACCUM_WINDOW}"] = _roll_within_seg(
            df, col, THERMAL_ACCUM_WINDOW, "mean")
    log.info("Тепловые: агрегаты Tgpz/Tturb + лаги/скользящие/EWMA/накопление по %d темп.",
             len(THERMAL_COLS))
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Календарные признаки: час суток (sin/cos) и день недели."""
    df = df.copy()
    # Час суток — циклическое кодирование (суточный отклик нагрузки).
    hour = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # День недели — недельный график нагрузки (отклик слабый, но дёшев).
    df["dow"] = df["time"].dt.dayofweek
    log.info("Календарные: hour_sin/cos, dow")
    return df


def build_features(clean_path: str = CLEAN_CSV,
                   out_path: str = FEATURES_PARQUET,
                   with_thermal: bool = False) -> pd.DataFrame:
    """Оркестратор этапа 3: собрать таблицу признаков из чистой таблицы.

    ``with_thermal=False`` — рабочий ПАРСИМОНИЧНЫЙ режимный набор (по умолчанию).
    ``with_thermal=True`` — добавляет тепловую динамику (лаги/EWMA/накопление
    температур); сохранён как абляция: на слое 1 out-of-sample R² не улучшил
    (0.042→0.036) и усилил переобучение — в рабочую f_reg не входит.
    """
    # Читаем артефакт ETL; seg/transient уже размечены, F1 без пропусков.
    df = pd.read_csv(clean_path, parse_dates=["time"])
    df = df.sort_values(["seg", "time"]).reset_index(drop=True)
    log.info("Вход: %s — %d строк, %d сегментов", clean_path, len(df), df["seg"].nunique())

    # Последовательно добавляем группы признаков (все — внутри сегментов).
    df = add_ramps(df)
    df = add_lags(df)
    df = add_rolling(df)
    # Тепловая динамика — только в режиме абляции (см. docstring).
    if with_thermal:
        df = add_thermal(df)
    df = add_calendar(df)

    # Сохраняем в parquet (компактно, сохраняет типы и NaN на старте сегментов).
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Сохранено: %s (%d строк, %d колонок)", out_path, len(df), df.shape[1])
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Сводка по таблице признаков (контроль состава и доли NaN)."""
    # Считаем сконструированные признаки (всё, кроме служебных/цели).
    base = {"time", "F1", "seg", "transient", "resid_prism", "dPa"}
    feats = [c for c in df.columns if c not in base]
    print("\n============== СВОДКА ПРИЗНАКОВ ==============")
    print(f"Строк:                    {len(df)}")
    print(f"Всего колонок:            {df.shape[1]}")
    print(f"Признаков (без служебных):{len(feats)}")
    print(f"  режимных:               {len(REGIME_COLS)}")
    print(f"  рампы/лаги/скользящие/кал: {len(feats) - len(REGIME_COLS)}")
    # Доля NaN: ожидаемо растёт на стартах сегментов (лаги/окна).
    na_share = 100 * df[feats].isna().mean().mean()
    full_rows = int((~df[feats].isna().any(axis=1)).sum())
    print(f"Средняя доля NaN в признаках: {na_share:.2f}%")
    print(f"Строк без единого NaN:    {full_rows} ({100 * full_rows / len(df):.1f}%)")
    print(f"Самый «дырявый» признак:  макс {100 * df[feats].isna().mean().max():.1f}% NaN "
          f"(окно 24ч на коротких сегментах)")
    print("=============================================\n")


def main() -> None:
    """CLI-точка входа: построить признаки и напечатать сводку."""
    import argparse
    # --thermal — собрать абляционный набор с тепловой динамикой в отдельный файл.
    parser = argparse.ArgumentParser(description="Конструирование признаков (этап 3)")
    parser.add_argument("--thermal", action="store_true",
                        help="абляция: добавить тепловую динамику (в modero_features_thermal.parquet)")
    args = parser.parse_args()
    out = os.path.join(OUTPUTS_DIR, "modero_features_thermal.parquet") if args.thermal else FEATURES_PARQUET
    df = build_features(out_path=out, with_thermal=args.thermal)
    print_summary(df)


if __name__ == "__main__":
    main()
