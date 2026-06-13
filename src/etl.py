"""Этап 1 — ETL: сборка единой чистой почасовой таблицы (МоДеРо).

Модуль собирает артефакт ``outputs/modero_clean.csv`` из двух исходных файлов:

* **Файл A** (сырой ПТК) — почасовые F1 и признаки режима, основа для обучения.
* **Файл B** (Prism) — подпериод A с остатками эталонной модели Prism (бенчмарк).

Пайплайн (см. раздел 4 брифа):
    1. парсинг сырого xlsx (две строки заголовков, почасовая метка времени);
    2. парсинг Prism-файла (остатки — текст с запятой-десятичной, ``replace(',', '.')``);
    3. merge по времени + проверка совпадения F1 (max|ΔF1| должна быть 0);
    4. сегментация по разрывам шага ≠ 1 ч (колонка ``seg``);
    5. флаг переходного часа ``transient = |ΔPa| > 20 МВт`` (ΔPa — внутри сегмента).

Запуск:  ``python -m src.etl``
Результат: ``outputs/modero_clean.csv`` + печатная сводка.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd

# Консоль Windows по умолчанию cp1251 — переводим вывод в UTF-8, чтобы
# кириллица и символы (стрелки) печатались без UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# --- Воспроизводимость и константы пайплайна ------------------------------
# RANDOM_STATE здесь не используется (ETL детерминирован), но фиксируется во
# всех модулях проекта по требованию воспроизводимости (раздел 9 брифа).
RANDOM_STATE = 42
HOUR = pd.Timedelta(hours=1)        # ожидаемый шаг непрерывного ряда
TRANSIENT_DPA_MW = 20.0             # порог переходного часа: |ΔPa| > 20 МВт
PRISM_SCALE_DP1 = 517.7169          # коэффициент масштаба «ДП1» = Prism × const

# Пути по умолчанию относительно корня проекта (на уровень выше src/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
CLEAN_CSV = os.path.join(OUTPUTS_DIR, "modero_clean.csv")

# Шаблоны имён исходных файлов (etl устойчив к точному имени/пробелам).
RAW_A_GLOB = "Данные*.xlsx"
PRISM_GLOB = "Prism*.xlsx"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("etl")

# --- Схема колонок --------------------------------------------------------
# В файле A данные начинаются с 3-й строки (две строки заголовков).
# Метка времени — колонка 5 (почасовой datetime), F1 — колонка 1;
# текстовая метка «...-55» в колонке 0 не используется.
A_TIME_COL = 5
A_F1_COL = 1
# Признаки режима: индекс колонки в файле A -> каноническое имя в чистой таблице.
A_FEATURE_COLS = {
    6: "Pa", 7: "Pr", 8: "Td", 9: "Tg", 10: "Tst",
    11: "Tcnd1", 12: "Tcnd2",                       # Тп.ЦНД-1/2
    13: "Tgpz_a", 14: "Tgpz_b", 15: "Tgpz_v", 16: "Tgpz_g",   # Т ГПЗ н.А–Г
    17: "Tturb_a", 18: "Tturb_b", 19: "Tturb_v", 20: "Tturb_g",  # Т турб. н.А–Г
    21: "Fr",
}
# Порядок признаков в выходной таблице (для предсказуемого CSV).
FEATURE_ORDER = list(A_FEATURE_COLS.values())

# В файле B данные начинаются с 5-й строки (четыре строки заголовков).
B_TIME_COL = 8       # «Дата MoДеРо» — почасовой datetime
B_F1_COL = 9         # F1 (должна точно совпадать с A на перекрытии)
B_RESID_PRISM_COL = 3  # «Остатки» Prism — текст с запятой-десятичной, 0 = не посчитано


def _find_file(pattern: str) -> str:
    """Найти единственный исходный файл по шаблону в ``data/``."""
    # Ищем по glob, чтобы не зависеть от пробелов/подчёркиваний в имени.
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"Не найден файл по шаблону {pattern!r} в {DATA_DIR}")
    # Если файлов несколько — берём первый и предупреждаем (неоднозначность).
    if len(matches) > 1:
        log.warning("По шаблону %r найдено несколько файлов, беру %s", pattern, matches[0])
    return matches[0]


def _comma_float(value) -> float:
    """Преобразовать значение с запятой-десятичной в float (NaN для пустого)."""
    # Текстовые остатки Prism записаны как «-0,00223287» → меняем запятую на точку.
    if isinstance(value, str):
        value = value.replace(",", ".")
    # Пустые ячейки/нечисловое → NaN, чтобы не ломать арифметику ниже.
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_raw_a(path: str) -> pd.DataFrame:
    """Загрузить файл A: почасовые F1 + признаки режима, отсортированные по времени."""
    # Читаем без заголовка: первые две строки — групповые/именные заголовки.
    raw = pd.read_excel(path, header=None)
    body = raw.iloc[2:].reset_index(drop=True)

    # Собираем чистый кадр: время, F1 и признаки режима по карте колонок.
    out = pd.DataFrame({"time": pd.to_datetime(body[A_TIME_COL])})
    out["F1"] = pd.to_numeric(body[A_F1_COL], errors="coerce")
    for col_idx, name in A_FEATURE_COLS.items():
        out[name] = pd.to_numeric(body[col_idx], errors="coerce")

    # Гарантируем хронологический порядок — основа для шага/сегментации.
    out = out.sort_values("time").reset_index(drop=True)
    log.info("Файл A: %d строк, %s → %s", len(out), out["time"].min(), out["time"].max())
    return out


def load_prism(path: str) -> pd.DataFrame:
    """Загрузить файл B: время, F1 и остаток Prism (текст-запятая, 0 → NaN)."""
    # Данные начинаются с 5-й строки (четыре строки заголовков).
    raw = pd.read_excel(path, header=None)
    body = raw.iloc[4:].reset_index(drop=True)

    # Время и F1 для сверки; остаток Prism парсим через запятую-десятичную.
    out = pd.DataFrame({"time": pd.to_datetime(body[B_TIME_COL])})
    out["F1_prism"] = pd.to_numeric(body[B_F1_COL], errors="coerce")
    resid = body[B_RESID_PRISM_COL].map(_comma_float)

    # DATA HYGIENE (важно для отчёта). Остаток валиден только в 1219 из 6679
    # часов перекрытия; остальные 5460 — РОВНЫЙ 0, разбросанный по всему
    # диапазону (не сплошным блоком). Для остатка модели с 6-значной F1
    # значение ровно 0.00000000 физически невозможно — это маркер «не
    # посчитано» (Prism не выдал прогноз на этот час), а НЕ нулевой остаток.
    # Поэтому 0 → NaN: иначе усреднение по нулям занижает RMSE остатка
    # (0.0050 Гц на 1219 реальных часах против ложных 0.0021 на всех 6679).
    out["resid_prism"] = resid.where(resid != 0, np.nan)

    out = out.sort_values("time").reset_index(drop=True)
    n_valid = int(out["resid_prism"].notna().sum())
    log.info("Файл B (Prism): %d строк, остаток Prism валиден в %d ч", len(out), n_valid)
    return out


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить ``seg`` — id непрерывного сегмента (разрыв шага ≠ 1 ч → новый seg)."""
    # Разница меток времени; ровно 1 ч = продолжение того же сегмента.
    step = df["time"].diff()
    breaks = step != HOUR
    # Первая строка ряда (diff = NaT) тоже открывает первый сегмент.
    breaks.iloc[0] = True
    # Кумулятивная сумма флагов разрыва даёт сквозную нумерацию сегментов с 1.
    df = df.copy()
    df["seg"] = breaks.cumsum().astype(int)
    log.info("Сегментация: %d сегментов (разрывов шага ≠ 1ч: %d)",
             df["seg"].nunique(), int(breaks.iloc[1:].sum()))
    return df


def add_transient(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить ΔPa (внутри сегмента) и флаг ``transient = |ΔPa| > 20 МВт``."""
    # ΔPa считаем ВНУТРИ сегмента: через разрыв разность не имеет смысла
    # (первый час каждого сегмента → NaN и не может быть переходным).
    df = df.copy()
    df["dPa"] = df.groupby("seg")["Pa"].diff()
    # Переходный час — скачок активной мощности выше порога по модулю.
    df["transient"] = (df["dPa"].abs() > TRANSIENT_DPA_MW).astype("boolean")
    # NaN ΔPa (старт сегмента) оставляем как <NA> в transient — не «ложный» режим.
    df.loc[df["dPa"].isna(), "transient"] = pd.NA
    share = 100 * df["transient"].mean(skipna=True)
    log.info("Переходные часы: %.3f%% (n=%d из %d валидных ΔPa)",
             share, int(df["transient"].sum()), int(df["transient"].notna().sum()))
    return df


def merge_prism(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Приклеить остаток Prism к таблице A по времени и сверить F1 (max|ΔF1| = 0)."""
    # Левый merge по метке времени: A — основа, B покрывает только подпериод.
    merged = df_a.merge(
        df_b[["time", "F1_prism", "resid_prism"]], on="time", how="left"
    )

    # Проверка целостности: на перекрытии F1 в A и B обязана совпадать точно.
    overlap = merged["F1_prism"].notna()
    max_df1 = float((merged.loc[overlap, "F1"] - merged.loc[overlap, "F1_prism"]).abs().max())
    log.info("Перекрытие с Prism: %d ч; max|ΔF1| = %.3g", int(overlap.sum()), max_df1)
    # Жёсткая проверка по брифу — расхождение означает рассинхрон меток времени.
    if max_df1 != 0:
        log.warning("max|ΔF1| ≠ 0 (%.3g) — проверьте согласование меток времени!", max_df1)

    # Служебную колонку F1_prism убираем — она нужна была только для сверки.
    return merged.drop(columns=["F1_prism"])


def build_clean(raw_a_path: str | None = None, prism_path: str | None = None,
                out_path: str = CLEAN_CSV) -> pd.DataFrame:
    """Оркестратор ETL: собрать, сегментировать, разметить и сохранить чистую таблицу."""
    # Разрешаем пути (по умолчанию — поиск по шаблону в data/).
    raw_a_path = raw_a_path or _find_file(RAW_A_GLOB)
    prism_path = prism_path or _find_file(PRISM_GLOB)

    # Шаги 1–2: загрузка двух источников.
    df_a = load_raw_a(raw_a_path)
    df_b = load_prism(prism_path)

    # Шаги 4–5: сегментация и флаг переходного часа на основе A.
    df = add_segments(df_a)
    df = add_transient(df)

    # Шаг 3: merge остатков Prism + сверка F1.
    df = merge_prism(df, df_b)

    # Упорядочиваем колонки: время, цель, признаки, служебные, остаток.
    cols = ["time", "F1", *FEATURE_ORDER, "seg", "dPa", "transient", "resid_prism"]
    df = df[cols]

    # Сохраняем артефакт для последующих этапов и отчёта.
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Сохранено: %s (%d строк, %d колонок)", out_path, len(df), df.shape[1])
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Печать сводки по чистой таблице (для контроля результата ETL)."""
    # Базовые объёмы и временной охват.
    n_overlap = int(df["resid_prism"].notna().sum())
    valid_tr = df["transient"].notna()
    print("\n================= СВОДКА ETL =================")
    print(f"Строк (часов):            {len(df)}")
    print(f"Период:                   {df['time'].min()} → {df['time'].max()}")
    print(f"Сегментов:                {df['seg'].nunique()}")
    print(f"Переходных часов:         {int(df['transient'].sum())} "
          f"({100 * df['transient'].mean(skipna=True):.2f}% от {int(valid_tr.sum())} валидных ΔPa)")
    print(f"Перекрытие с Prism:       {n_overlap} ч с валидным остатком")
    if n_overlap:
        r = df["resid_prism"].dropna()
        print(f"  RMSE остатка Prism:     {np.sqrt((r ** 2).mean()):.4f} Гц")
    print(f"F1: mean={df['F1'].mean():.4f}  std={df['F1'].std():.4f}  "
          f"min={df['F1'].min():.4f}  max={df['F1'].max():.4f}")
    print(f"Пропусков в F1/признаках: {int(df[['F1', *FEATURE_ORDER]].isna().sum().sum())}")
    print("=============================================\n")


def main() -> None:
    """CLI-точка входа: собрать чистую таблицу и напечатать сводку."""
    # Параметры пути опциональны — по умолчанию ищем файлы в data/.
    parser = argparse.ArgumentParser(description="ETL МоДеРо (этап 1)")
    parser.add_argument("--raw-a", default=None, help="путь к сырому файлу A (.xlsx)")
    parser.add_argument("--prism", default=None, help="путь к файлу Prism (.xlsx)")
    parser.add_argument("--out", default=CLEAN_CSV, help="путь к выходному CSV")
    args = parser.parse_args()

    # Собираем таблицу и показываем итоговую сводку.
    df = build_clean(args.raw_a, args.prism, args.out)
    print_summary(df)


if __name__ == "__main__":
    main()
