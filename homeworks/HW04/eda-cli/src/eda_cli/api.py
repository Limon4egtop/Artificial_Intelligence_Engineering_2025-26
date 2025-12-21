from __future__ import annotations

from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
)


# ---------- Pydantic схемы ----------


class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус сервиса")


class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок в датасете")
    max_missing_share: float = Field(..., ge=0.0, le=1.0, description="Максимальная доля пропусков по колонкам (0..1)")
    numeric_cols: int = Field(..., ge=0, description="Количество числовых колонок")
    categorical_cols: int = Field(..., ge=0, description="Количество категориальных колонок")


class QualityResponse(BaseModel):
    ok_for_model: bool = Field(..., description="Можно ли (по эвристикам) обучать модель на этих данных")
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с простыми эвристиками качества",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета (n_rows, n_cols)",
    )


# ---------- /health ----------


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


# ---------- /quality: заглушка (без чтения реального CSV) ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"], summary="Оценка качества по агрегированным признакам")
def quality(req: QualityRequest) -> QualityResponse:
    start = perf_counter()

    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    score = 1.0
    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    if flags["too_many_missing"]:
        score -= 0.5
    if flags["no_numeric_columns"]:
        score -= 0.1
    if flags["no_categorical_columns"]:
        score -= 0.05

    score = max(0.0, min(1.0, float(score)))
    ok_for_model = score >= 0.7

    message = (
        "Датасет выглядит пригодным для обучения модели (по текущим эвристикам)."
        if ok_for_model
        else "Датасет выглядит проблемным для обучения модели (по текущим эвристикам)."
    )

    latency_ms = (perf_counter() - start) * 1000.0

    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    start = perf_counter()

    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file with .csv extension.")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}") from e

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty (no rows).")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # ВАЖНО: передаём df=df, чтобы считались HW03-флаги, которым нужен исходный DataFrame
    flags_all = compute_quality_flags(summary, missing_df, df=df)

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета берём из summary (если там есть поля n_rows/n_cols),
    # иначе — напрямую из DataFrame.
    n_rows = int(getattr(summary, "n_rows", df.shape[0]))
    n_cols = int(getattr(summary, "n_cols", df.shape[1]))

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- HW04: /quality-flags-from-csv (возвращает только флаги качества) ----------


class QualityFlagsResponse(BaseModel):
    flags: dict[str, bool] = Field(
        ...,
        description="Полный набор булевых флагов качества, рассчитанных на основе EDA-ядра.",
    )


@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["quality"],
    summary="Вернуть только булевы флаги качества по CSV-файлу (HW04)",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    HW04-эндпоинт.

    Принимает CSV-файл, читает его в DataFrame, запускает:
    - summarize_dataset
    - missing_table
    - compute_quality_flags

    И возвращает ТОЛЬКО булевы флаги (без quality_score и прочих численных тех. полей).
    """
    start = perf_counter()

    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file with .csv extension.")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}") from e

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty (no rows).")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df, df=df)

    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    latency_ms = (perf_counter() - start) * 1000.0
    print(
        f"[quality-flags-from-csv] filename={file.filename!r} "
        f"n_rows={df.shape[0]} n_cols={df.shape[1]} latency_ms={latency_ms:.1f} ms"
    )

    return QualityFlagsResponse(flags=flags_bool)
