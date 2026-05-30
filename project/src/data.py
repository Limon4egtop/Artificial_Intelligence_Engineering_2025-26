from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"text", "label"}
ALLOWED_LABELS = {"normal", "spam", "profanity"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {path}")

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют колонки: {missing_columns}")

    df = df[["text", "label"]].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[df["text"] != ""]
    df = df[df["label"] != ""]

    unknown_labels = set(df["label"].unique()) - ALLOWED_LABELS
    if unknown_labels:
        raise ValueError(f"Найдены неизвестные классы: {unknown_labels}")

    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    return df