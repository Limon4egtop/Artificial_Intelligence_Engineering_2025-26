from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from project.src.data import load_dataset
from project.src.model import build_model
from project.src.preprocessing import normalize_text


DATASET_PATH = Path("data/messages.csv")
MODEL_PATH = Path("artifacts/models/message_classifier.joblib")
METRICS_PATH = Path("artifacts/metrics/classification_report.txt")
CONFUSION_MATRIX_PATH = Path("artifacts/metrics/confusion_matrix.csv")


def main() -> None:
    df = load_dataset(DATASET_PATH)

    df["text"] = df["text"].apply(normalize_text)

    print("Размер датасета:", len(df))
    print("Распределение классов:")
    print(df["label"].value_counts())

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = build_model()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_,
    )

    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(report, encoding="utf-8")

    pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_,
    ).to_csv(CONFUSION_MATRIX_PATH, sep=";", encoding="utf-8-sig")

    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Отчёт сохранён: {METRICS_PATH}")
    print(f"Матрица ошибок сохранена: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()