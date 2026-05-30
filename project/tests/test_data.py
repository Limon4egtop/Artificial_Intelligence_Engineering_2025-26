from pathlib import Path

from project.src.data import load_dataset


def test_load_dataset_has_required_columns():
    dataset = load_dataset("project/data/messages.csv")

    assert "text" in dataset.columns
    assert "label" in dataset.columns
    assert len(dataset) > 0


def test_load_dataset_has_three_classes():
    dataset = load_dataset("project/data/messages.csv")

    assert set(dataset["label"].unique()) == {
        "normal",
        "spam",
        "profanity",
    }


def test_dataset_has_no_empty_texts():
    dataset = load_dataset("project/data/messages.csv")

    assert dataset["text"].str.strip().ne("").all()


def test_dataset_file_exists():
    assert Path("project/data/messages.csv").exists()