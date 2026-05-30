from project.src.model import build_model


def test_model_can_fit_and_predict_multiclass():
    texts = [
        "Привет, завтра во сколько встречаемся?",
        "Отправил файл с отчетом",
        "Купи рекламу и заработай деньги",
        "Скидки только сегодня, пиши в личку",
        "Ты дурак",
        "Матерное сообщение",
    ]

    labels = [
        "normal",
        "normal",
        "spam",
        "spam",
        "profanity",
        "profanity",
    ]

    model = build_model()
    model.fit(texts, labels)

    prediction = model.predict(["Привет, как дела?"])[0]

    assert prediction in {"normal", "spam", "profanity"}


def test_model_returns_probabilities():
    texts = [
        "Привет",
        "Купи рекламу",
        "Ты дурак",
    ]

    labels = [
        "normal",
        "spam",
        "profanity",
    ]

    model = build_model()
    model.fit(texts, labels)

    probabilities = model.predict_proba(["Привет"])[0]

    assert len(probabilities) == 3
    assert round(sum(probabilities), 5) == 1.0