from project.src.preprocessing import normalize_text


def test_normalize_text_lowercase():
    assert normalize_text("ПРИВЕТ") == "привет"


def test_normalize_text_replaces_yo():
    assert normalize_text("ёлка") == "елка"


def test_normalize_text_removes_extra_spaces():
    assert normalize_text("привет     как дела") == "привет как дела"


def test_normalize_text_replaces_url():
    result = normalize_text("смотри https://example.com")

    assert "URL" in result or "url" in result