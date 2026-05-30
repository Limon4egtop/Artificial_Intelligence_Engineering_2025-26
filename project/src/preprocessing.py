import re
import unicodedata


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
EXTRA_SPACES_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_PATTERN.sub(" URL ", text)
    text = MENTION_PATTERN.sub(" USER ", text)
    text = text.replace("ё", "е")
    text = EXTRA_SPACES_PATTERN.sub(" ", text)
    return text.strip()