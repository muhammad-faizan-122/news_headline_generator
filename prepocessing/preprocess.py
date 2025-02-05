import re
import unicodedata
import html
from abc import ABC


class TextPreprocessor(ABC):
    def __init__(self):
        pass

    def norm_whitespaces(self, text: str) -> str:
        """Normalize whitespace"""
        return re.sub(r"\s+", " ", text).strip()

    def norm_text(self, text: str) -> str:
        """
        Normalize text while preserving non-ASCII characters.
        Only converts to lowercase and normalizes equivalent Unicode representations.
        """
        # Convert to lowercase
        text = text.lower()

        # Normalize unicode characters to composed form (NFC)
        # This combines characters that should be single units
        # while preserving the original characters
        text = unicodedata.normalize("NFC", text)

        return text

    def remove_html_tags(self, text: str) -> str:
        """Clean HTML entities and tags"""
        # Decode HTML entities
        text = html.unescape(text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        return text

    def remove_urls(self, text: str) -> str:
        """Remove or normalize URLs"""
        url_pattern = r"https?://\S+|www\.\S+"
        text = re.sub(url_pattern, "[URL]", text)
        return text

    def remove_special_chars(self, text: str, preserve_chars=".,!?") -> str:
        """Remove special characters while preserving specified ones"""
        pattern = f'[^a-zA-Z0-9\s{re.escape("".join(preserve_chars))}]'
        text = re.sub(pattern, " ", text)
        return text
