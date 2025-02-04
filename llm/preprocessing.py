from abc import ABC, abstractmethod
import re
import html
import unicodedata
from typing import List, Optional
import contractions


class TextPreprocessorInterface(ABC):
    @abstractmethod
    def preprocess(self, text: str) -> str:
        pass


class TextNormalizer(TextPreprocessorInterface):
    def preprocess(self, text: str) -> str:
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


class HTMLCleaner(TextPreprocessorInterface):
    def preprocess(self, text: str) -> str:
        """Clean HTML entities and tags"""
        # Decode HTML entities
        text = html.unescape(text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        return text


class URLCleaner(TextPreprocessorInterface):
    def preprocess(self, text: str) -> str:
        """Remove or normalize URLs"""
        url_pattern = r"https?://\S+|www\.\S+"
        text = re.sub(url_pattern, "[URL]", text)
        return text


class SpecialCharacterCleaner(TextPreprocessorInterface):
    def __init__(self, preserve_chars: Optional[str] = None):
        self.preserve_chars = set(preserve_chars) if preserve_chars else set()

    def preprocess(self, text: str) -> str:
        """Remove special characters while preserving specified ones"""
        pattern = f'[^a-zA-Z0-9\s{re.escape("".join(self.preserve_chars))}]'
        text = re.sub(pattern, " ", text)
        return text


class WhitespaceCleaner(TextPreprocessorInterface):
    def preprocess(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        # Remove leading and trailing whitespace
        text = text.strip()
        return text


class ContractionExpander(TextPreprocessorInterface):
    def preprocess(self, text: str) -> str:
        """Expand contractions (e.g., "don't" to "do not")"""
        return contractions.fix(text)


class TextPreprocessor:
    def __init__(
        self,
        steps: Optional[List[TextPreprocessorInterface]] = None,
    ):
        self.steps = steps or self._get_default_steps()

    def _get_default_steps(self) -> List[TextPreprocessorInterface]:
        """Get default preprocessing steps"""
        return [
            HTMLCleaner(),
            URLCleaner(),
            ContractionExpander(),
            TextNormalizer(),
            SpecialCharacterCleaner(preserve_chars=".,!?"),
            WhitespaceCleaner(),
        ]

    def add_step(self, step: TextPreprocessorInterface) -> None:
        """Add a new preprocessing step"""
        self.steps.append(step)

    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to the input text"""
        for step in self.steps:
            text = step.preprocess(text)
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts"""
        return [self.preprocess(text) for text in texts]


def get_cleaned_text(raw_text):
    """Apply preprocessing step on raw text and return cleaned text"""
    preprocessor = TextPreprocessor(
        [
            HTMLCleaner(),
            URLCleaner(),
            TextNormalizer(),
            WhitespaceCleaner(),
        ]
    )
    clean_text = preprocessor.preprocess(raw_text)

    return clean_text
