from .preprocess import TextPreprocessor


class LLMTextPreprocessor(TextPreprocessor):
    def __init__(self):
        pass

    def get_cleaned_text(self, text: str) -> str:
        clean_text = self.norm_whitespaces(text)
        return clean_text
