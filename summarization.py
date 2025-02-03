from transformers import AutoModelWithLMHead, AutoTokenizer
from prompts.headline_generator import prompt
import ollama
import re


class Summarizer:
    def __init__(self):
        pass

    def summarize(self, text):
        pass


class FlanT5(Summarizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mrm8488/t5-base-finetuned-summarize-news"
        )
        self.model = AutoModelWithLMHead.from_pretrained(
            "mrm8488/t5-base-finetuned-summarize-news"
        )
        self.MAX_TOKEN_LEN = 512

    def summarize(self, text):
        max_length = 150
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
        )

        # TODO: cater greater than max token length
        # print(len(input_ids[0]))
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=2,
            max_length=max_length,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )

        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]

        return preds[0]


class OllamaLLM(Summarizer):
    def __init__(self, model):
        self.model = model

    def extract_output(self, text):
        reason = " ".join(re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL))
        print(f"reason: {reason}")
        headline = re.split(r"</think>", text)[-1].strip()
        return reason, headline

    def summarize(self, news):
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": f"News detail: ```{news}``` Headline: ",
                    },
                ],
            )
            headline = response["message"]["content"]
            # Remove think tags from the generated headline to avoid repeating content
            reason, headline = self.extract_output(headline) if headline else headline
            return reason, headline

        except Exception as e:
            return f"{self.model} failed to generate Headline due to {e}"
