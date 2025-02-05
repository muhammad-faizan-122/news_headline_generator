from transformers import AutoModelWithLMHead, AutoTokenizer


class Summarizer:
    def __init__(self):
        pass

    def summarize(self, text: str) -> str:
        pass


class FlanT5(Summarizer):
    def __init__(
        self,
        model="mrm8488/t5-base-finetuned-summarize-news",
        inp_token_size=512,
        out_token_size=150,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelWithLMHead.from_pretrained(model)
        self.inp_token_size = inp_token_size
        self.out_token_size = out_token_size

    def summarize(self, text):
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=True
        )

        # TODO: cater greater than max token length
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=2,
            max_length=self.out_token_size,
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
        return preds
