from transformers import AutoModelWithLMHead, AutoTokenizer


class FlanT5:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mrm8488/t5-base-finetuned-summarize-news"
        )
        self.model = AutoModelWithLMHead.from_pretrained(
            "mrm8488/t5-base-finetuned-summarize-news"
        )
        self.MAX_TOKEN_LEN = 512

    def summarize(self, text, max_length=150):
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


l_news = """The Indira Gandhi Institute of Medical Sciences (IGIMS) in Patna amended its marital declaration form on Thursday, replacing the word ?virgin? with ?unmarried? after controversy.Until now, new recruits to the super-specialty medical institute in the state capital were required to declare if they were bachelors, widowers or virgins.IGIMS medical superintendent Dr Manish Mandal said institute director Dr NR Biswas held a meeting on Thursday morning before directing that the word ?virgin? on the marital declaration form be immediately replaced with ?unmarried?. Dr Biswas had just returned after a four-day leave of absence.Earlier, Bihar health minister Mangal Pandey had ended up redefining the very meaning of virginity in his attempts to justify the awkward phrasing of the question in the form. Following a public furore over the document on Wednesday, the minister told news channels that there was nothing wrong with using the word ?virgin? because it simply meant ?kanya? or ?kunwari? ? which means an unmarried girl.Pandey had joined the cabinet just three days ago.Sources said the chief minister?s office had also taken cognizance of the issue, and asked for a copy of the form. It had even asked why the question was introduced in the first place.In its response, the management of the autonomous super-specialty health facility had clarified on Wednesday that it was in adherence to the central civil services rules followed by the All India Institute of Medical Sciences in New Delhi.The previous version of the marital declaration form, which purportedly asked new recruits if they were virgins.\r\n\t\t\t\t\t\t\t(HT Photo)\r\n\t\t\t\t\t\tThe marital declaration form had been in existence since the inception of the institute in 1983. Some officials blamed the faux pas on poor translation on the part of individuals who drafted the document.?The word ?virgin? mentioned on the form had nothing to do with the virginity of any employee. It only sought to know the employees? marital status, so their dues could be settled on the basis of their declaration in the event of death while in service,? said Dr Mandal."""
