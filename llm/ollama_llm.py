from llm.base import LLM
import ollama


class Ollama(LLM):
    def __init__(self, model):
        """input model name of same name as pulled using ollama command"""
        self.model = model

    def generate(self, system_prompt, user_prompt):
        """return output of ollama model result"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )
            output = response["message"]["content"]

            return output

        except Exception as e:
            raise f"{self.model} failed to generate Headline due to {e}"
