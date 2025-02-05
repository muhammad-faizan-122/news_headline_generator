from abc import ABC, abstractmethod


class LLM(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(system_prompt, user_prompt):
        """
        generate llm output
        system_prompt: System prompt message with instructions.
        user_prompt: Contain input text.
        """
        pass
