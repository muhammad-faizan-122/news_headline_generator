from models.llm.ollama_llm import DeepSeekR1
from models.llm.prompts.headline_generator import system_prompt, user_prompt
from prepocessing.llm import LLMTextPreprocessor
import streamlit as st
import time
from logger import get_logger


class HeadlineGenerator:
    def __init__(self):
        st.session_state["preprocessor"] = LLMTextPreprocessor()
        st.session_state["logger"] = get_logger()
        if "model" not in st.session_state:
            s = time.time()
            st.session_state["model"] = DeepSeekR1()
            e = time.time()
            st.session_state["logger"].info(f"Model loaded in {e - s:.2f} seconds.")
        else:
            st.session_state["logger"].info("Model already loaded.")

    def run(self):
        st.title("News Headline Generator")
        """get the input news text."""
        news_text = st.text_area("**Enter the News**", height=200)

        if st.button("Generate Headline"):
            if news_text:
                s = time.time()
                # apply preprocess
                news_text = st.session_state["preprocessor"].get_cleaned_text(news_text)

                st.session_state["logger"].debug(f"preprocessed news text: {news_text}")
                st.session_state["logger"].debug(
                    f"LLM prompt: {system_prompt+user_prompt.format(news_text)}"
                )

                # llm inference
                llm_out = st.session_state["model"].generate(
                    system_prompt,
                    user_prompt.format(news_text),
                )

                # # postprocess
                _, headline = st.session_state["model"].extract_headline(llm_out)
                e = time.time()

                st.session_state["logger"].info(
                    f"Headline generated in {e - s:.2f} seconds."
                )
                st.subheader("Generated Headline:")
                st.write(headline)
            else:
                st.warning("Please enter some text to generate a headline.")


if __name__ == "__main__":
    HeadlineGenerator().run()
