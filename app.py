import streamlit as st
from summarization import FlanT5, OllamaLLM
import time


class HeadlineGenerator:
    def __init__(self):
        # print(f"st.session_state: {st.session_state}")
        if "model" not in st.session_state:
            s = time.time()
            # st.session_state["model"] = FlanT5()
            st.session_state["model"] = OllamaLLM("deepseek-r1:1.5b")
            e = time.time()
            print(f"Summarizer loaded in {e - s:.2f} seconds.")
        else:
            print("Summarizer already loaded.")

    # Streamlit app
    def run(self):
        st.title("News Headline Generator")

        # Inputself box for user to enter text
        input_text = st.text_area("Enter the News Article", height=200)

        # Button to generate headline
        if st.button("Generate Headline"):
            if input_text:
                s = time.time()
                # Placeholder for headline generation logic
                _, headline = st.session_state["model"].summarize(input_text)
                e = time.time()
                print(f"Headline generated in {e - s:.2f} seconds.")
                st.subheader("Generated Headline:")
                st.write(headline)
            else:
                st.warning("Please enter some text to generate a headline.")


if __name__ == "__main__":
    HeadlineGenerator().run()
