"""
Streamlit Frontend for CodeMentor-LLM
Simple single page coding assistant interface.
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://Abdulmoiz123-codementor-llm-api.hf.space")

# Page config
st.set_page_config(
    page_title="CodeMentor-LLM",
    page_icon="💻",
    layout="centered"
)

# Title
st.title("💻 CodeMentor-LLM")
st.markdown("A fine-tuned coding assistant powered by Llama-3.2-3B-Instruct (SFT + DPO)")
st.divider()

# Input
prompt = st.text_area(
    label="Ask a coding question:",
    placeholder="e.g. Write a Python function to reverse a string.",
    height=120
)

# Submit button
if st.button("Generate Response", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a coding question.")
    else:
        with st.spinner("Generating response..."):
            try:
                response = requests.post(
                    f"{API_URL}/generate",
                    json={
                        "prompt": prompt,
                        "max_new_tokens": 512
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    if data["success"]:
                        st.divider()
                        st.markdown("### Response")
                        st.markdown(data["response"])
                        st.caption(f"Latency: {data['latency_ms']:.2f} ms")
                    else:
                        st.error(f"Error: {data['response']}")
                else:
                    st.error(f"API Error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the backend is running.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# Footer
st.divider()
st.caption("CodeMentor-LLM — Fine-tuned Llama-3.2-3B-Instruct | SFT + DPO Pipeline")