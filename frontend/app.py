"""
Streamlit Frontend for CodeMentor-LLM
Simple single page coding assistant interface.
"""

import streamlit as st
import requests
import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

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
        def stream_response():
            with requests.post(
                f"{API_URL}/generate/stream",
                json={"prompt": prompt, "max_new_tokens": 512},
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        return

                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    if "token" in event:
                        yield event["token"]

        try:
            st.divider()
            st.markdown("### Response")
            st.write_stream(stream_response())
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the backend is running.")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# Footer
st.divider()
st.caption("CodeMentor-LLM — Fine-tuned Llama-3.2-3B-Instruct | SFT + DPO Pipeline")

# Admin logs
if st.checkbox("Show inference logs (admin)"):
    try:
        r = requests.get(f"{API_URL}/logs?limit=50", timeout=30)
        if r.status_code == 200:
            data = r.json()
            st.markdown(f"**Total logs: {data['total']}**")
            if data["logs"]:
                df = pd.DataFrame(data["logs"])
                df = df[["id", "prompt", "response", "latency_ms", "timestamp", "success"]]
                st.dataframe(df)
            else:
                st.info("No logs yet.")
    except Exception as e:
        st.error(f"Could not fetch logs: {e}")
