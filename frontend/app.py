"""
Gradio Frontend for CodeMentor-LLM
Simple chat interface for the coding assistant.
"""

import gradio as gr
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://Abdulmoiz123-codementor-llm-api.hf.space")

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Answer coding questions clearly and concisely with working code examples."
)


def generate_response(prompt: str) -> str:
    """Call FastAPI backend and return response."""
    if not prompt.strip():
        return "Please enter a coding question."

    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={"prompt": prompt, "max_new_tokens": 512},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                return f"{data['response']}\n\n*Latency: {data['latency_ms']:.0f}ms*"
            else:
                return f"Error: {data['response']}"
        else:
            return f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Cannot connect to API. Please try again later."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# Gradio interface
demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(
        label="Ask a coding question:",
        placeholder="e.g. Write a Python function to reverse a string.",
        lines=3
    ),
    outputs=gr.Markdown(label="Response:"),
    title="💻 CodeMentor-LLM",
    description="A fine-tuned coding assistant powered by Llama-3.2-3B-Instruct (SFT + DPO)",
    examples=[
        ["Write a Python function to reverse a string."],
        ["What is the difference between a list and a tuple in Python?"],
        ["Write a SQL query to find duplicate records in a table."],
        ["Explain what a decorator is in Python with an example."],
        ["Fix this code: myList = [1, 2, 3"],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()