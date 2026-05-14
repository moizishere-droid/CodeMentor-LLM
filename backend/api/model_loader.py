"""
Model Loader for CodeMentor-LLM API
Uses HuggingFace Inference Router for model serving.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Answer coding questions clearly and concisely with working code examples."
)


def load_model():
    """No local model loading — using HF Inference Router."""
    print("Using HuggingFace Inference Router for model serving")
    print(f"Model: {MODEL}")


def generate_response(prompt: str, max_new_tokens: int = 512) -> dict:
    """Generate response using HuggingFace Inference Router."""

    if not prompt or not prompt.strip():
        return {
            "response": "Input cannot be empty",
            "latency_ms": 0,
            "success": False
        }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        start_time = time.time()
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"].strip()
            return {
                "response": generated_text,
                "latency_ms": round(latency_ms, 2),
                "success": True
            }
        else:
            return {
                "response": f"API Error: {response.status_code} — {response.text}",
                "latency_ms": 0,
                "success": False
            }

    except requests.exceptions.Timeout:
        return {
            "response": "Request timed out. Please try again.",
            "latency_ms": 0,
            "success": False
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "latency_ms": 0,
            "success": False
        }