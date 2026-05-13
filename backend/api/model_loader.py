"""
Model Loader for CodeMentor-LLM API
Calls HuggingFace Inference API instead of loading model locally.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_URL = "https://api-inference.huggingface.co/models/Abdulmoiz123/codementor-llm-merged"

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Answer coding questions clearly and concisely with working code examples."
)


def load_model():
    """No local model loading needed — using HF Inference API."""
    print("Using HuggingFace Inference API for model serving")
    print(f"Model: {MODEL_URL}")


def generate_response(prompt: str, max_new_tokens: int = 512) -> dict:
    """
    Generate response using HuggingFace Inference API.

    Args:
        prompt        : user coding question
        max_new_tokens: maximum tokens to generate

    Returns:
        dict with response, latency_ms, success
    """
    if not prompt or not prompt.strip():
        return {
            "response": "Input cannot be empty",
            "latency_ms": 0,
            "success": False
        }

    # Format prompt with system prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
            "return_full_text": False,
        }
    }

    try:
        start_time = time.time()
        response = requests.post(
            MODEL_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            generated_text = result[0]["generated_text"].strip()
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