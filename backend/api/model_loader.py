"""
Model Loader for CodeMentor-LLM API
Calls Modal serverless GPU endpoint serving
the fine-tuned Llama-3.2-3B-Instruct merged model.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

MODAL_ENDPOINT = os.getenv(
    "MODAL_ENDPOINT",
    "https://moizishere-droid--codementor-llm-generate-endpoint.modal.run"
)

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Answer coding questions clearly and concisely with working code examples."
)

def load_model():
    """No local model loading — using Modal serverless GPU."""
    print("Using Modal serverless GPU endpoint")
    print(f"Endpoint: {MODAL_ENDPOINT}")


def generate_response(prompt: str, max_new_tokens: int = 512) -> dict:
    """
    Generate response using Modal serverless GPU endpoint.
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

    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens
    }

    try:
        start_time = time.time()
        response = requests.post(
            MODAL_ENDPOINT,
            json=payload,
            timeout=120
        )
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            return {
                "response": result["response"],
                "latency_ms": round(latency_ms, 2),
                "success": result["success"]
            }
        else:
            return {
                "response": f"API Error: {response.status_code}",
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