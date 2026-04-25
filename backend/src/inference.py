"""
Inference Pipeline for CodeMentor-LLM
Loads merged fine-tuned model and generates responses
for coding questions.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODEL_ID, INFERENCE_CONFIG, SYSTEM_PROMPT

# Merged model ID
MERGED_MODEL_ID = "Abdulmoiz123/codementor-llm-merged"


def load_model_and_tokenizer(model_id: str = MERGED_MODEL_ID):
    """
    Load merged model and tokenizer.
    Args:
        model_id: HuggingFace Hub model ID
    Returns:
        tuple of model and tokenizer
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"Model loaded — {model.get_memory_footprint() / 1024**3:.2f} GB")

    return model, tokenizer


def validate_input(prompt: str, max_length: int = 2048) -> tuple[bool, str]:
    """
    Validate user input before inference
    Args:
        prompt: user input string
        max_length: maximum allowed character length
    Returns:
        tuple of (is_valid, error_message)
    """
    if not prompt:
        return False, "Input cannot be empty"
    if not isinstance(prompt, str):
        return False, "Input must be a string"
    if len(prompt.strip()) == 0:
        return False, "Input cannot be whitespace only"
    if len(prompt) > max_length:
        return False, f"Input too long — maximum {max_length} characters allowed"
    return True, ""


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = INFERENCE_CONFIG["max_new_tokens"],
    temperature: float = INFERENCE_CONFIG["temperature"],
    top_p: float = INFERENCE_CONFIG["top_p"],
) -> dict:
    """
    Generate response for a given coding prompt.
    Args:
        model: loaded model
        tokenizer: loaded tokenizer
        prompt: user coding question
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling parameter
    Returns:
        dict with response, latency_ms, and success flag
    """
    # Validate input
    is_valid, error_message = validate_input(prompt)
    if not is_valid:
        return {
            "response": error_message,
            "latency_ms": 0,
            "success": False
        }

    try:
        # Apply chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate response
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=INFERENCE_CONFIG["do_sample"],
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
            )
        latency_ms = (time.time() - start_time) * 1000

        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        return {
            "response": response,
            "latency_ms": round(latency_ms, 2),
            "success": True
        }

    except Exception as e:
        return {
            "response": f"Error generating response: {str(e)}",
            "latency_ms": 0,
            "success": False
        }


if __name__ == "__main__":
    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Test prompts
    test_prompts = [
        "Write a Python function to reverse a string.",
        "What is the difference between a list and a tuple?",
        "",  # edge case — empty input
        "Fix this code: myList = [1, 2, 3",  # debug
    ]

    for prompt in test_prompts:
        result = generate_response(model, tokenizer, prompt)
        print(f"\nPrompt  : {prompt}")
        print(f"Response: {result['response'][:200]}")
        print(f"Latency : {result['latency_ms']} ms")
        print(f"Success : {result['success']}")
        print("=" * 50)