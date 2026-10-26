"""
Modal deployment for CodeMentor-LLM
Serves fine-tuned Llama-3.2-3B-Instruct on serverless GPU.
"""

import modal
import traceback

app = modal.App("codementor-llm")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.49.0",
        "peft==0.14.0",
        "bitsandbytes==0.45.3",
        "accelerate==1.5.1",
        "torch==2.6.0",
        "huggingface-hub",
        "safetensors",
        "fastapi[standard]",
    )
)

MERGED_MODEL_ID = "Abdulmoiz123/codementor-llm-merged"

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Answer coding questions clearly and concisely with working code examples."
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=120,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class CodeMentorModel:

    @modal.enter()
    def load_model(self):
        """Load the model once when the container starts."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        print(f"Loading model {MERGED_MODEL_ID}...")

        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_ID)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                MERGED_MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.model.eval()

            print("Model loaded successfully")
        except Exception:
            print("Model startup failed:")
            traceback.print_exc()
            raise

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 512) -> dict:
        """Generate a response for a coding prompt."""
        import time
        import torch

        if not prompt or not prompt.strip():
            return {
                "response": "Input cannot be empty",
                "latency_ms": 0,
                "success": False,
            }

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to("cuda")

            start_time = time.time()

            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            latency_ms = (time.time() - start_time) * 1000

            response = self.tokenizer.decode(
                outputs[0][inputs.shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            return {
                "response": response,
                "latency_ms": round(latency_ms, 2),
                "success": True,
            }

        except Exception as e:
            print("Generation failed:")
            traceback.print_exc()
            return {
                "response": f"Error: {str(e)}",
                "latency_ms": 0,
                "success": False,
            }

    @modal.method()
    def generate_stream(self, prompt: str, max_new_tokens: int = 512):
        """Yield generated text as it becomes available."""
        import threading
        import torch
        from transformers import TextIteratorStreamer

        if not prompt or not prompt.strip():
            raise ValueError("Input cannot be empty")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        errors = []

        def run_generation():
            try:
                with torch.inference_mode():
                    self.model.generate(
                        inputs,
                        streamer=streamer,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            except Exception as exc:
                errors.append(exc)
                streamer.end()

        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()
        for text_chunk in streamer:
            if text_chunk:
                yield text_chunk
        worker.join()
        if errors:
            raise errors[0]


model = CodeMentorModel()


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_endpoint(item: dict) -> dict:
    # Match the existing FastAPI schema's 1024-token maximum for longer answers.
    try:
        requested_tokens = int(item.get("max_new_tokens", 512))
    except (TypeError, ValueError):
        requested_tokens = 512

    max_new_tokens = max(1, min(requested_tokens, 1024))

    if item.get("stream"):
        import json
        from fastapi.responses import StreamingResponse

        def event_stream():
            try:
                for text_chunk in model.generate_stream.remote_gen(
                    item.get("prompt", ""), max_new_tokens
                ):
                    yield f"data: {json.dumps({'token': text_chunk})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return model.generate.remote(
        item.get("prompt", ""),
        max_new_tokens,
    )
