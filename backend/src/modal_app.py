"""
Modal deployment for CodeMentor-LLM
Serves fine-tuned Llama-3.2-3B-Instruct on serverless GPU.
"""

'''
I used Modal to deploy my fine-tuned LLM on a cloud GPU as a scalable API service. 
Modal simplified infrastructure management, GPU allocation, and deployment, while 
4-bit quantization reduced memory usage and improved inference efficiency.
'''

import modal

# Define Modal app
app = modal.App("codementor-llm")

# Define container image with all dependencies
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

# Model ID
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
    min_containers=1
)
class CodeMentorModel:

    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading model {MERGED_MODEL_ID}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Model loaded successfully")

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 512) -> dict:
        """Generate response for a coding prompt."""
        import torch
        import time

        if not prompt or not prompt.strip():
            return {"response": "Input cannot be empty", "latency_ms": 0, "success": False}

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to("cuda")

            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                )
            latency_ms = (time.time() - start_time) * 1000

            response = self.tokenizer.decode(
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
                "response": f"Error: {str(e)}",
                "latency_ms": 0,
                "success": False
            }


model = CodeMentorModel()
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_endpoint(item: dict) -> dict:
    return model.generate.remote(
        item.get("prompt", ""),
        item.get("max_new_tokens", 512)
    )