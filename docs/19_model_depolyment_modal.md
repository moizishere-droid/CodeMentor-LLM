## Model Deployment

After fine-tuning the LLM using SFT and DPO, the next step was deploying the model so users could interact with it through an API.

The deployment was done using [Modal](https://moizishere-droid--codementor-llm-generate-endpoint.modal.run), a serverless GPU platform that simplifies AI model hosting without manually managing infrastructure like Docker, Kubernetes, or GPU servers.

---

# Why Modal Was Used

Modal was selected because it:

* provides cloud GPUs on demand
* supports serverless deployment
* automatically manages scaling
* reduces DevOps complexity
* allows direct Python-based deployment

This made deployment faster and easier for serving the fine-tuned LLM online.

---

# Why A10G GPU Was Used

The deployment used an NVIDIA A10G GPU because:

* it provides 24 GB VRAM
* supports efficient LLM inference
* works well with QLoRA and 4-bit quantized models
* balances performance and cost

---

# Why Quantization Was Used

The model was loaded in 4-bit quantized mode using bitsandbytes.

This was done to:

* reduce GPU memory usage
* improve inference efficiency
* allow large LLM deployment on smaller GPUs

NF4 quantization and bfloat16 computation were used for better numerical stability and performance.

---

# How the Deployment Works

The deployment workflow is:

```text id="gl1rtg"
User Request
      ↓
FastAPI Endpoint
      ↓
Modal GPU Container
      ↓
Load Fine-Tuned LLM
      ↓
Generate Response
      ↓
Return JSON Output
```

---

# Internal Working

When the container starts:

* the tokenizer and fine-tuned model are loaded from Hugging Face
* the model is placed on GPU memory
* quantization settings are applied

During inference:

* user prompts are converted into tokens
* the LLM generates new tokens autoregressively
* generated tokens are decoded into readable text
* the response is returned through the API

---

# Why FastAPI Was Used

FastAPI was used because it:

* creates lightweight REST APIs
* is fast and asynchronous
* integrates easily with AI systems
* is commonly used in production ML deployment

---

# Why Serverless Deployment Was Important

Serverless deployment helps:

* avoid idle GPU costs
* automatically scale resources
* simplify infrastructure management
* deploy AI systems quickly

---

# Short Final Summary

The fine-tuned LLM was deployed using Modal with an A10G GPU and 4-bit quantization. FastAPI endpoints were used to expose the model as an online inference API. Quantization reduced VRAM usage, while Modal handled GPU infrastructure, scaling, and deployment automatically.

# Source file
backend/src/modal.py