# 💻 CodeMentor-LLM

A production-grade coding assistant built by fine-tuning **Llama-3.2-3B-Instruct** using a two-stage training pipeline (SFT + DPO). Ask any coding question and get accurate, concise answers with working code examples.

[![CI](https://github.com/moizishere-droid/CodeMentor-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/moizishere-droid/CodeMentor-LLM/actions/workflows/ci.yml)

---

## 🚀 Live Demo

| Component | URL |
|---|---|
| 🖥️ Frontend | [https://abdulmoiz123-codementor-llm-app.hf.space](https://abdulmoiz123-codementor-llm-app.hf.space) |
| ⚡ Backend API | [https://abdulmoiz123-codementor-llm-api.hf.space](https://abdulmoiz123-codementor-llm-api.hf.space) |
| 📖 API Docs | [https://abdulmoiz123-codementor-llm-api.hf.space/docs](https://abdulmoiz123-codementor-llm-api.hf.space/docs) |
| 🤗 Model | [Abdulmoiz123/codementor-llm-merged](https://huggingface.co/Abdulmoiz123/codementor-llm-merged) |

---

## 📌 Project Overview

CodeMentor-LLM is a **fresher-level production-grade LLM fine-tuning project** that demonstrates the complete MLOps pipeline from data collection to deployment.

### Problem Statement
Developers — especially beginners — constantly need help with:
- Understanding why their code is broken
- Getting clear explanations for coding concepts
- Writing working code snippets fast

### Solution
Fine-tune Llama-3.2-3B-Instruct on 5,000 high-quality coding instruction-response pairs using QLoRA + SFT + DPO alignment pipeline.

---

## 🏗️ Architecture
codementor-llm/
├── notebooks/          ← experiment notebooks (phases 3-22)
├── backend/
│   ├── src/            ← training + inference scripts
│   │   ├── config.py
│   │   ├── data_formatter.py
│   │   ├── data_cleaner.py
│   │   ├── train_sft.py
│   │   ├── train_dpo.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── merge_model.py
│   │   └── modal_app.py
│   │   └── config.yaml
│   ├── api/            ← FastAPI endpoints
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   ├── model_loader.py
│   │   └── database.py
│   │   └── model_loader_prod.py
│   ├── tests/          ← pytest test files
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py          ← Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── data/               ← datasets
├── docs/               ← phase documentation (24 phases)
├── docker-compose.yml
├── .env.example
└── README.md

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11
- GPU recommended for training (Google Colab T4)

### Installation

```bash
git clone https://github.com/moizishere-droid/CodeMentor-LLM.git
cd CodeMentor-LLM
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r backend/requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
# Fill in your tokens
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
MODAL_ENDPOINT=your_modal_endpoint
```

### Run Locally

```bash
# Start backend
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (new terminal)
cd frontend
streamlit run app.py
```

### Run with Docker

```bash
docker-compose up
```

---

## 🧪 Testing

```bash
cd backend
pytest tests/test_api.py -v
```

---

## 📈 Experiment Tracking

W&B Project: [codementor-llm](https://wandb.ai/abdulmoiz2004-2-institution-of-engineering-and-technology/codementor-llm)

- SFT Run: `sft-llama3-8b-codealapaca`
- DPO Run: `dpo-llama3-3b`

---

## 🤗 HuggingFace Assets

| Asset | Link |
|---|---|
| Merged Model | [codementor-llm-merged](https://huggingface.co/Abdulmoiz123/codementor-llm-merged) |
| SFT Adapter | [codementor-llm-sft](https://huggingface.co/Abdulmoiz123/codementor-llm-sft) |
| DPO Adapter | [codementor-llm-dpo](https://huggingface.co/Abdulmoiz123/codementor-llm-dpo) |
| Dataset Splits | [codementor-llm-splits](https://huggingface.co/datasets/Abdulmoiz123/codementor-llm-splits) |
| Preference Dataset | [codementor-llm-preference](https://huggingface.co/datasets/Abdulmoiz123/codementor-llm-preference) |

---

## 📚 Documentation

Each phase is documented in `docs/` folder:

| Phase | Document |
|---|---|
| 1 | Problem Framing |
| 2 | Environment Setup |
| 3 | Literature & Baseline Research |
| 4 | Dataset Collection |
| 5 | Data Formatting |
| 6 | Data Cleaning |
| 7 | EDA |
| 8 | Train/Val/Test Split |
| 9 | Quantization Config |
| 10 | LoRA Config |
| 11 | SFT Training |
| 12 | Experiment Tracking |
| 13 | SFT Evaluation |
| 14 | Preference Dataset |
| 15 | DPO Training |
| 16 | DPO Evaluation |
| 17 | Model Merging |
| 18 | Inference Pipeline |
| 19 | Modal Deployment |
| 20 | FastAPI REST API |
| 21 | Streamlit Frontend |
| 22 | Testing |
| 23 | CI/CD |
| 24 | Deployment |

---

## 👤 Author

**Abdul Moiz**
- GitHub: [@moizishere-droid](https://github.com/moizishere-droid)
- HuggingFace: [Abdulmoiz123](https://huggingface.co/Abdulmoiz123)

---

## 📄 License

Apache 2.0