# Phase 02 — Environment & Hardware Setup

## System Configuration
- OS: Windows 11
- Python Version: 3.11.9
- Virtual Environment: venv

## Project Structure
- Backend: backend/
- Frontend: frontend/
- Notebooks: notebooks/
- Data: data/
- Docs: docs/

## Dependencies

### Backend (backend/requirements.txt)
- torch==2.6.0
- transformers==4.49.0
- peft==0.14.0
- trl==0.15.2
- bitsandbytes==0.45.3
- accelerate==1.5.1
- datasets==3.3.2
- evaluate==0.4.2
- huggingface-hub==0.29.3
- safetensors==0.5.3
- rouge-score==0.1.2
- bert-score==0.3.13
- wandb
- fastapi
- uvicorn
- pydantic
- sqlalchemy
- python-dotenv
- numpy==1.26.4
- pandas==2.2.3
- matplotlib==3.10.1

### Frontend (frontend/requirements.txt)
- streamlit==1.35.0
- requests==2.32.0
- python-dotenv==1.0.1

## Configuration Files
- backend/config.yaml — all model, training, inference configs
- .env — environment variables (gitignored)
- .env.example — template for environment variables

## GPU Training
- Local machine: no GPU — used for non-GPU phases only
- Google Colab: T4/A100 — used for training phases (9–16)

## Notes
- bitsandbytes requires Linux/Colab for GPU quantization
- All GPU phases will be run on Google Colab
- Local VS Code used for data, API, frontend, testing phases