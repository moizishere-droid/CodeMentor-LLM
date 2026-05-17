# Commads to push the frontend to Hugging Face 

cd frontend
git init
git add .
git commit -m "Initial frontend deployment"
git remote add space https://Abdulmoiz123:HF_TOKEN@huggingface.co/spaces/Abdulmoiz123/codementor-llm-app
git push space main --force

# Workflow of hf space

You push source code to HF Spaces git repo
        ↓
HF Spaces detects Dockerfile
        ↓
HF Spaces builds Docker image on their servers
        ↓
HF Spaces runs the container
        ↓
Your app is live

# You never touched Docker locally. HF Spaces did everything