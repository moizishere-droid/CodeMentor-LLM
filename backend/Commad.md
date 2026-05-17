# Commads to push the backend to Hugging Face 

cd backend
git init
git add .
git commit -m "Initial backend deployment"
git remote add space https://Abdulmoiz123:HF_TOKEN@huggingface.co/spaces/Abdulmoiz123/codementor-llm-api
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