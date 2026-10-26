# Accounts and deployment

Login and Create account are required before generating answers or viewing history. Passwords use salted PBKDF2 hashing; sessions expire after seven days and logout revokes them. Log in again after reopening the browser to restore your latest answer. Request Log contains all your saved responses, with pagination. Earlier anonymous records are preserved but cannot safely be assigned to a person because they have no owner.

## Database persistence

The local database is backend/codementor.db. The combined app uses hf-combined/codementor.db. DATABASE_PATH overrides either location. Docker Compose sets DATABASE_PATH=/app/data/codementor.db and mounts ./data there. If you already have a database from an older deployment, back it up and move it to that configured location before starting; the schema migration retains existing records.

For Hugging Face, configure durable storage that supports SQLite file locking and set DATABASE_PATH to a database file on that storage, for example /data/codementor.db when durable /data storage is provisioned. The default Space filesystem is temporary: login does not protect it from resets. Verify SQLite compatibility before using a bucket-mounted filesystem. If durable SQLite-compatible storage is unavailable, host the API/database on a persistent disk and set the Space API_URL to its HTTPS URL. Never upload the live database or .env to GitHub or Hugging Face.

## PowerShell commands (run from project root)

Redeploy the Modal inference service:

```powershell
modal deploy backend/src/modal_app.py
```

Publish the updated combined frontend/API (upload triggers a Space rebuild):

```powershell
hf auth login
hf upload Abdulmoiz123/codementor-llm-combined ./hf-combined . --repo-type space --exclude ".git/*" "**/__pycache__/*" "*.pyc" "*.db" "*.db-*" ".env"
```

The hf-combined directory is also a separate Git repository. Commit its source changes explicitly before committing the parent repository reference:

```powershell
git -C hf-combined add api/auth.py api/database.py api/main.py api/routes.py streamlit_app.py ui/index.html ui/app.css .gitignore
git -C hf-combined commit -m "Add accounts, private history, and navigation fixes"
git -C hf-combined push space HEAD:main
git add backend/api/auth.py backend/api/database.py backend/api/main.py backend/api/routes.py backend/tests/test_accounts.py backend/tests/test_api.py frontend/app.py frontend/ui/index.html frontend/ui/app.css docker-compose.yml .gitignore docs/ACCOUNT_DEPLOYMENT.md hf-combined
git commit -m "Add accounts, persistent private history, and UI fixes"
git push origin HEAD
```

Use either hf upload or the nested repository push to publish the Space. The nested commit is still needed if you want the parent Git repository reference to record the updated combined source.

These changes do not require retraining or uploading model weights. If you separately retrain and merge weights, upload your actual merged-model folder with:

```powershell
hf upload Abdulmoiz123/codementor-llm-merged ./PATH_TO_MERGED_MODEL . --repo-type model
modal deploy backend/src/modal_app.py
```
