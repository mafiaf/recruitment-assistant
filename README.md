# Recruitment Assistant

This FastAPI project provides recruitment-related features such as chat and resume management.

## Project History

The `/projects` page lists previous project descriptions along with their ranking tables, helping track hiring decisions.

## Environment Variables

Create a `.env.development` or `.env.production` file based on `.env.example` and set the following variables:

```bash
MONGO_URI=...       # MongoDB connection string
MONGO_DB_NAME=...   # database name
OPENAI_API_KEY=...
SESSION_SECRET=...
OWNER_USER=...
OWNER_PASS=...
```

Set `ENV=production` when deploying so the application loads `.env.production`. Any other value results in loading `.env.development`.

## Running Tests

1. Install Python dependencies (requires network access):

```bash
pip install -r requirements.txt
pip install pytest
```

2. Execute the test suite with:

```bash
pytest
```

The tests use FastAPI's `TestClient` and check the login flow.

## Running the App

After setting up your environment variables, start the API with:

```bash
uvicorn main:app
```

For local development you may add the `--reload` flag to enable auto-reload.
