# Recruitment Assistant

This FastAPI project provides recruitment-related features such as chat and resume management.

## Setup

Create a `.env.development` file in the project root with the following keys:

```bash
MONGO_URI=mongodb://localhost:27017
OPENAI_API_KEY=<your-openai-key>
SESSION_SECRET=<long-random-string>
```

When the application starts it tries to connect to MongoDB. If the connection
is successful you will see `mongo_utils ready` in the log. If the connection
fails the server switches to **NO‑DB** mode so you can still test other
features.

## Project History

The `/projects` page lists previous project descriptions along with their ranking tables, helping track hiring decisions.

## Running Tests

1. Install Python dependencies (requires network access):

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio
```

2. Execute the test suite with:

```bash
pytest
```

The tests use FastAPI's `TestClient` and check the login flow.

## Session Cookies

Session cookies are signed using `itsdangerous`. When `ENV` is set to
`production`, cookies are marked as `Secure` so browsers only send them over
HTTPS.
