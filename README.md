# Recruitment Assistant

This FastAPI project provides recruitment-related features such as chat and resume management.

## Project History

The `/projects` page lists previous project descriptions along with their ranking tables, helping track hiring decisions.

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
