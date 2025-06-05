import asyncio
import types
import pytest
from starlette.requests import Request
import main

# disable startup DB seed
if main.seed_owner in main.app.router.on_startup:
    main.app.router.on_startup.remove(main.seed_owner)

@pytest.fixture
def fake_verify(monkeypatch):
    def fake(username: str, password: str):
        if username == "user" and password == "pass":
            return {"username": username, "role": "user"}
        return None
    monkeypatch.setattr(main, "verify_password", fake)
    main.signer = types.SimpleNamespace(dumps=lambda obj: "tok")

def make_request() -> Request:
    return Request({"type": "http", "headers": []})

def test_login_page(fake_verify):
    resp = main.login_form(make_request())
    assert resp.status_code == 200

def test_login_success(fake_verify):
    resp = asyncio.run(main.login_post(make_request(), username="user", password="pass"))
    assert resp.status_code == 303

def test_login_failure(fake_verify):
    resp = asyncio.run(main.login_post(make_request(), username="bad", password="wrong"))
    assert resp.status_code == 401
