import types
import pytest
from fastapi.responses import HTMLResponse
import main

# disable startup DB seed
if main.seed_owner in main.app.router.on_startup:
    main.app.router.on_startup.remove(main.seed_owner)

class DummyRequest:
    def __init__(self, path="/"):
        self.url = types.SimpleNamespace(path=path)
        self.cookies = {}

@pytest.fixture
def dummy(monkeypatch):
    def fake_render(request, template_name, ctx=None, page_title=None, status_code=200, active=None):
        return HTMLResponse("OK", status_code=status_code)

    def fake_verify(username: str, password: str):
        if username == "user" and password == "pass":
            return {"username": username, "role": "user"}
        return None

    monkeypatch.setattr(main, "render", fake_render)
    monkeypatch.setattr(main, "verify_password", fake_verify)
    monkeypatch.setattr(main, "set_session", lambda resp, user: None)
    return DummyRequest

def test_login_page(dummy):
    resp = main.login_form(dummy("/login"))
    assert resp.status_code == 200

@pytest.mark.asyncio
async def test_login_success(dummy):
    resp = await main.login_post(dummy("/login"), username="user", password="pass")
    assert resp.status_code == 303

@pytest.mark.asyncio
async def test_login_failure(dummy):
    resp = await main.login_post(dummy("/login"), username="bad", password="wrong")
    assert resp.status_code == 401
