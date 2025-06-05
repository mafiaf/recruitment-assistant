import types
import pytest
from starlette.requests import Request
from fastapi.responses import HTMLResponse
import main

# turn off the optional DB seed that runs at startup, if present
if main.seed_owner in main.app.router.on_startup:
    main.app.router.on_startup.remove(main.seed_owner)


@pytest.fixture
def make_request(monkeypatch):
    """Provide a fresh Request plus all required stubs for each test."""

    # fake template renderer so login_form works without Jinja2
    async def fake_render(
        request,
        template_name,
        ctx=None,
        page_title=None,
        status_code=200,
        active=None,
    ):
        return HTMLResponse("OK", status_code=status_code)

    # fake password check
    async def fake_verify(username: str, password: str):
        if username == "user" and password == "pass":
            return {"username": username, "role": "user"}
        return None

    # patch the bits in main that the handlers rely on
    monkeypatch.setattr(main, "render", fake_render, raising=False)
    monkeypatch.setattr(main, "verify_password", fake_verify)
    monkeypatch.setattr(main, "set_session", lambda resp, user: None, raising=False)
    main.signer = types.SimpleNamespace(dumps=lambda obj: "tok")

    # helper to create a minimal Starlette Request object
    return lambda: Request({"type": "http", "headers": []})


@pytest.mark.asyncio
async def test_login_page(make_request):
    resp = await main.login_form(make_request())
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_login_success(make_request):
    resp = await main.login_post(make_request(), username="user", password="pass")
    assert resp.status_code == 303


@pytest.mark.asyncio
async def test_login_failure(make_request):
    resp = await main.login_post(make_request(), username="bad", password="wrong")
    assert resp.status_code == 401
