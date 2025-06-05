from fastapi.testclient import TestClient
import types
import main
import pytest

# disable startup DB seed
if main.seed_owner in main.app.router.on_startup:
    main.app.router.on_startup.remove(main.seed_owner)

@pytest.fixture
def client(monkeypatch):
    def fake_verify(username: str, password: str):
        if username == "user" and password == "pass":
            return {"username": username, "role": "user"}
        return None
    monkeypatch.setattr(main, "verify_password", fake_verify)
    with TestClient(main.app) as c:
        yield c

def test_login_page(client):
    resp = client.get("/login")
    assert resp.status_code == 200

def test_login_success(client):
    resp = client.post("/login", data={"username": "user", "password": "pass"}, allow_redirects=False)
    assert resp.status_code == 303

def test_login_failure(client):
    resp = client.post("/login", data={"username": "bad", "password": "wrong"})
    assert resp.status_code == 401
