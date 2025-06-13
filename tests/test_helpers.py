import re
import types
import pytest
import main


class DummyRequest:
    def __init__(self, path="/"):
        self.url = types.SimpleNamespace(path=path)
        self.cookies = {}

def test_guess_name_priority():
    # explicit name has highest priority
    assert main.guess_name("Alice", "resume.pdf", "") == "Alice"


def test_guess_name_from_filename():
    # fallback to file name when no explicit name
    assert main.guess_name("", "John Doe.pdf", "") == "John Doe"


def test_guess_name_from_text():
    # fallback to first non-empty text line
    assert main.guess_name("", "", "\nFirst Line\nSecond") == "First Line"


def test_guess_name_default():
    # final fallback when everything empty
    assert main.guess_name("", "", "") == "Unnamed résumé"


def test_slugify_basic():
    assert main.slugify("Hello World!") == "hello-world"


def test_slugify_truncate():
    long = "a" * 60
    assert main.slugify(long) == "a" * 48


def test_slugify_uuid(monkeypatch):
    # all punctuation triggers uuid fallback
    res = main.slugify("!!!")
    assert re.fullmatch(r"[0-9a-f-]{36}", res)


@pytest.mark.asyncio
async def test_match_project_logic_ok(monkeypatch):
    monkeypatch.setattr(main, "embed_text", lambda text: [0.0] * 1536)
    result = await main.match_project_logic("desc")
    assert result == ""


@pytest.mark.asyncio
async def test_match_project_logic_error(monkeypatch):
    monkeypatch.setattr(main, "embed_text", lambda text: None)
    result = await main.match_project_logic("desc")
    assert "embed project description" in result


def test_extract_years_requirement():
    assert main.extract_years_requirement("need 5 years of experience") == 5
    assert main.extract_years_requirement("minimum 3 years experience") == 3
    assert main.extract_years_requirement("no numbers") is None



@pytest.mark.asyncio
async def test_match_project_score_explanation(monkeypatch):
    table = (
        "| Candidate | Fit % | Why Fit? | Improve |\n"
        "|:--|:--:|:--|:--|\n"
        "| Alice | 90 | ok | - |\n"
        "| Bob | 80 | ok | - |"
    )
    fake_openai = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=table))]
                )
            )
        )
    )
    monkeypatch.setattr(main, "openai", fake_openai)
    monkeypatch.setattr(main, "embed_text", lambda text: [0.0] * 1536)

    class DummyMatch:
        def __init__(self, name, years):
            self.id = name
            self.values = [0.0] * 1536
            self.metadata = {"name": name, "text": "resume", "years": years}

    class DummyIndex:
        def query(self, *a, **k):
            return types.SimpleNamespace(matches=[
                DummyMatch("Alice", 2),
                DummyMatch("Bob", 4),
            ])

    monkeypatch.setattr(main, "index", DummyIndex())
    monkeypatch.setattr(main, "add_project_history", lambda *a, **k: None)
    async def _req_login():
        return {"username": "u"}
    async def _cur_user(*a, **k):
        return {"username": "u"}
    monkeypatch.setattr(main, "require_login", _req_login)
    monkeypatch.setattr(main, "get_current_user", _cur_user)

    resp = await main.match_project(
        DummyRequest("/match"),
        description="need 3 years", file=None, candidate_ids=None
    )
    html = resp.body.decode()
    assert "<table" in html


@pytest.mark.asyncio
async def test_match_project_dutch_header(monkeypatch):
    table = (
        "| Kandidaat | Fit % | Waarom geschikt? | Verbeter |\n"
        "|:--|:--:|:--|:--|\n"
        "| Alice | 90 | ok | - |\n"
        "| Bob | 80 | ok | - |"
    )
    fake_openai = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=table))]
                )
            )
        )
    )
    monkeypatch.setattr(main, "openai", fake_openai)
    monkeypatch.setattr(main, "embed_text", lambda text: [0.0] * 1536)

    class DummyMatch:
        def __init__(self, name):
            self.id = name
            self.values = [0.0] * 1536
            self.metadata = {"name": name, "text": "resume"}

    class DummyIndex:
        def query(self, *a, **k):
            return types.SimpleNamespace(matches=[
                DummyMatch("Alice"),
                DummyMatch("Bob"),
            ])

    monkeypatch.setattr(main, "index", DummyIndex())
    monkeypatch.setattr(main, "add_project_history", lambda *a, **k: None)
    async def _req_login():
        return {"username": "u"}
    async def _cur_user(*a, **k):
        return {"username": "u"}
    monkeypatch.setattr(main, "require_login", _req_login)
    monkeypatch.setattr(main, "get_current_user", _cur_user)

    resp = await main.match_project(
        DummyRequest("/match"),
        description="project", file=None, candidate_ids=None
    )
    html = resp.body.decode()
    assert html.count("<tr>") == 3

