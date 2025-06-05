import re
import pytest
import main


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
