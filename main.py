from __future__ import annotations

import io
import os
import re
import uuid
from types import SimpleNamespace
from typing import List, Optional
from urllib.parse import urlencode
from bson import ObjectId

import mammoth
import numpy as np
import openai
import pdfplumber
import PyPDF2
from docx import Document
from fastapi import Body, FastAPI, File, Form, Request, UploadFile, Depends, HTTPException, status, Cookie
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeTimedSerializer, BadSignature
from starlette.responses import RedirectResponse
from passlib.context import CryptContext
from datetime import datetime

from settings import settings
from logger import logger


# ── local helper modules ──────────────────────────────────────────────────────
from mongo_utils import (
    db,
    _users,
    update_resume,
    delete_resume_by_id,
    ENV,
    _guard,
    chat_find_one,
    chat_upsert,
    resumes_all,
    resumes_by_ids,
    resumes_collection,
    add_project_history,
    delete_project,
    update_project_description,
    ensure_indexes,
    resumes_count,
    resumes_page,
)
from pymongo import errors
from pinecone_utils import (
    add_resume_to_pinecone,
    embed_text,
    index,
    search_best_resumes,
)
from utils import sanitize_markdown
from schemas import ResumeUpload, ChatRequest



openai.api_key = settings.OPENAI_API_KEY

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.svg")


# ═════════════════════════════════════════════════════════════════════════════
# USERS
# ═════════════════════════════════════════════════════════════════════════════
# password‐hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# reuse your existing `db` from mongo_utils / main.py
users_coll = db["users"]

# default password used when resetting user accounts
DEFAULT_PASSWORD = settings.DEFAULT_PASS

# ── helpers ────────────────────────────────────────────────────────────────
async def render(request: Request,
           template_name: str,
           ctx: dict | None = None,
           page_title: str | None = None,
           status_code: int = 200,
           active: str | None = None) -> HTMLResponse:
           
    """
    Wrapper around TemplateResponse that always injects:
      * request (FastAPI requirement)
      * active      – current path for nav highlighting
      * page_title  – headline for <h1>
      * user        – current logged-in user (or None)
    """
    ctx = ctx or {}
    ctx.setdefault("request", request)
    ctx.setdefault("active",  active or request.url.path)
    ctx.setdefault("page_title", page_title or "")
    ctx.setdefault("user", await get_current_user(request.cookies.get(COOKIE_NAME)))
    return templates.TemplateResponse(template_name, ctx,
                                      status_code=status_code)

async def get_user(username: str):
    return await users_coll.find_one({"username": username})

async def create_user(username: str, password: str, role: str = "user"):
    await users_coll.insert_one(
        {
            "username": username,
            "hashed_password": pwd_context.hash(password),
            "role": role,
            "created": datetime.utcnow(),
        }
    )

async def verify_password(username: str, password: str):
    user = await get_user(username)
    if user and pwd_context.verify(password, user["hashed_password"]):
        return user
    return None

# ── signed-cookie session ──────────────────────────────────────────────────
SECRET_KEY = settings.SESSION_SECRET   # set a strong one!
signer = URLSafeTimedSerializer(SECRET_KEY, salt="auth-cookie")
COOKIE_NAME = "session"

def set_session(resp, user: dict):
    token = signer.dumps({"u": user["username"], "r": user["role"]})
    resp.set_cookie(
        COOKIE_NAME,
        token,
        max_age=8 * 3600,
        httponly=True,
        secure=(ENV == "production"),
        samesite="strict"
    )

def clear_session(resp):
    resp.delete_cookie(COOKIE_NAME)

async def get_current_user(session: str = Cookie(None)):
    if not session:
        return None
    try:
        data = signer.loads(session, max_age=8 * 3600)
        return await get_user(data["u"])
    except BadSignature:
        return None

async def require_login(user=Depends(get_current_user)):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_302_FOUND, headers={"Location": "/login"}
        )
    return user

async def require_owner(user=Depends(require_login)):
    if user["role"] != "owner":
        raise HTTPException(status_code=403, detail="Owners only")
    return user

@app.on_event("startup")
async def seed_owner() -> None:
    if await _guard("seed_owner"):
        logger.warning("Mongo not reachable – owner account not seeded")
        return

    await ensure_indexes()

    if await users_coll.count_documents({"role": "owner"}) == 0:
        await create_user(
            settings.OWNER_USER,
            settings.OWNER_PASS,
            role="owner",
        )
        logger.info("Created initial owner account")

# ═════════════════════════════════════════════════════════════════════════════
# PDF & DOCX extraction helpers
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Read text from a PDF, first via pdfplumber, fall‑back to PyPDF2."""
    pages: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes), strict=False) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
    except Exception:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(data: bytes) -> str:
    return mammoth.extract_raw_text(io.BytesIO(data)).value.strip()


def extract_text(data: bytes, filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if fn.endswith(".docx"):
        return extract_text_from_docx(data)
    return ""

# ═════════════════════════════════════════════════════════════════════════════
# Routes – basic pages
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user=Depends(require_login)):
    # Fetch recent resumes for the front page (best effort – works without DB)
    try:
        docs = await resumes_all()
    except Exception as e:  # pragma: no cover - DB might be down
        logger.warning("resumes_all() failed: %s", e)
        docs = []

    # sort newest first and limit to 5 entries
    docs.sort(key=lambda d: getattr(d.get("_id"), "generation_time", datetime.min),
              reverse=True)
    docs = docs[:5]

    resumes = []
    for d in docs:
        oid = d.get("_id")
        added = "—"
        if hasattr(oid, "generation_time"):
            added = oid.generation_time.strftime("%Y-%m-%d")

        resumes.append({
            "id": d.get("resume_id", "—"),
            "name": d.get("name", "Unknown"),
            "added": added,
        })

    return await render(request, "index.html", {"resumes": resumes}, page_title="Home")



# ═════════════════════════════════════════════════════════════════════════════
# upload résumé – saves to Pinecone + Mongo (if available)
# ═════════════════════════════════════════════════════════════════════════════

def guess_name(name_field: str, filename: str, text: str) -> str:
    """
    1) explicit <input name="name"> if the user filled it
    2) otherwise first ~40 chars of the file name (strip .pdf/.docx etc.)
    3) otherwise first non-empty line of the résumé text
    """
    n = name_field.strip()
    if n:
        return n

    stem = os.path.splitext(os.path.basename(filename))[0].strip()
    if stem:
        return stem[:40]

    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:60]
    return "Unnamed résumé"          # shouldn’t really happen


def slugify(val: str) -> str:
    return (
        re.sub(r"[^a-z0-9]+", "-", val.lower())
        .strip("-")[:48]              # keep it short
        or str(uuid.uuid4())          # absolute fallback
    )


def extract_years_requirement(text: str) -> int | None:
    """Return the first integer that appears before 'year' or 'years'."""
    m = re.search(r"(\d+)\s*\+?\s*(?=years?)", text, flags=re.I)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

# ── upload validation constants --------------------------------------------
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
# default max file size: 5 MB
MAX_FILE_SIZE = 5 * 1024 * 1024

# ── main upload handler --------------------------------------------------
@app.post("/upload_resume", response_class=HTMLResponse)
async def upload_resume(
    request: Request,
    file: List[UploadFile] | UploadFile | None = File(None),
    name: str | None = Form(None),
    text: str | None = Form(None),
    skills: str | None = Form(None),
    location: str | None = Form(None),
    years: int | None = Form(None),
    tags: str | None = Form(None),
    resume: ResumeUpload | None = Body(None),
    user=Depends(require_login),
):
    """Handle résumé uploads from JSON or multipart forms. Supports multiple files."""

    added_names: List[str] = []

    def parse_skills(val):
        if not val:
            return []
        if isinstance(val, list):
            return [s.strip() for s in val if s.strip()]
        return [s.strip() for s in str(val).split(',') if s.strip()]

    def parse_tags(val):
        if not val:
            return []
        if isinstance(val, list):
            return [t.strip() for t in val if t.strip()]
        return [t.strip() for t in str(val).split(',') if t.strip()]

    # ── JSON payload -------------------------------------------------------
    if resume is not None:
        name = resume.name or name or ""
        text = (resume.text or text or "").strip()
        skill_list = parse_skills(resume.skills)
        tag_list = parse_tags(resume.tags)
        loc_val = resume.location or location or ""
        years_val = resume.years if resume.years is not None else years

        if not text:
            return await render(
                request,
                "index.html",
                {"popup": "Résumé text is empty."},
                page_title="Home",
            )

        display_name = guess_name(name, "", text)
        resume_id = slugify(display_name)
        meta = {"name": display_name, "text": text}
        meta["file_type"] = "txt"
        if tag_list:
            meta["tags"] = tag_list
        if skill_list:
            meta["skills"] = skill_list
        if loc_val:
            meta["location"] = loc_val
        if years_val is not None:
            meta["years"] = years_val
        add_resume_to_pinecone(text, resume_id, meta, "resumes")
        doc = {"resume_id": resume_id, "name": display_name, "text": text, "file_type": "txt"}
        if tag_list:
            doc["tags"] = tag_list
        if skill_list:
            doc["skills"] = skill_list
        if loc_val:
            doc["location"] = loc_val
        if years_val is not None:
            doc["years"] = years_val
        try:
            await resumes_collection.insert_one(doc)
        except Exception as e:  # pragma: no cover
            logger.error("Mongo insert failed: %s", e)
        added_names.append(display_name)

    else:
        # ── multipart/form-data (possibly multiple files) -----------------
        files: List[UploadFile] = []
        if isinstance(file, list):
            files = file
        elif file is not None:
            files = [file]

        skill_list = parse_skills(skills)
        tag_list = parse_tags(tags)
        loc_val = location or ""
        years_val = years

        if not files and text:
            # fallback: text field without file
            display_name = guess_name(name or "", "", text)
            resume_id = slugify(display_name)
            meta = {"name": display_name, "text": text, "file_type": "txt"}
            if tag_list:
                meta["tags"] = tag_list
            if skill_list:
                meta["skills"] = skill_list
            if loc_val:
                meta["location"] = loc_val
            if years_val is not None:
                meta["years"] = years_val
            add_resume_to_pinecone(text, resume_id, meta, "resumes")
            doc = {"resume_id": resume_id, "name": display_name, "text": text, "file_type": "txt"}
            if tag_list:
                doc["tags"] = tag_list
            if skill_list:
                doc["skills"] = skill_list
            if loc_val:
                doc["location"] = loc_val
            if years_val is not None:
                doc["years"] = years_val
            try:
                await resumes_collection.insert_one(doc)
            except Exception as e:  # pragma: no cover
                logger.error("Mongo insert failed: %s", e)
            added_names.append(display_name)

        for f in files:
            filename = f.filename or ""
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue
            data = await f.read()
            if len(data) > MAX_FILE_SIZE:
                continue
            file_text = extract_text(data, filename)
            if not file_text.strip():
                continue
            display_name = guess_name(name or "", filename, file_text)
            resume_id = slugify(display_name)
            ftype = ext.lstrip('.') or 'file'
            meta = {"name": display_name, "text": file_text, "file_type": ftype}
            if tag_list:
                meta["tags"] = tag_list
            if skill_list:
                meta["skills"] = skill_list
            if loc_val:
                meta["location"] = loc_val
            if years_val is not None:
                meta["years"] = years_val
            add_resume_to_pinecone(file_text, resume_id, meta, "resumes")
            doc = {"resume_id": resume_id, "name": display_name, "text": file_text, "file_type": ftype}
            if tag_list:
                doc["tags"] = tag_list
            if skill_list:
                doc["skills"] = skill_list
            if loc_val:
                doc["location"] = loc_val
            if years_val is not None:
                doc["years"] = years_val
            try:
                await resumes_collection.insert_one(doc)
            except Exception as e:  # pragma: no cover
                logger.error("Mongo insert failed: %s", e)
            added_names.append(display_name)

    if not added_names:
        return await render(
            request,
            "index.html",
            {"popup": "No valid résumé uploaded."},
            page_title="Home",
        )

    msg = "Résumés added: " + ", ".join(added_names) if len(added_names) > 1 else f"Résumé added: {added_names[0]}"
    return await render(request, "index.html", {"popup": msg}, page_title="Home")


# ---------------------------------------------------------------------------
# Helper for the chat route (returns a short list, not a table)
# ---------------------------------------------------------------------------
async def match_project_logic(description: str,
                              candidate_ids: Optional[List[str]] = None) -> str:
    """Helper used by /chat route. We now let the front‑end’s /match_project
    endpoint render the full HTML table, so this function should **not** try
    to summarise anything.  
    By returning an empty string the assistant bubble stays blank and only the
    properly formatted table (inserted by JavaScript after the XHR call to
    `/match_project`) will appear in the chat history.
    """

    # We still embed once simply to validate the description – otherwise we
    # fall back with a brief warning so the user knows something is wrong.
    proj_vec = embed_text(description)
    if proj_vec is None:
        return "⚠️ Could not embed project description."

    # If everything is fine just return an empty string – the real rendering
    # is handled by the `/match_project` route and injected on the client.
    return ""

    matches.sort(key=lambda m: m.score, reverse=True)
    return "\n".join(f"* **{m.name}** – {m.score}%" for m in matches[:5])

# ---------------------------------------------------------------------------
# /match_project – returns the full HTML table fragment
# ---------------------------------------------------------------------------
@app.post("/match_project", response_class=HTMLResponse)
async def match_project(
    request: Request,
    user=Depends(require_login),
    description: str = Form(""),
    file: UploadFile = File(None),
    candidate_ids: Optional[List[str]] = Form(None),
):
    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    # 1️⃣ extract text --------------------------------------------------------
    if file and file.filename:
        data = await file.read()
        fn = file.filename.lower()
        if fn.endswith(".pdf"):
            description = extract_text_from_pdf(data)
        elif fn.endswith(".docx"):
            doc = Document(io.BytesIO(data))
            description = "\n".join(p.text for p in doc.paragraphs)

    if not description.strip():
        return templates.TemplateResponse(
            "index.html", {"request": request, "popup": "Project description is empty."}
        )

    proj_vec = np.asarray(embed_text(description))
    if proj_vec is None:
        return templates.TemplateResponse(
            "index.html", {"request": request, "popup": "Failed to embed project."}
        )

    # 2️⃣ choose candidate set -----------------------------------------------
    if candidate_ids:
        fetched = index.fetch(ids=candidate_ids, namespace="resumes").vectors
    else:
        fetched = {m.id: m for m in index.query(
            vector=proj_vec.tolist(),
            top_k=10,
            include_values=True,
            include_metadata=True,
            namespace="resumes",
        ).matches}

    # 3️⃣ score and keep best 5 ----------------------------------------------
    matches = []
    for cid, vec_obj in fetched.items():
        vals = vec_obj.values or []
        if len(vals) != 1536:
            continue
        vec = np.asarray(vals)
        sim = float(np.dot(vec, proj_vec) /
                    (np.linalg.norm(vec) * np.linalg.norm(proj_vec)))
        matches.append(
            SimpleNamespace(
                id=cid,
                metadata=vec_obj.metadata,
                sim_pct=round(sim*100,1)
            )
        )

    matches = sorted(matches, key=lambda m: m.sim_pct, reverse=True)[:5]
    if not matches:
        return templates.TemplateResponse(
            "index.html", {"request": request, "popup": "No candidates found."}
        )

    # 4️⃣ build GPT prompt ----------------------------------------------------
    expected_years = extract_years_requirement(description)
    snippets = []
    years_info = []
    for m in matches:
        tags_str = ", ".join(m.metadata.get('tags', []))
        tag_part = f" [tags: {tags_str}]" if tags_str else ""
        years = m.metadata.get('years')

        if expected_years and years is not None:
            years_part = f" ({years}/{expected_years} yrs)"
            years_info.append(f"{years}/{expected_years}")
        elif years is not None:
            years_part = f" ({years} yrs)"
            years_info.append(str(years))
        else:
            years_part = ""
            years_info.append("—" if expected_years else "—")

        snippet = (
            f"- **{m.metadata['name']}**{years_part}{tag_part}: "
            f"{m.metadata['text'].replace(chr(10),' ')[:300]}…"
        )
        snippets.append(snippet)
    candidates_block = "\n".join(snippets)

    header = (
        "| Candidate | Fit % | Why Fit? (≤90 w.) | Improve (≤45 w.) |\n"
        "|:--|:--:|:--|:--|\n"          # ← keep this on the *same* assignment
    )
    rows_md = "\n".join(
        f"| {m.metadata['name']} | {m.sim_pct} | | |" for m in matches
    )

    rubric = (
        "### Scoring rules\n"
        "* 90-100 % = near-perfect match of role, domain, experience\n"
        "* 70-89 %  = strong match, minor gaps\n"
        "* 40-69 %  = partial match, clear gaps\n"
        "* below 40 % = weak match\n\n"
        "The **Fit %** column already contains a cosine baseline. "
        "You may adjust it **±15 points max** if the résumé clearly warrants.  \n"
        "**For “Improve”** give **one concrete, resume-focused action**: "
        "what to **add**, **reword**, **reorder** or **strengthen** in their résumé "
        "(new bullet, certification, project, keywords) to boost their fit."
    )
    if expected_years:
        rubric += f"\n\nTarget experience: around {expected_years} years"

    prompt = (
        f"Project description (focus on role / domain / experience):\n"
        f"{description.replace(chr(10),' ')}\n\n"
        "Candidate snippets:\n"
        f"{candidates_block}\n\n"
        f"{rubric}\n\n"
        "Fill ONLY the blank cells in this Markdown table. "
        "Maintain exactly these 4 columns; no extra rows or commentary.\n\n"
        f"{header}{rows_md}"
    )

    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a rigorous recruitment assistant."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )
    table_md = resp.choices[0].message.content.strip()

    # 5️⃣ parse table ---------------------------------------------------------
    rows = []
    alignment_re = re.compile(r"\|\s*:?[-]{3,}")     # ← correct regex

    for ln in table_md.splitlines():
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        # skip header and alignment
        if ln.lower().startswith("| candidate") or alignment_re.match(ln):
            continue

        cells = [c.strip() for c in ln.split("|")[1:-1]]
        if len(cells) >= 4:
            name, fit, why, improve = cells[:4]
            rows.append(
                {
                    "name": name,
                    "fit": fit.rstrip("%"),
                    "why": why,
                    "improve": improve,
                }
            )

    # add experience column if lengths match
    if rows and len(rows) == len(years_info):
        for r, y in zip(rows, years_info):
            r["years"] = y

    # fallback – if parsing still fails just show raw markdown
    if not rows:
        table_html = "<pre style='white-space:pre-wrap'>" + table_md + "</pre>"
    else:
        table_html = templates.get_template("resume_rank_table.html").render(
            rows=rows,
            expected_years=expected_years,
        )

        # 6️⃣ render ---------------------------------------------------
        # Log raw Markdown in the server console for debugging

    logger.debug("\n— GPT table markdown —\n%s\n——————————————", table_md)


    html_fragment = (
        '<div class="bubble assist"><strong>Assistant:</strong><br>'
        f'{table_html}'
        '</div>'
    )

    project = {
        "description": description,
        "table_md": table_md,
        "table_html": table_html,
        "candidates": [
            {"name": m.metadata["name"],
             "fit": m.sim_pct,
             "text": m.metadata["text"]}
            for m in matches
        ],
    }

    await add_project_history(user_id, project)

    return HTMLResponse(content=html_fragment)

RESUMES_PER_PAGE = 20


@app.get("/resumes", response_class=HTMLResponse)
async def list_resumes(
    request: Request,
    page: int = 1,
    skill: str = "",
    location: str = "",
    min_years: str = "",
    max_years: str = "",
    user=Depends(require_login),
):
    """Paginated résumé overview. Works even when MongoDB is offline."""
    filters = {}
    if skill:
        filters["skill"] = skill
    if location:
        filters["location"] = location
    if min_years:
        try:
            filters["min_years"] = int(min_years)
        except ValueError:
            pass
    if max_years:
        try:
            filters["max_years"] = int(max_years)
        except ValueError:
            pass

    try:
        docs = await resumes_page(page, RESUMES_PER_PAGE, filters)
        total = await resumes_count(filters)
    except Exception as e:  # pragma: no cover
        logger.warning("resumes_page() failed: %s", e)
        docs = []
        total = 0

    resumes_for_tpl = []
    for d in docs:
        oid = d.get("_id")
        added = "—"
        if hasattr(oid, "generation_time"):
            added = oid.generation_time.strftime("%Y-%m-%d")

        resumes_for_tpl.append(
            {
                "id": d.get("resume_id", "—"),
                "name": d.get("name", "Unknown"),
                "text": (d.get("text") or "")[:400] + "…",
                "skills": d.get("skills", []),
                "location": d.get("location", ""),
                "tags": d.get("tags", []),
                "years": d.get("years"),
                "file_type": d.get("file_type", ""),
                "added": added,
            }
        )

    pages = (total + RESUMES_PER_PAGE - 1) // RESUMES_PER_PAGE
    query = {}
    if skill:
        query["skill"] = skill
    if location:
        query["location"] = location
    if min_years:
        query["min_years"] = min_years
    if max_years:
        query["max_years"] = max_years
    qs = urlencode(query)

    def remove_param(key: str) -> str:
        q = dict(query)
        q.pop(key, None)
        return urlencode(q)

    ctx = {
        "resumes": resumes_for_tpl,
        "page": page,
        "pages": pages,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < pages else None,
        "skill": skill,
        "location": location,
        "min_years": min_years,
        "max_years": max_years,
        "qs": qs,
        "rm_skill": remove_param("skill"),
        "rm_location": remove_param("location"),
        "rm_min_years": remove_param("min_years"),
        "rm_max_years": remove_param("max_years"),
    }
    return await render(
        request,
        "resumes.html",
        ctx,
        page_title="Résumés",
        active="/resumes",
    )


PROJECTS_PER_PAGE = 10


@app.get("/projects", response_class=HTMLResponse)
async def project_history(
    request: Request, page: int = 1, user=Depends(require_login)
):
    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    doc = await chat_find_one({"user_id": user_id}) or {}
    projects = doc.get("projects", [])
    projects = sorted(projects, key=lambda p: p.get("ts", 0), reverse=True)
    total = len(projects)
    start = (page - 1) * PROJECTS_PER_PAGE
    end = start + PROJECTS_PER_PAGE
    page_projects = projects[start:end]
    pages = (total + PROJECTS_PER_PAGE - 1) // PROJECTS_PER_PAGE
    ctx = {
        "projects": page_projects,
        "page": page,
        "pages": pages,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < pages else None,
    }
    return await render(
        request,
        "projects.html",
        ctx,
        page_title="Projects",
        active="/projects",
    )


@app.post("/delete_project")
async def delete_project_route(request: Request,
                         ts: str = Form(...),
                         user=Depends(require_login)):
    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    deleted = await delete_project(user_id, ts)
    if deleted == 0:
        logger.warning("No project found with ts %s", ts)
    else:
        logger.info("Deleted project with ts %s", ts)
    return RedirectResponse("/projects", status_code=303)


@app.get("/edit_project", response_class=HTMLResponse)
async def edit_project_form(request: Request, ts: str, user=Depends(require_login)):
    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    doc = await chat_find_one({"user_id": user_id}) or {}
    projects = doc.get("projects", [])
    project = next((p for p in projects if p.get("ts") and p["ts"].isoformat() == ts), None)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return await render(
        request,
        "edit_project.html",
        {"project": project},
        page_title="Edit Project",
        active="/projects",
    )


@app.post("/edit_project")
async def edit_project_post(request: Request, ts: str = Form(...), description: str = Form(...), user=Depends(require_login)):
    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    await update_project_description(user_id, ts, description)
    return RedirectResponse("/projects", status_code=303)

@app.get("/view_resumes", response_class=HTMLResponse)
async def view_resumes(request: Request, user=Depends(require_login)):
    # Search all resumes from the "resumes" namespace
    try:
        logger.info("Querying Pinecone for all resumes...")
        results = index.query(
            vector=[0] * 1536,  # Placeholder vector for getting all records
            top_k=10,  # Adjust the number of records returned
            include_metadata=True,
            namespace="resumes"
        )

        if results.matches:
            logger.info("Found %d resumes in Pinecone.", len(results.matches))
        else:
            logger.info("No matches found in Pinecone.")

        # Prepare results to send to the frontend
        resumes = []
        for match in results.matches:
            logger.info(
                "Resume: %s, Name: %s",
                match.id,
                match.metadata.get("name", "Unknown"),
            )
            resumes.append({
                "id": match.id,
                "name": match.metadata.get("name", "Unknown"),
                "text": match.metadata.get("text", "No description available."),
                "added": match.metadata.get("added", "—"),
            })

        return await render(request, "resumes.html",
                      {"resumes": resumes},
                        page_title="Résumés", active="/resumes")

    except Exception as e:
        logger.error("Pinecone query failed: %s", e)
        return await render(request, "resumes.html",
                      {"resumes": resumes},
                        page_title="Résumés", active="/resumes")


@app.get("/edit_resume", response_class=HTMLResponse)
async def edit_resume(request: Request, id: str, user=Depends(require_login)):
    """
    Fetch the résumé by either resume_id or Mongo _id.
    """
    doc = None
    try:
        # 1) try resume_id
        doc = await resumes_collection.find_one({"resume_id": id})
        # 2) fall back to _id if the first lookup failed
        if not doc and ObjectId.is_valid(id):
            doc = await resumes_collection.find_one({"_id": ObjectId(id)})
    except Exception as e:
        logger.error("Mongo lookup failed: %s", e)

    if not doc:                        # still nothing → 404
        return PlainTextResponse("Résumé not found", status_code=404)

    return await render(request, "edit_resume.html",
        {
            "request": request,
            "resume": {
                "resume_id": doc["resume_id"],   # ← key renamed
                "name":      doc["name"],
                "text":      doc["text"],
                "skills":    ", ".join(doc.get("skills", [])),
                "location":  doc.get("location", ""),
                "years":     doc.get("years", ""),
            },
        },
        page_title="Edit résumé")

@app.post("/update_resume")
async def update_resume_route(
        resume_id: str = Form(...),   # ★ must match the hidden input’s name
        name: str      = Form(...),
        text: str      = Form(...),
        skills: str    = Form(""),
        location: str  = Form(""),
        years: int | None = Form(None),
        tags: str = Form("")):
    skill_list = [s.strip() for s in skills.split(',') if s.strip()]
    tag_list = [t.strip() for t in tags.split(',') if t.strip()]
    loc_val = location or None
    modified = await update_resume(resume_id, name, text,
                                   skill_list if skills else None,
                                   loc_val,
                                   years,
                                   tag_list if tags else None)
    if not modified:
        logger.warning("Nothing updated for resume_id %s", resume_id)
    return RedirectResponse("/resumes", status_code=303)

@app.post("/delete_resume")
async def delete_resume_route(id: str = Form(...)):
    logger.info("Deleting resume with ID: %s", id)

    # 1. Delete from MongoDB
    deleted_count = await delete_resume_by_id(id)
    if deleted_count > 0:
        logger.info("Deleted %d doc(s) from MongoDB.", deleted_count)
    else:
        logger.warning("No MongoDB doc with resume_id %s, will still try Pinecone.", id)

    # 2. Delete from Pinecone
    try:
        index.delete(ids=[id], namespace="resumes")
        logger.info("Deleted %s from Pinecone.", id)
    except Exception as e:
        logger.error("Pinecone deletion failed for %s: %s", id, e)

    return RedirectResponse("/resumes", status_code=303)

from routes.users import router as users_router, login_form, login_post, logout, user_admin, create_user_admin, delete_user_admin, edit_user_form, update_user_admin, reset_user_password, profile, change_password
from routes.chat import router as chat_router, chat_interface, chat, chat_action, chat_followup

app.include_router(users_router)
app.include_router(chat_router)
