from __future__ import annotations

import io
import os
import re
import uuid
from types import SimpleNamespace
from typing import List, Optional
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
    ensure_indexes,
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
        print("⚠️  Mongo not reachable – owner account not seeded")
        return

    await ensure_indexes()

    if await users_coll.count_documents({"role": "owner"}) == 0:
        await create_user(
            settings.OWNER_USER,
            settings.OWNER_PASS,
            role="owner",
        )
        print("🟢 Created initial owner account")

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
        print("⚠️  resumes_all() failed:", e)
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


@app.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request, user=Depends(require_login)):
    # sidebar list (may be empty if Mongo down)
    candidates = [
        {"id": r["resume_id"], "name": r["name"]}
        for r in await resumes_all()
    ]

    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    doc      = await chat_find_one({"user_id": user_id}) or {}
    history  = doc.get("messages", [])

    if not isinstance(history, list):    # ⬅️ guard against bad data
        history = []

    last_proj = doc.get("last_project", {})   # (keep if you still need it)

    safe_history = [
        {"role": m.get("role"), "content": sanitize_markdown(m.get("content", ""))}
        for m in history if isinstance(m, dict)
    ]

    return await render(
        request,
        "chat.html",
        {"history": safe_history, "candidates": candidates},
        page_title="Chat",
    )


# ──────────────────────────────────────────────────────────────────────────
# /chat  – text follow-ups (no file upload)
# ──────────────────────────────────────────────────────────────────────────
@app.post("/chat", response_class=JSONResponse)
async def chat(request: Request,
               chat_data: ChatRequest = Body(...),
               user=Depends(require_login)):
    user_text    = chat_data.text.strip()
    selected_ids = chat_data.candidate_ids        # list[str]
    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id      = session_user["username"] if session_user else "anon"

    # 1️⃣ fetch previous convo & project context
    doc        = await chat_find_one({"user_id": user_id}) or {}
    history    = doc.get("messages", [])
    last_proj  = doc.get("last_project", {})               # may be {}

    # 2️⃣ guarantee history is a list (avoid slice KeyError)
    if not isinstance(history, list):
        history = []

    # 3️⃣ build résumé snippets to inject (filtered by sidebar checkboxes)
    snippets = []
    for c in last_proj.get("candidates", []):
        if not selected_ids or c["name"] in selected_ids:
            snippets.append(
                f"**{c['name']}** ({c['fit']} %)\n{c['text'][:600]}…"
            )

    # 4️⃣ system-level context blocks
    system_blocks = []
    if desc := last_proj.get("description"):
        system_blocks.append(f"### Project\n{desc}")
    if snippets:
        system_blocks.append("### Candidate résumés\n" + "\n\n".join(snippets))

    # 5️⃣ assemble message list for GPT
    messages = [
        {
            "role": "system",
            "content": (
                "You are a recruitment assistant who can answer follow-up "
                "questions about the project and the candidate résumés provided."
            ),
        }
    ]
    if system_blocks:
        messages.append({"role": "system", "content": "\n\n".join(system_blocks)})

    # last 6 legitimate turns (skip empty / malformed entries)
    for turn in history[-6:]:
        if isinstance(turn, dict) and turn.get("content", "").strip():
            messages.append({"role": turn["role"], "content": turn["content"]})

    # current user turn
    messages.append({"role": "user", "content": user_text})

    # 6️⃣ call OpenAI
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=600,
    )
    assistant_reply = resp.choices[0].message.content.strip()
    safe_reply = sanitize_markdown(assistant_reply)

    # 7️⃣ persist updated history
    history.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_reply},
        ]
    )
    await chat_upsert(user_id, {"messages": history})

    return {"reply": safe_reply}


# ═════════════════════════════════════════════════════════════════════════════
# /chat_action – simple buttons (top‑5, python expert)
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/chat_action", response_class=HTMLResponse)
async def chat_action(
    request: Request,
    user=Depends(require_login),
    action: str = Form(...),
    candidate_ids: List[str] = Form([]),
):
    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    if action == "top5":
        user_instr = "Pick the top 5 candidates and explain briefly why."
    elif action == "python_expert":
        user_instr = "Which candidate has the most Python experience?"
    else:
        user_instr = "Sorry, I didn’t understand that request."

    # candidate set
    if candidate_ids:
        matches = await resumes_by_ids(candidate_ids)
    else:
        matches = search_best_resumes("(the project text…)", top_k=10, namespace="resumes")[:5]

    block = "".join(
        f"---\nName: {m.metadata['name']}\nRésumé:\n{m.metadata['text']}\n\n" for m in matches
    )

    prompt = (
        "You are a helpful recruitment assistant.\n\n" + user_instr + "\n\nHere are the candidates:\n" + block
    )

    reply = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful recruitment assistant."},
            {"role": "user", "content": prompt},
        ],
    ).choices[0].message.content
    safe_reply = sanitize_markdown(reply)

    candidates = [{"id": r["resume_id"], "name": r["name"]} for r in await resumes_all()]
    return await render(
        request,
        "chat.html",
        {"reply": safe_reply, "candidates": candidates},
        page_title="Chat",
    )

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
    resume: ResumeUpload | None = Body(None),
    user=Depends(require_login),
):
    """Handle résumé uploads from JSON or multipart forms. Supports multiple files."""

    added_names: List[str] = []

    # ── JSON payload -------------------------------------------------------
    if resume is not None:
        name = resume.name or name or ""
        text = (resume.text or text or "").strip()

        if not text:
            return await render(
                request,
                "index.html",
                {"popup": "Résumé text is empty."},
                page_title="Home",
            )

        display_name = guess_name(name, "", text)
        resume_id = slugify(display_name)
        add_resume_to_pinecone(text, resume_id, {"name": display_name, "text": text}, "resumes")
        try:
            await resumes_collection.insert_one({"resume_id": resume_id, "name": display_name, "text": text})
        except Exception as e:  # pragma: no cover
            print("🛑 Mongo insert failed:", e)
        added_names.append(display_name)

    else:
        # ── multipart/form-data (possibly multiple files) -----------------
        files: List[UploadFile] = []
        if isinstance(file, list):
            files = file
        elif file is not None:
            files = [file]

        if not files and text:
            # fallback: text field without file
            display_name = guess_name(name or "", "", text)
            resume_id = slugify(display_name)
            add_resume_to_pinecone(text, resume_id, {"name": display_name, "text": text}, "resumes")
            try:
                await resumes_collection.insert_one({"resume_id": resume_id, "name": display_name, "text": text})
            except Exception as e:  # pragma: no cover
                print("🛑 Mongo insert failed:", e)
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
            add_resume_to_pinecone(file_text, resume_id, {"name": display_name, "text": file_text}, "resumes")
            try:
                await resumes_collection.insert_one({"resume_id": resume_id, "name": display_name, "text": file_text})
            except Exception as e:  # pragma: no cover
                print("🛑 Mongo insert failed:", e)
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
    snippets = [
        f"- **{m.metadata['name']}**: {m.metadata['text'].replace(chr(10),' ')[:300]}…"
        for m in matches
    ]
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
                {"name": name,
                "fit":  fit.rstrip("%"),
                "why":  why,
                "improve": improve}
            )

    # fallback – if parsing still fails just show raw markdown
    if not rows:
        table_html = "<pre style='white-space:pre-wrap'>" + table_md + "</pre>"
    else:
        table_html = templates.get_template("resume_rank_table.html").render(rows=rows)

        # 6️⃣ render ---------------------------------------------------
        # Log raw Markdown in the server console for debugging

    print(
        "\n— GPT table markdown —\n"
        f"{table_md}\n"
        "——————————————\n",
        flush=True,
    )

    if rows:
        table_html = templates.get_template(
            "resume_rank_table.html"
        ).render(rows=rows)
    else:
        # fall‑back to raw markdown if parsing failed
        table_html = (
            "<pre style='white-space:pre-wrap'>" +
            table_md +
            "</pre>"
        )

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

    add_project_history(user_id, project)

    return HTMLResponse(content=html_fragment)

@app.get("/resumes", response_class=HTMLResponse)
async def list_resumes(request: Request, user=Depends(require_login)):
    """
    Returns an overview of all résumés. Works even when MongoDB is offline.
    """
    try:
        docs = await resumes_all()          # <- new helper (returns [] if DB down)
    except Exception as e:            # pragma: no cover
        print("⚠️  resumes_all() failed:", e)
        docs = []

    # normalise for the template
    resumes_for_tpl = []
    for d in docs:
        oid = d.get("_id")
        added = "—"
        if hasattr(oid, "generation_time"):
            added = oid.generation_time.strftime("%Y-%m-%d")

        resumes_for_tpl.append({
            "id":    d.get("resume_id", "—"),
            "name":  d.get("name",       "Unknown"),
            "text":  (d.get("text") or "")[:400] + "…",
            "added": added,
        })

    return await render(request, "resumes.html",
                  {"resumes": resumes_for_tpl},
                  page_title="Résumés", active="/resumes")


@app.get("/projects", response_class=HTMLResponse)
async def project_history(request: Request, user=Depends(require_login)):
    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    doc = await chat_find_one({"user_id": user_id}) or {}
    projects = doc.get("projects", [])
    projects = sorted(projects, key=lambda p: p.get("ts", 0), reverse=True)
    return await render(
        request,
        "projects.html",
        {"projects": projects},
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
        print(f"🛑 No project found with ts {ts}")
    else:
        print(f"🟢 Deleted project with ts {ts}")
    return RedirectResponse("/projects", status_code=303)

@app.get("/view_resumes", response_class=HTMLResponse)
async def view_resumes(request: Request, user=Depends(require_login)):
    # Search all resumes from the "resumes" namespace
    try:
        print("🟢 Querying Pinecone for all resumes...")
        results = index.query(
            vector=[0] * 1536,  # Placeholder vector for getting all records
            top_k=10,  # Adjust the number of records returned
            include_metadata=True,
            namespace="resumes"
        )

        if results.matches:
            print(f"🟢 Found {len(results.matches)} resumes in Pinecone.")
        else:
            print("🛑 No matches found in Pinecone.")

        # Prepare results to send to the frontend
        resumes = []
        for match in results.matches:
            print(f"🟢 Resume: {match.id}, Name: {match.metadata.get('name', 'Unknown')}")
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
        print("🛑 Pinecone query failed:", e)
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
        print("🛑 Mongo lookup failed:", e)

    if not doc:                        # still nothing → 404
        return PlainTextResponse("Résumé not found", status_code=404)

    return await render(request, "edit_resume.html",
        {
            "request": request,
            "resume": {
                "resume_id": doc["resume_id"],   # ← key renamed
                "name":      doc["name"],
                "text":      doc["text"],
            },
        },
        page_title="Edit résumé")

@app.post("/update_resume")
async def update_resume_route(
        resume_id: str = Form(...),   # ★ must match the hidden input’s name
        name: str      = Form(...),
        text: str      = Form(...)):
    modified = await update_resume(resume_id, name, text)
    if not modified:
        print(f"🛑 Nothing updated for resume_id {resume_id!r}")
    return RedirectResponse("/resumes", status_code=303)

@app.post("/delete_resume")
async def delete_resume_route(id: str = Form(...)):
    print(f"🟢 Deleting resume with ID: {id}")

    # 1. Delete from MongoDB
    deleted_count = await delete_resume_by_id(id)
    if deleted_count > 0:
        print(f"✅ Deleted {deleted_count} doc(s) from MongoDB.")
    else:
        print(f"🛑 No MongoDB doc with resume_id {id}, will still try Pinecone.")

    # 2. Delete from Pinecone
    try:
        index.delete(ids=[id], namespace="resumes")
        print(f"✅ Deleted {id} from Pinecone.")
    except Exception as e:
        print(f"🛑 Pinecone deletion failed for {id}: {e}")

    return RedirectResponse("/resumes", status_code=303)


@app.post("/chat_followup", response_class=JSONResponse)
async def chat_followup(request: Request, user=Depends(require_login)):
    payload        = await request.json()
    user_text      = payload.get("message", "").strip()
    session_user = await get_current_user(request.cookies.get(COOKIE_NAME))
    user_id        = session_user["username"] if session_user else "anon"

    doc = await chats.find_one({"user_id": user_id}) or {}
    history  = doc.get("messages", [])
    project  = doc.get("last_project")

    # ── 1. if no project context fall back to generic model ────────────────
    if not project:
        history.append({"role":"user","content":user_text})
        messages = (
            [{"role":"system","content":"You are a helpful recruitment assistant."}]
            + history
        )
        reply = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        ).choices[0].message.content
        history.append({"role":"assistant","content":reply})
        await chats.update_one({"user_id":user_id},{"$set":{"messages":history}},upsert=True)
        return {"reply": reply}

    # ── 2. build alias map & snippets ──────────────────────────────────────
    alias   = {}               #  e.g. {"baris": "Baris Firat", ...}
    snippet_lines = []
    for c in project["candidates"]:
        fullname = c["name"]
        first    = fullname.split()[0].lower()
        alias[first] = fullname
        snippet_lines.append(
            f"- **{fullname}** (Fit {c['fit']} %): {c['text'][:250]}…"
        )

    alias_block = ", ".join(f"'{k}'→'{v}'" for k,v in alias.items())

    # ── 3. compose system prompt ───────────────────────────────────────────
    system_prompt = (
        "You are an expert recruitment assistant.\n"
        "Use ONLY the project description, markdown ranking table and résumé "
        "snippets provided below. You are allowed to quote and compare the "
        "candidates. Keep answers ≤120 words unless user asks for more.\n\n"
        f"Alias map: {alias_block}\n\n"
        "## Project description\n"
        f"{project['description']}\n\n"
        "## Ranking table (markdown)\n"
        f"{project['table_md']}\n\n"
        "## Candidate résumé snippets\n" +
        "\n".join(snippet_lines)
    )

    # ── 4. add new user turn & send to GPT ─────────────────────────────────
    history.append({"role":"user","content":user_text})
    messages = [{"role":"system","content":system_prompt}] + history

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=350
    )
    assistant_reply = response.choices[0].message.content.strip()

    # ── 5. persist & return ────────────────────────────────────────────────
    history.append({"role":"assistant","content":assistant_reply})
    await chats.update_one(
        {"user_id":user_id},
        {"$set":{"messages":history}},
        upsert=True
    )
    return {"reply": assistant_reply}


    # ── Login ────────────────────────────────────────────────


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
       return await render(request, "login.html", page_title="Login", active="/login")

@app.post("/login")
async def login_post(request: Request,
                     username: str = Form(...),
                     password: str = Form(...)):
    user = await verify_password(username, password)
    if not user:
        return await render(request, "login.html",
                      {"error": "Invalid credentials"},
                      page_title="Login", active="/login",
                      status_code=401)
    resp = RedirectResponse("/", status_code=303)
    set_session(resp, user)
    return resp

@app.get("/logout")
def logout():
    resp = RedirectResponse("/", status_code=303)
    clear_session(resp)
    return resp



@app.get("/admin/users", response_class=HTMLResponse, dependencies=[Depends(require_owner)])
async def user_admin(request: Request):
    cursor = users_coll.find({}, {"_id": 0, "hashed_password": 0})
    users = await cursor.to_list(None)
    return await render(request, "admin_users.html",
                  {"users": users},
                  page_title="User admin", active="/admin/users")

@app.post("/admin/users", dependencies=[Depends(require_owner)])
async def create_user_admin(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("user"),
):
    if await users_coll.find_one({"username": username}):
        raise HTTPException(400, "User already exists")
    await create_user(username, password, role)
    return RedirectResponse("/admin/users", status_code=303)

@app.post("/admin/users/delete", dependencies=[Depends(require_owner)])
async def delete_user_admin(
        username: str = Form(...),
        current   = Depends(require_owner)):             # you are the owner
    """
    Deletes the account given by *username*.  
    • Owner may not delete themselves to avoid lock-out.
    """
    if username == current["username"]:
        raise HTTPException(400, "You cannot delete your own owner account")

    res = await users_coll.delete_one({"username": username})
    if res.deleted_count == 0:
        raise HTTPException(404, "User not found")

    return RedirectResponse("/admin/users", status_code=303)


@app.get("/admin/users/{username}/edit", response_class=HTMLResponse,
         dependencies=[Depends(require_owner)])
async def edit_user_form(request: Request, username: str):
    user = await users_coll.find_one({"username": username}, {"_id": 0, "hashed_password": 0})
    if not user:
        raise HTTPException(404, "User not found")
    return await render(request, "edit_user.html",
                  {"user": user},
                  page_title="Edit user", active="/admin/users")


@app.post("/admin/users/{username}/edit", dependencies=[Depends(require_owner)])
async def update_user_admin(username: str, new_username: str = Form(...)):
    if username != new_username:
        if await users_coll.find_one({"username": new_username}):
            raise HTTPException(400, "Username already exists")
        res = await users_coll.update_one({"username": username}, {"$set": {"username": new_username}})
        if res.matched_count == 0:
            raise HTTPException(404, "User not found")
    return RedirectResponse("/admin/users", status_code=303)


@app.post("/admin/users/{username}/reset", dependencies=[Depends(require_owner)])
async def reset_user_password(username: str):
    res = await users_coll.update_one(
        {"username": username},
        {"$set": {"hashed_password": pwd_context.hash(DEFAULT_PASSWORD)}}
    )
    if res.matched_count == 0:
        raise HTTPException(404, "User not found")
    return RedirectResponse("/admin/users", status_code=303)


@app.get("/profile", response_class=HTMLResponse, dependencies=[Depends(require_login)])
async def profile(request: Request, user = Depends(get_current_user)):
    return await render(request, "profile.html",
                  {"user": user},
                  page_title="Profile", active="/profile")

# change password
@app.post("/profile/password", dependencies=[Depends(require_login)])
async def change_password(
        old: str = Form(...),
        new: str = Form(...),
        user = Depends(get_current_user)
    ):
    if not await verify_password(user["username"], old):
        raise HTTPException(400, "Old password incorrect")
    await users_coll.update_one(
        {"username": user["username"]},
        {"$set": {"hashed_password": pwd_context.hash(new)}}
    )
    return RedirectResponse("/profile?ok=1", status_code=303)
