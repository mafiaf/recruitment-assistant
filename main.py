from __future__ import annotations

import io
import os
import re
import uuid
from types import SimpleNamespace
from typing import List, Optional

import mammoth
import numpy as np
import openai
import pdfplumber
import PyPDF2
from docx import Document
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── local helper modules ──────────────────────────────────────────────────────
from db import (
    chat_find_one,
    chat_upsert,
    resumes_all,
    resumes_by_ids,
)
from mongo_utils import update_resume, delete_resume_by_id
from pinecone_utils import (
    add_resume_to_pinecone,
    embed_text,
    index,
    search_best_resumes,
)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
def chat_interface(request: Request):
    # sidebar list (may be empty if Mongo down)
    candidates = [
        {"id": r["resume_id"], "name": r["name"]}
        for r in resumes_all()
    ]

    # previous conversation
    user_id = "demo_user"  # TODO real session cookie
    doc = chat_find_one({"user_id": user_id}) or {}
    history = doc.get("messages", [])

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "history": history,
            "candidates": candidates,
        },
    )

# ═════════════════════════════════════════════════════════════════════════════
# /chat (JSON) – text only follow‑ups
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_class=JSONResponse)
async def chat(request: Request):
    payload = await request.json()
    user_text = payload.get("text", "").strip()
    selected_ids: list[str] = payload.get("candidate_ids", [])
    user_id = "demo_user"

    doc = chat_find_one({"user_id": user_id}) or {}
    history = doc.get("messages", [])
    last_proj = doc.get("last_project", {})

    # ── contextual blocks (project & snippets) ────────────────────────────
    snippets: list[str] = []
    for c in last_proj.get("candidates", []):
        if not selected_ids or c["name"] in selected_ids:
            snippets.append(f"**{c['name']}** ({c['fit']} %)\n{c['text'][:600]}…")

    system_ctx = []
    if desc := last_proj.get("description"):
        system_ctx.append(f"### Project\n{desc}")
    if snippets:
        system_ctx.append("### Candidate résumés\n" + "\n\n".join(snippets))

    # assemble messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a recruitment assistant who can answer follow‑up "
                "questions about the project and the candidate résumés provided."
            ),
        }
    ]
    if system_ctx:
        messages.append({"role": "system", "content": "\n\n".join(system_ctx)})

    messages.extend({"role": h["role"], "content": h["content"]} for h in history[-6:])
    messages.append({"role": "user", "content": user_text})

    # call GPT
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=600,
    )
    assistant_reply = resp.choices[0].message.content

    # persist
    history += [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_reply},
    ]
    chat_upsert(user_id, {"messages": history})

    return {"reply": assistant_reply}

# ═════════════════════════════════════════════════════════════════════════════
# /chat_action – simple buttons (top‑5, python expert)
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/chat_action", response_class=HTMLResponse)
async def chat_action(
    request: Request,
    action: str = Form(...),
    candidate_ids: List[str] = Form([]),
):
    if action == "top5":
        user_instr = "Pick the top 5 candidates and explain briefly why."
    elif action == "python_expert":
        user_instr = "Which candidate has the most Python experience?"
    else:
        user_instr = "Sorry, I didn’t understand that request."

    # candidate set
    if candidate_ids:
        matches = resumes_by_ids(candidate_ids)
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

    candidates = [{"id": r["resume_id"], "name": r["name"]} for r in resumes_all()]
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "reply": reply, "candidates": candidates},
    )

# ═════════════════════════════════════════════════════════════════════════════
# upload résumé – saves to Pinecone + Mongo (if available)
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/upload_resume", response_class=HTMLResponse)
async def upload_resume(
    request: Request,
    name: str = Form(""),
    text: str = Form(""),
    file: UploadFile = File(None),
):
    if file and file.filename:
        data = await file.read()
        text = extract_text(data, file.filename)
        if not text:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "popup": "Unsupported file type."},
            )

    if not text.strip():
        return templates.TemplateResponse(
            "index.html", {"request": request, "popup": "Résumé text is empty."}
        )

    resume_id = name.strip() or str(uuid.uuid4())
    add_resume_to_pinecone(text, resume_id, {"name": name or resume_id, "text": text}, "resumes")

    # Mongo may be offline, wrap in try/except
    try:
        resumes_collection.insert_one({"name": name or resume_id, "resume_id": resume_id, "text": text})
    except Exception as e:  # pragma: no cover
        print("🛑 Mongo insert failed:", e)

    return templates.TemplateResponse(
        "index.html", {"request": request, "popup": f"Résumé added as {resume_id}."}
    )


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
    description: str = Form(""),
    file: UploadFile = File(None),
    candidate_ids: Optional[List[str]] = Form(None),
):
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

    chats.update_one(
    {"user_id": "demo_user"},
    {"$set": {
        "last_project": {
            "description": description,
            "table_md": table_md,        # raw Markdown
            "candidates": [
                {                       # minimal info to retrieve later
                  "name": m.metadata["name"],
                  "fit" : m.sim_pct,
                  "text": m.metadata["text"]
                } for m in matches
            ]
        }
    }},
    upsert=True
)

    return HTMLResponse(content=html_fragment)

@app.get("/resumes", response_class=HTMLResponse)
def list_resumes(request: Request):
    """
    Returns an overview of all résumés. Works even when MongoDB is offline.
    """
    try:
        docs = resumes_all()          # <- new helper (returns [] if DB down)
    except Exception as e:            # pragma: no cover
        print("⚠️  resumes_all() failed:", e)
        docs = []

    # normalise for the template
    resumes_for_tpl = [
        {
            "id":   d.get("resume_id", "—"),
            "name": d.get("name",       "Unknown"),
            "text": (d.get("text") or "")[:400] + "…",
        }
        for d in docs
    ]

    return templates.TemplateResponse(
        "resumes.html",
        {"request": request, "resumes": resumes_for_tpl},
    )

@app.get("/view_resumes", response_class=HTMLResponse)
async def view_resumes(request: Request):
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
                "text": match.metadata.get("text", "No description available.")
            })

        return templates.TemplateResponse("resumes.html", {"request": request, "resumes": resumes})

    except Exception as e:
        print("🛑 Pinecone query failed:", e)
        return templates.TemplateResponse("resumes.html", {"request": request, "resumes": []})


@app.post("/update_resume")
async def update_resume_route(id: str = Form(...), name: str = Form(...), text: str = Form(...)):
    update_resume(id, name, text)
    return RedirectResponse("/resumes", status_code=303)

@app.post("/delete_resume")
async def delete_resume_route(id: str = Form(...)):
    print(f"🟢 Deleting resume with ID: {id}")

    # 1. Delete from MongoDB
    deleted_count = delete_resume_by_id(id)
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
async def chat_followup(request: Request):
    payload        = await request.json()
    user_text      = payload.get("message", "").strip()
    user_id        = "demo_user"                          # TODO real session

    doc = chats.find_one({"user_id": user_id}) or {}
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
        chats.update_one({"user_id":user_id},{"$set":{"messages":history}},upsert=True)
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
    chats.update_one(
        {"user_id":user_id},
        {"$set":{"messages":history}},
        upsert=True
    )
    return {"reply": assistant_reply}