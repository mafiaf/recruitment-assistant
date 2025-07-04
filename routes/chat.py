from typing import List
from datetime import datetime
from fastapi import APIRouter, Request, Depends, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import main
from schemas import ChatRequest

router = APIRouter()

# default set of quick prompt templates shown in the chat UI
QUICK_PROMPTS = [
    {
        "title": "Cv-audit",
        "text": (
            "Doe alsof je een recruiter bent met meer dan 10 jaar ervaring. "
            "Bekijk dit cv en vertel me precies waarom het wordt genegeerd. "
            "Wees direct, geen mooipraterij. Benoem wat zwak is en hoe het beter kan."
        ),
        "icon": "fa-solid fa-magnifying-glass",
        "group": "Audit",
    },
    {
        "title": "ATS-optimalisatie",
        "text": (
            "Herschrijf mijn cv volledig ATS-proof voor de functie "
            "[voeg functietitel in]. Voeg branchespecifieke trefwoorden en vaardigheden toe "
            "en kwantificeer mijn resultaten zonder robotachtig te klinken."
        ),
        "icon": "fa-solid fa-chart-line",
        "group": "Rewrite",
    },
    {
        "title": "Vacature-afstemming",
        "text": (
            "Neem deze vacaturetekst en stem mijn cv er regel voor regel op af. "
            "Laat toon, vaardigheden en resultaten aansluiten op hun wensen – scherp en eerlijk."
        ),
        "icon": "fa-solid fa-link",
        "group": "Rewrite",
    },
    {
        "title": "Zelfverzekerde herschrijving",
        "text": (
            "Herschrijf mijn cv zodat het krachtig en prestatiegericht klinkt. "
            "Geen passieve formuleringen. Maak het resultaatgericht en zelfverzekerd, "
            "alsof iedere werkgever me wil aannemen."
        ),
        "icon": "fa-solid fa-pen",
        "group": "Rewrite",
    },
    {
        "title": "Sollicitatiebrief",
        "text": (
            "Schrijf een beknopte (onder 200 woorden) en menselijk klinkende sollicitatiebrief "
            "voor deze functie: [voeg functietitel in]. Toon enthousiasme, competentie en benadruk "
            "de waarde die ik bied."
        ),
        "icon": "fa-solid fa-envelope",
        "group": "Rewrite",
    },
    {
        "title": "Interview-oefening",
        "text": (
            "Geef me de 10 meest voorkomende gedragsvragen voor de functie "
            "[voeg functietitel in] en beantwoord ze volgens de STAR-methode "
            "met concrete, sterke voorbeelden"
        ),
        "icon": "fa-solid fa-comments",
        "group": "Interview",
    },

]

# System prompts for each chat mode
MODE_PROMPTS = {
    "general": (
        "Je bent een wervingsassistent die vervolgvragen kan beantwoorden over het project "
        "en de aangeleverde cv's. Houd je antwoorden behulpzaam en to the point en reageer in het Nederlands."
    ),
    "ats": (
        "Je bent een ATS-optimalisatie-expert. Reageer in het Nederlands en richt je op het herschrijven van cv's "
        "zodat ze goed scoren in Applicant Tracking Systems, binnen de context van het project en de cv's."
    ),
    "role": (
        "Je bent een loopbaancoach gespecialiseerd in functiegericht advies. Gebruik de projectinformatie en cv's "
        "om gerichte begeleiding te geven en antwoord steeds in het Nederlands."
    ),
}

@router.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request, user=Depends(main.require_login)):
    candidates = [
        {
            "id": r.get("resume_id"),
            "name": r.get("name"),
            "location": r.get("location", ""),
            "skills": r.get("skills", [])[:3],
        }
        for r in await main.resumes_all()
    ]
    session_user = await main.get_current_user(request.cookies.get(main.COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    doc = await main.chat_find_one({"user_id": user_id}) or {}
    history = doc.get("messages", [])
    if not isinstance(history, list):
        history = []
    safe_history = [
        {
            "role": m.get("role"),
            "content": main.sanitize_markdown(m.get("content", "")),
            "time": m.get("time"),
        }
        for m in history
        if isinstance(m, dict)
    ]
    if not safe_history:
        safe_history.append({
            "role": "assistant",
            "content": "Hallo! Wat wil je vandaag doen?",
            "time": datetime.utcnow().isoformat(timespec="seconds"),
        })
    return await main.render(
        request,
        "chat.html",
        {
            "history": safe_history,
            "candidates": candidates,
            "quick_prompts": QUICK_PROMPTS,
        },
        page_title="Chat",
    )

@router.post("/chat", response_class=JSONResponse)
async def chat(request: Request, chat_data: ChatRequest = Body(...), user=Depends(main.require_login)):
    user_text = chat_data.text.strip()
    selected_ids = chat_data.candidate_ids
    mode = chat_data.mode
    session_user = await main.get_current_user(request.cookies.get(main.COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    doc = await main.chat_find_one({"user_id": user_id}) or {}
    history = doc.get("messages", [])
    last_proj = doc.get("last_project", {})
    # ── optional user instruction filter ─────────────────────────────────
    lowered = user_text.lower()
    blocked_phrases = [
        "ignore previous text",
        "ignore previous instructions",
        "write me a recipe",
        "write a recipe",
    ]
    if any(p in lowered for p in blocked_phrases):
        refusal = "Sorry, I can't comply with that request."
        ts = datetime.utcnow().isoformat(timespec="seconds")
        history.extend([
            {"role": "user", "content": user_text, "time": ts},
            {"role": "assistant", "content": refusal, "time": ts},
        ])
        await main.chat_upsert(user_id, {"messages": history})
        return JSONResponse({"reply": main.sanitize_markdown(refusal), "time": ts})
    if not isinstance(history, list):
        history = []
    snippets = []
    selected_docs = await main.resumes_by_ids(selected_ids) if selected_ids else []
    selected_names = {d.metadata["name"] for d in selected_docs}

    cand_data = last_proj.get("candidates", [])
    if cand_data:
        for c in cand_data:
            if not selected_names or c["name"] in selected_names:
                snippets.append(f"**{c['name']}** ({c['fit']} %)\n{c['text'][:600]}…")
    else:
        for d in selected_docs:
            snippets.append(f"**{d.metadata['name']}**\n{d.metadata['text'][:600]}…")
    system_blocks = []
    if desc := last_proj.get("description"):
        system_blocks.append(f"### Project\n{desc}")
    if snippets:
        system_blocks.append("### Candidate résumés\n" + "\n\n".join(snippets))
    system_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["general"])
    messages = [{"role": "system", "content": system_prompt}]
    if system_blocks:
        messages.append({"role": "system", "content": "\n\n".join(system_blocks)})
    for turn in history[-6:]:
        if isinstance(turn, dict) and turn.get("content", "").strip():
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_text})
    resp = main.openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=600,
    )
    assistant_reply = resp.choices[0].message.content.strip()
    safe_reply = main.sanitize_markdown(assistant_reply)
    ts = datetime.utcnow().isoformat(timespec="seconds")
    history.extend([
        {"role": "user", "content": user_text, "time": ts},
        {"role": "assistant", "content": assistant_reply, "time": ts},
    ])
    await main.chat_upsert(user_id, {"messages": history})
    return JSONResponse({"reply": safe_reply, "time": ts})

@router.post("/chat_action", response_class=HTMLResponse)
async def chat_action(
    request: Request,
    user=Depends(main.require_login),
    action: str = Form(...),
    candidate_ids: List[str] = Form([]),
):
    session_user = await main.get_current_user(request.cookies.get(main.COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    if action == "top5":
        user_instr = "Pick the top 5 candidates and explain briefly why."
    elif action == "python_expert":
        user_instr = "Which candidate has the most Python experience?"
    else:
        user_instr = "Sorry, I didn’t understand that request."
    if candidate_ids:
        matches = await main.resumes_by_ids(candidate_ids)
    else:
        matches = main.search_best_resumes("(the project text…)", top_k=10, namespace="resumes")[:5]
    block = "".join(
        f"---\nName: {m.metadata['name']}\nRésumé:\n{m.metadata['text']}\n\n" for m in matches
    )
    prompt = "You are a helpful recruitment assistant.\n\n" + user_instr + "\n\nHere are the candidates:\n" + block
    reply = main.openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful recruitment assistant."},
            {"role": "user", "content": prompt},
        ],
    ).choices[0].message.content
    safe_reply = main.sanitize_markdown(reply)
    candidates = [{"id": r["resume_id"], "name": r["name"]} for r in await main.resumes_all()]
    return await main.render(
        request,
        "chat.html",
        {
            "reply": safe_reply,
            "candidates": candidates,
            "quick_prompts": QUICK_PROMPTS,
        },
        page_title="Chat",
    )

@router.post("/chat_followup", response_class=JSONResponse)
async def chat_followup(request: Request, user=Depends(main.require_login)):
    payload = await request.json()
    user_text = payload.get("message", "").strip()
    session_user = await main.get_current_user(request.cookies.get(main.COOKIE_NAME))
    user_id = session_user["username"] if session_user else "anon"
    doc = await main.chats.find_one({"user_id": user_id}) or {}
    history = doc.get("messages", [])
    project = doc.get("last_project")
    if not project:
        history.append({"role": "user", "content": user_text})
        messages = ([{"role": "system", "content": "You are a helpful recruitment assistant."}] + history)
        reply = main.openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        ).choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        await main.chats.update_one({"user_id": user_id}, {"$set": {"messages": history}}, upsert=True)
        return {"reply": reply}
    alias = {}
    snippet_lines = []
    for c in project["candidates"]:
        fullname = c["name"]
        first = fullname.split()[0].lower()
        alias[first] = fullname
        snippet_lines.append(f"- **{fullname}** (Fit {c['fit']} %): {c['text'][:250]}…")
    alias_block = ", ".join(f"'{k}'→'{v}'" for k, v in alias.items())
    system_prompt = (
        "You are an expert recruitment assistant.\n"
        "Use ONLY the project description, markdown ranking table and résumé snippets provided below. You are allowed to quote and compare the candidates. Keep answers ≤120 words unless user asks for more.\n\n"
        f"Alias map: {alias_block}\n\n"
        "## Project description\n"
        f"{project['description']}\n\n"
        "## Ranking table (markdown)\n"
        f"{project['table_md']}\n\n"
        "## Candidate résumé snippets\n" + "\n".join(snippet_lines)
    )
    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": system_prompt}] + history
    response = main.openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=350,
    )
    assistant_reply = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": assistant_reply})
    await main.chats.update_one(
        {"user_id": user_id},
        {"$set": {"messages": history}},
        upsert=True,
    )
    return {"reply": assistant_reply}
