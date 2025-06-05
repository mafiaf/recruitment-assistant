
import os, certifi, logging
from datetime import datetime
from typing import Any, Dict, List
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from bson import ObjectId
from types import SimpleNamespace

# ── pick the right .env file ─────────────────────────────────────────
env_file = ".env.production" if os.getenv("ENV", "development").lower() == "production" else ".env.development"
load_dotenv(env_file)                      # ← now we load that file only
# ─────────────────────────────────────────────────────────────────────

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("MONGO_DB_NAME", "recruitment_app")

# decide TLS once
TLS_ON = MONGO_URI.startswith("mongodb+srv://")

client = MongoClient(
    MONGO_URI,
    connect=False,                    # lazy handshake
    tls=TLS_ON,
    tlsCAFile=certifi.where() if TLS_ON else None,
    serverSelectionTimeoutMS=5_000,
)

db      = client[DB_NAME]
chats   = db["chats"]
resumes_collection = db["resumes"]

_MONGO_DOWN = False

def _guard(op: str) -> bool:
    global _MONGO_DOWN
    if _MONGO_DOWN:
        logging.warning(f"⚠️  mongo down – {op} skipped")
        return True
    try:
        client.admin.command("ping")
        return False
    except errors.PyMongoError as e:
        logging.error("🛑  Mongo unreachable – NO-DB mode")
        logging.error(e)
        _MONGO_DOWN = True
        return True

# ------------------------------------------------------------------ #
# tiny wrappers used by main.py                                      #
# ------------------------------------------------------------------ #
def chat_find_one(query: Dict[str, Any]):
    if _guard("chat_find_one"):
        return None
    return chats.find_one(query)

def chat_upsert(user_id: str, messages: List[Dict[str,str]]):
    if _guard("chat_upsert"):
        return
    chats.update_one(
        {"user_id": user_id},
        {"$set": {"messages": messages, "ts": datetime.utcnow()}},
        upsert=True
    )

def resumes_all():
    if _guard("resumes_all"):
        return []
    return list(resumes_collection.find())

def resumes_by_ids(id_list: List[str]):
    if _guard("resumes_by_ids"):
        return []
    cur = resumes_collection.find(
        {"_id": {"$in": [ObjectId(i) for i in id_list]}}
    )
    return [
        SimpleNamespace(
            id=doc["resume_id"],
            metadata={"name": doc["name"], "text": doc["text"]},
        )
        for doc in cur
    ]

def add_project_history(user_id: str, project: Dict[str, Any]):
    """Append a project entry and update last_project."""
    if _guard("add_project_history"):
        return
    project.setdefault("ts", datetime.utcnow())
    chats.update_one(
        {"user_id": user_id},
        {
            "$push": {"projects": project},
            "$set": {"last_project": project, "ts": datetime.utcnow()}
        },
        upsert=True,
    )
