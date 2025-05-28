# db.py  (replace the file contents)

import os, certifi, logging
from datetime import datetime
from typing import Any, Dict, List
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from bson import ObjectId
from types import SimpleNamespace

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("MONGO_DB_NAME", "recruitment_app")

# ------------------------------------------------------------------ #
# lazy client – no handshake until first real op                     #
# ------------------------------------------------------------------ #
client = MongoClient(
    MONGO_URI,
    connect=False,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5_000,   # fail fast
)

db = client[DB_NAME]
chats  = db["chats"]
resumes_collection = db["resumes"]

_MONGO_DOWN = False            # global flag we flip after first failure

def _guard(op: str) -> bool:
    """Return True if Mongo is down – calling code should return a harmless
    default instead of querying."""
    global _MONGO_DOWN
    if _MONGO_DOWN:
        logging.warning(f"⚠️  mongo down – {op} skipped")
        return True
    try:
        client.admin.command("ping")
        return False
    except errors.PyMongoError as e:
        logging.error("🛑  Mongo unreachable, NO-DB mode")
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
