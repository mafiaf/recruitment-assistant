# mongo_utils.py  –  “lazy” connection, non-blocking startup
import os, certifi, logging
from typing import List
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from bson import ObjectId
from types import SimpleNamespace
from passlib.hash import bcrypt
from datetime import datetime
from pinecone_utils import add_resume_to_pinecone

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")                 # Atlas SRV string
DB_NAME   = os.getenv("MONGO_DB_NAME", "recruitment_app")

# ------------------------------------------------------------------ #
# 1) create client but DO NOT connect yet (connect=False)            #
# ------------------------------------------------------------------ #
client = MongoClient(
    MONGO_URI,
    connect=False,                    # <- no handshake at import time
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5_000,   # 5 s max when we *do* touch Mongo
)

db = client[DB_NAME]
_resumes = db["resumes"]
logging.info(
    "🟢 mongo_utils ready – DB: %s / collection: %s",
    DB_NAME,
    _resumes.name,
)

# ------------------------------------------------------------------ #
# 2) small wrapper: first call triggers handshake; if it fails we   #
#    flip a global flag so later calls just return harmless defaults #
# ------------------------------------------------------------------ #
_MONGO_DOWN = False

_users = db["users"]

def get_user(username: str):
    return _users.find_one({"username": username})

def create_user(username: str, pw: str, role: str = "user"):
    _users.insert_one({
        "username": username,
        "password_hash": bcrypt.hash(pw),
        "role": role,
        "created": datetime.utcnow()
    })

def verify_password(username: str, pw: str) -> dict | None:
    doc = get_user(username)
    if doc and bcrypt.verify(pw, doc["password_hash"]):
        return doc
    return None

def _guard(op: str):
    global _MONGO_DOWN
    if _MONGO_DOWN:
        logging.warning(f"⚠️  mongo down – {op} returns empty/0")
        return True
    try:
        # cheapest “are we connected” check
        client.admin.command("ping")
        return False
    except errors.PyMongoError as e:
        logging.error("🛑  Mongo unreachable, switching to NO-DB mode")
        logging.error(e)
        _MONGO_DOWN = True
        return True

# ------------------------------------------------------------------ #
# 3) helpers                                                         #
# ------------------------------------------------------------------ #
def get_all_resumes() -> List[dict]:
    if _guard("get_all_resumes"):
        return []
    return list(_resumes.find())

def get_resumes_by_ids(id_list: List[str]):
    if _guard("get_resumes_by_ids"):
        return []
    docs = _resumes.find({"_id": {"$in": [ObjectId(i) for i in id_list]}})
    return [
        SimpleNamespace(
            id=doc["resume_id"],
            metadata={"name": doc["name"], "text": doc["text"]},
        )
        for doc in docs
    ]

def update_resume(resume_id: str, name: str, text: str) -> int:
    """
    Update Mongo *and* Pinecone.  
      • If only the name changed, we upsert metadata only  
      • If the text changed too, we re-embed & upsert the full vector
    Returns: number of Mongo documents modified (0 or 1)
    """
    # 1) look up current doc to see whether text really changed
    old = _resumes.find_one({"resume_id": resume_id})
    if not old:
        return 0

    # 2) update Mongo
    res = _resumes.update_one(
        {"resume_id": resume_id},
        {"$set": {"name": name, "text": text}},
    )

    # 3) update Pinecone
    if text.strip() != old.get("text", ""):
        # text changed → recompute embedding & full upsert
        add_resume_to_pinecone(
            text,
            resume_id,
            {"name": name, "text": text},
            namespace="resumes",
        )
    else:
        # only name changed → metadata-only upsert (lighter / cheaper)
        from pinecone_utils import index
        index.upsert(
            vectors=[(resume_id, None, {"name": name})],
            namespace="resumes",
        )

    return res.modified_count

def delete_resume_by_id(resume_id: str) -> int:
    if _guard("delete_resume_by_id"):
        return 0
    res = _resumes.delete_one({"resume_id": resume_id.strip()})
    return res.deleted_count


