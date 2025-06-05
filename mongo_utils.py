# mongo_utils.py – environment-aware Mongo setup (dev vs prod)
import os
import certifi
import logging
from datetime import datetime
from types import SimpleNamespace
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient, errors
from bson import ObjectId
from passlib.hash import bcrypt

from pinecone_utils import add_resume_to_pinecone

env_file = ".env.production" if os.getenv("ENV", "development").lower() == "production" else ".env.development"
load_dotenv(env_file)

# ------------------------------------------------------------------ #
# 0) environment & configuration                                     #
# ------------------------------------------------------------------ #
ENV = os.getenv("ENV", "development").lower()

MONGO_URI = os.environ["MONGO_URI"]            # Atlas SRV string for both envs
DB_NAME   = "recruitment_app" if ENV == "production" else "recruitment_app_dev"

if ENV == "production":
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5_000,
    )
else:  # development – local Mongo, no TLS
    client = MongoClient(
        MONGO_URI,          # now "mongodb://localhost:27017"
        connect=False,
        tls=False,
        serverSelectionTimeoutMS=5_000,
    )

db       = client[DB_NAME]
_resumes = db["resumes"]
_users   = db["users"]
resumes_collection = _resumes  # backward compatibility for main.py
chats = db["chats"]

print(f"🟢 mongo_utils ready – env:{ENV} DB:{DB_NAME} collection:{_resumes.name}")

# create unique index only in production
if ENV == "production":
    try:
        _users.create_index("username", unique=True)
    except errors.PyMongoError as exc:
        logging.error("🛑 could not create users.username index")
        logging.error(exc)

# ------------------------------------------------------------------ #
# 1) resilient guard layer                                           #
# ------------------------------------------------------------------ #
_MONGO_DOWN = False

def _guard(op: str) -> bool:
    global _MONGO_DOWN
    if _MONGO_DOWN:
        logging.warning(f"⚠️  mongo down – {op} returns empty/0")
        return True
    try:
        client.admin.command("ping")
        return False
    except errors.PyMongoError as exc:
        logging.error("🛑 Mongo unreachable – switching to NO-DB mode")
        logging.error(exc)
        _MONGO_DOWN = True
        return True

# ------------------------------------------------------------------ #
# 2) CRUD helpers                                                    #
# ------------------------------------------------------------------ #
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
        ) for doc in docs
    ]

def update_resume(resume_id: str, name: str, text: str) -> int:
    old = _resumes.find_one({"resume_id": resume_id})
    if not old:
        return 0

    res = _resumes.update_one(
        {"resume_id": resume_id},
        {"$set": {"name": name, "text": text}},
    )

    if text.strip() != old.get("text", ""):
        add_resume_to_pinecone(
            text,
            resume_id,
            {"name": name, "text": text},
            namespace="resumes",
        )
    else:
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

# ------------------------------------------------------------------ #
# 3) chat and project helpers (migrated from db.py)                   #
# ------------------------------------------------------------------ #

def chat_find_one(query: dict):
    if _guard("chat_find_one"):
        return None
    return chats.find_one(query)


def chat_upsert(user_id: str, messages: List[dict]):
    if _guard("chat_upsert"):
        return
    chats.update_one(
        {"user_id": user_id},
        {"$set": {"messages": messages, "ts": datetime.utcnow()}},
        upsert=True,
    )


def resumes_all():
    if _guard("resumes_all"):
        return []
    return list(resumes_collection.find())


def resumes_by_ids(id_list: List[str]):
    if _guard("resumes_by_ids"):
        return []
    cur = resumes_collection.find({"_id": {"$in": [ObjectId(i) for i in id_list]}})
    return [
        SimpleNamespace(
            id=doc["resume_id"],
            metadata={"name": doc["name"], "text": doc["text"]},
        )
        for doc in cur
    ]


def add_project_history(user_id: str, project: dict):
    """Append a project entry and update last_project."""
    if _guard("add_project_history"):
        return
    project.setdefault("ts", datetime.utcnow())
    chats.update_one(
        {"user_id": user_id},
        {"$push": {"projects": project}, "$set": {"last_project": project, "ts": datetime.utcnow()}},
        upsert=True,
    )
