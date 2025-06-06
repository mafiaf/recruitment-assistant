# mongo_utils.py – environment-aware Mongo setup (dev vs prod)
import certifi
from logger import logger
from datetime import datetime
from types import SimpleNamespace
from typing import List

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import errors
from bson import ObjectId
from passlib.hash import bcrypt

from settings import settings
from pinecone_utils import add_resume_to_pinecone

# ------------------------------------------------------------------ #
# 0) environment & configuration                                     #
# ------------------------------------------------------------------ #
ENV = settings.ENV

MONGO_URI = settings.MONGO_URI            # Atlas SRV string for both envs
DB_NAME = settings.db_name

if ENV == "production":
    client = AsyncIOMotorClient(
        MONGO_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5_000,
    )
else:  # development – local Mongo, no TLS
    client = AsyncIOMotorClient(
        MONGO_URI,          # now "mongodb://localhost:27017"
        tls=False,
        serverSelectionTimeoutMS=5_000,
    )

db       = client[DB_NAME]
_resumes = db["resumes"]
_users   = db["users"]
resumes_collection = _resumes  # backward compatibility for main.py
chats = db["chats"]

logger.info(
    f"mongo_utils ready – env:{ENV} DB:{DB_NAME} collection:{_resumes.name}"
)

# create unique index only in production
async def ensure_indexes():
    if ENV == "production":
        try:
            await _users.create_index("username", unique=True)
        except errors.PyMongoError as exc:
            logger.error("🛑 could not create users.username index")
            logger.error(exc)

# ------------------------------------------------------------------ #
# 1) resilient guard layer                                           #
# ------------------------------------------------------------------ #
_MONGO_DOWN = False

async def _guard(op: str) -> bool:
    global _MONGO_DOWN
    if _MONGO_DOWN:
        logger.warning(f"⚠️  mongo down – {op} returns empty/0")
        return True
    try:
        await client.admin.command("ping")
        return False
    except errors.PyMongoError as exc:
        logger.error("🛑 Mongo unreachable – switching to NO-DB mode")
        logger.error(exc)
        _MONGO_DOWN = True
        return True

# ------------------------------------------------------------------ #
# 2) CRUD helpers                                                    #
# ------------------------------------------------------------------ #
async def get_user(username: str):
    return await _users.find_one({"username": username})

async def create_user(username: str, pw: str, role: str = "user"):
    await _users.insert_one({
        "username": username,
        "password_hash": bcrypt.hash(pw),
        "role": role,
        "created": datetime.utcnow()
    })

async def verify_password(username: str, pw: str) -> dict | None:
    doc = await get_user(username)
    if doc and bcrypt.verify(pw, doc["password_hash"]):
        return doc
    return None

async def get_all_resumes() -> List[dict]:
    if await _guard("get_all_resumes"):
        return []
    cursor = _resumes.find()
    return await cursor.to_list(None)

async def get_resumes_by_ids(id_list: List[str]):
    if await _guard("get_resumes_by_ids"):
        return []

    mongo_ids = [ObjectId(i) for i in id_list if ObjectId.is_valid(i)]
    resume_ids = [i for i in id_list if not ObjectId.is_valid(i)]

    query_parts = []
    if mongo_ids:
        query_parts.append({"_id": {"$in": mongo_ids}})
    if resume_ids:
        query_parts.append({"resume_id": {"$in": resume_ids}})
    if not query_parts:
        return []

    query = {"$or": query_parts} if len(query_parts) > 1 else query_parts[0]

    cursor = _resumes.find(query)
    docs = await cursor.to_list(None)
    return [
        SimpleNamespace(
            id=doc["resume_id"],
            metadata={"name": doc["name"], "text": doc["text"]},
        )
        for doc in docs
    ]

async def update_resume(
    resume_id: str,
    name: str,
    text: str,
    skills: List[str] | None = None,
    location: str | None = None,
    years: int | None = None,
    tags: List[str] | None = None,
) -> int:
    old = await _resumes.find_one({"resume_id": resume_id})
    if not old:
        return 0

    update_fields = {"name": name, "text": text}
    if skills is not None:
        update_fields["skills"] = skills
    if location is not None:
        update_fields["location"] = location
    if years is not None:
        update_fields["years"] = years
    if tags is not None:
        update_fields["tags"] = tags

    res = await _resumes.update_one(
        {"resume_id": resume_id},
        {"$set": update_fields},
    )

    meta = {"name": name, "text": text if text.strip() else old.get("text", "")}
    if tags is not None:
        meta["tags"] = tags
    elif old.get("tags"):
        meta["tags"] = old.get("tags")
    if skills is not None:
        meta["skills"] = skills
    elif old.get("skills"):
        meta["skills"] = old.get("skills")
    if location is not None:
        meta["location"] = location
    elif old.get("location"):
        meta["location"] = old.get("location")
    if years is not None:
        meta["years"] = years
    elif old.get("years") is not None:
        meta["years"] = old.get("years")

    if text.strip() != old.get("text", ""):
        add_resume_to_pinecone(
            text,
            resume_id,
            meta,
            namespace="resumes",
        )
    else:
        from pinecone_utils import index
        index.upsert(
            vectors=[(resume_id, None, meta)],
            namespace="resumes",
        )

    return res.modified_count

async def delete_resume_by_id(resume_id: str) -> int:
    if await _guard("delete_resume_by_id"):
        return 0
    res = await _resumes.delete_one({"resume_id": resume_id.strip()})
    return res.deleted_count

# ------------------------------------------------------------------ #
# 3) chat and project helpers (migrated from db.py)                   #
# ------------------------------------------------------------------ #

async def chat_find_one(query: dict):
    if await _guard("chat_find_one"):
        return None
    return await chats.find_one(query)


async def chat_upsert(user_id: str, messages: List[dict]):
    if await _guard("chat_upsert"):
        return
    await chats.update_one(
        {"user_id": user_id},
        {"$set": {"messages": messages, "ts": datetime.utcnow()}},
        upsert=True,
    )


async def resumes_all():
    if await _guard("resumes_all"):
        return []
    cursor = resumes_collection.find()
    return await cursor.to_list(None)


def _build_filter_query(filters: dict | None) -> dict:
    query: dict = {}
    if not filters:
        return query
    if skill := filters.get("skill"):
        query["skills"] = skill
    if loc := filters.get("location"):
        query["location"] = loc
    min_y = filters.get("min_years")
    max_y = filters.get("max_years")
    years = {}
    if min_y is not None:
        years["$gte"] = int(min_y)
    if max_y is not None:
        years["$lte"] = int(max_y)
    if years:
        query["years"] = years
    return query


async def resumes_count(filters: dict | None = None) -> int:
    """Return total number of résumé documents."""
    if await _guard("resumes_count"):
        return 0
    query = _build_filter_query(filters)
    return await resumes_collection.count_documents(query)


async def resumes_page(page: int, per_page: int, filters: dict | None = None):
    """Return a single page of résumés sorted by _id descending."""
    if await _guard("resumes_page"):
        return []
    skip = max(0, (page - 1) * per_page)
    query = _build_filter_query(filters)
    cursor = (
        resumes_collection.find(query)
        .sort("_id", -1)
        .skip(skip)
        .limit(per_page)
    )
    return await cursor.to_list(None)


async def resumes_by_ids(id_list: List[str]):
    if await _guard("resumes_by_ids"):
        return []
    mongo_ids = [ObjectId(i) for i in id_list if ObjectId.is_valid(i)]
    resume_ids = [i for i in id_list if not ObjectId.is_valid(i)]

    query_parts = []
    if mongo_ids:
        query_parts.append({"_id": {"$in": mongo_ids}})
    if resume_ids:
        query_parts.append({"resume_id": {"$in": resume_ids}})
    if not query_parts:
        return []

    query = {"$or": query_parts} if len(query_parts) > 1 else query_parts[0]
    cursor = resumes_collection.find(query)
    docs = await cursor.to_list(None)
    return [
        SimpleNamespace(
            id=doc["resume_id"],
            metadata={"name": doc["name"], "text": doc["text"]},
        )
        for doc in docs
    ]


async def add_project_history(user_id: str, project: dict):
    """Append a project entry and update last_project."""
    if await _guard("add_project_history"):
        return
    project.setdefault("ts", datetime.utcnow())
    await chats.update_one(
        {"user_id": user_id},
        {"$push": {"projects": project}, "$set": {"last_project": project, "ts": datetime.utcnow()}},
        upsert=True,
    )


async def delete_project(user_id: str, ts_iso: str) -> int:
    """Delete a project by timestamp for the given user."""
    if await _guard("delete_project"):
        return 0
    try:
        ts = datetime.fromisoformat(ts_iso)
    except ValueError:
        return 0
    res = await chats.update_one(
        {"user_id": user_id},
        {"$pull": {"projects": {"ts": ts}}},
    )
    return res.modified_count
