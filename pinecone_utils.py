from pinecone import Pinecone, ServerlessSpec
import openai

from settings import settings
from logger import logger

ENV = settings.ENV

openai.api_key = settings.OPENAI_API_KEY
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# picks resumes-index for prod, resumes-dev for dev unless overridden
INDEX_NAME = settings.pinecone_index
PINECONE_ENV = settings.PINECONE_ENV


def _ensure_index():
    if INDEX_NAME not in pc.list_indexes().names():
        region = "-".join(PINECONE_ENV.split("-")[:-1])  # eg "europe-west4"
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region=region)
        )

_ensure_index()
index = pc.Index(INDEX_NAME)

# ------------------------------------------------------------------ #
# 2) helpers                                                         #
# ------------------------------------------------------------------ #


def embed_text(text: str):
    """Return an OpenAI embedding or None on failure."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("Error generating embedding: %s", e)
        return None


DEFAULT_NAMESPACE = "resumes"  # namespace stays the same in both envs


def add_resume_to_pinecone(
    text: str,
    candidate_id: str,
    metadata: dict | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    tags: list[str] | None = None,
):
    """Insert or update a résumé vector in Pinecone."""
    metadata = metadata.copy() if metadata else {}
    if tags is not None:
        metadata["tags"] = tags
    vector = embed_text(text)
    if not vector or len(vector) != 1536:
        logger.warning(
            "Invalid embedding length: %s",
            len(vector) if vector else "None",
        )
        return

    try:
        logger.info(
            "Upserting to index '%s' namespace '%s': %s",
            INDEX_NAME,
            namespace,
            candidate_id,
        )
        index.upsert(
            vectors=[{
                "id": candidate_id,
                "values": vector,
                "metadata": metadata,
            }],
            namespace=namespace,
        )
        logger.info("Upsert complete")
    except Exception as e:
        logger.error("Pinecone upsert failed: %s", e)


def search_best_resumes(
    project_description: str,
    top_k: int = 5,
    namespace: str = DEFAULT_NAMESPACE,
):
    """Return top_k matches for the given description."""
    vector = embed_text(project_description)
    if not vector:
        return []
    try:
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )
        return results.matches
    except Exception as e:
        logger.error("Pinecone search failed: %s", e)
        return []


def delete_resume_from_pinecone(resume_id: str, namespace: str = DEFAULT_NAMESPACE):
    """Delete a résumé vector by id."""
    try:
        logger.info(
            "Deleting from index '%s' namespace '%s': %s",
            INDEX_NAME,
            namespace,
            resume_id,
        )
        index.delete(ids=[resume_id], namespace=namespace)
        logger.info("Deleted from Pinecone: %s", resume_id)
    except Exception as e:
        logger.error("Error deleting from Pinecone: %s - %s", resume_id, e)
