import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai

env_file = ".env.production" if os.getenv("ENV", "development").lower() == "production" else ".env.development"
try:
    load_dotenv(env_file)
except TypeError:  # tests may stub load_dotenv without args
    load_dotenv()

ENV = os.getenv("ENV", "development").lower()

openai.api_key = os.environ["OPENAI_API_KEY"]
pc  = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# picks resumes-index for prod, resumes-dev for dev unless overridden
INDEX_NAME = os.getenv(
    "PINECONE_INDEX",
    "resumes-index" if ENV == "production" else "resumes-dev"
)
PINECONE_ENV = os.environ["PINECONE_ENV"]

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
        print("🛑 Error generating embedding:", e)
        return None


DEFAULT_NAMESPACE = "resumes"  # namespace stays the same in both envs


def add_resume_to_pinecone(
    text: str,
    candidate_id: str,
    metadata: dict,
    namespace: str = DEFAULT_NAMESPACE,
):
    """Insert or update a résumé vector in Pinecone."""
    vector = embed_text(text)
    if not vector or len(vector) != 1536:
        print(f"🛑 Invalid embedding length: {len(vector) if vector else 'None'}")
        return

    try:
        print(f"🟢 Upserting to index '{INDEX_NAME}' namespace '{namespace}': {candidate_id}")
        index.upsert(
            vectors=[{
                "id": candidate_id,
                "values": vector,
                "metadata": metadata,
            }],
            namespace=namespace,
        )
        print("✅ Upsert complete")
    except Exception as e:
        print("🛑 Pinecone upsert failed:", e)


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
        print("🛑 Pinecone search failed:", e)
        return []


def delete_resume_from_pinecone(resume_id: str, namespace: str = DEFAULT_NAMESPACE):
    """Delete a résumé vector by id."""
    try:
        print(f"🟢 Deleting from index '{INDEX_NAME}' namespace '{namespace}': {resume_id}")
        index.delete(ids=[resume_id], namespace=namespace)
        print("✅ Deleted from Pinecone: {resume_id}")
    except Exception as e:
        print(f"🛑 Error deleting from Pinecone: {resume_id} - {e}")
