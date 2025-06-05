import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai
import logging

load_dotenv()

# Load API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "resumes-index"

# Create index if it doesn't exist
def _ensure_index():
    if INDEX_NAME not in pc.list_indexes().names():
        region = "-".join(PINECONE_ENV.split("-")[:-1])  # Extract region e.g. "europe-west4"
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region=region)
        )

_ensure_index()

# Connect to the index
index = pc.Index(INDEX_NAME)

# 🔹 Get embedding for a text string
def embed_text(text: str):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error("🛑 Error generating embedding: %s", e)
        return None

# 🔹 Add résumé vector to Pinecone
def add_resume_to_pinecone(text: str, candidate_id: str, metadata: dict, namespace: str = "resumes"):
    """
    Inserts or updates a résumé embedding into Pinecone under the given namespace.
    """
    vector = embed_text(text)
    if not vector or len(vector) != 1536:
        logging.error("🛑 Invalid embedding length: %s", len(vector) if vector else 'None')
        return

    try:
        logging.info("🟢 Upserting to namespace '%s': %s", namespace, candidate_id)
        index.upsert(
            vectors=[{
                "id": candidate_id,
                "values": vector,
                "metadata": metadata
            }],
            namespace=namespace
        )
        logging.info("✅ Upsert complete")
        stats = index.describe_index_stats()
        logging.info("📊 Index stats: %s", stats)
    except Exception as e:
        logging.error("🛑 Pinecone upsert failed: %s", e)

# 🔹 Search best candidates by matching to a project
def search_best_resumes(project_description: str, top_k: int = 5, namespace: str = "resumes"):
    """
    Queries Pinecone for the top_k resumes most similar to the project_description.
    Returns a list of matches with metadata.
    """
    vector = embed_text(project_description)
    if not vector:
        return []
    try:
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        return results.matches
    except Exception as e:
        logging.error("🛑 Pinecone search failed: %s", e)
        return []

# 🔹 Delete a résumé from Pinecone
def delete_resume_from_pinecone(resume_id: str, namespace: str = "resumes"):
    """
    Deletes the vector with the given resume_id from Pinecone namespace.
    """
    try:
        logging.info(
            "🟢 Deleting from Pinecone: %s namespace=%s",
            resume_id,
            namespace,
        )
        index.delete(ids=[resume_id], namespace=namespace)
        logging.info("✅ Deleted from Pinecone: %s", resume_id)
    except Exception as e:
        logging.error("🛑 Error deleting from Pinecone: %s - %s", resume_id, e)
