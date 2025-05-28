import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai

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
        print("🛑 Error generating embedding:", e)
        return None

# 🔹 Add résumé vector to Pinecone
def add_resume_to_pinecone(text: str, candidate_id: str, metadata: dict, namespace: str = "resumes"):
    """
    Inserts or updates a résumé embedding into Pinecone under the given namespace.
    """
    vector = embed_text(text)
    if not vector or len(vector) != 1536:
        print(f"🛑 Invalid embedding length: {len(vector) if vector else 'None'}")
        return

    try:
        print(f"🟢 Upserting to namespace '{namespace}': {candidate_id}")
        index.upsert(
            vectors=[{
                "id": candidate_id,
                "values": vector,
                "metadata": metadata
            }],
            namespace=namespace
        )
        print("✅ Upsert complete")
        stats = index.describe_index_stats()
        print("📊 Index stats:", stats)
    except Exception as e:
        print("🛑 Pinecone upsert failed:", e)

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
        print("🛑 Pinecone search failed:", e)
        return []

# 🔹 Delete a résumé from Pinecone
def delete_resume_from_pinecone(resume_id: str, namespace: str = "resumes"):
    """
    Deletes the vector with the given resume_id from Pinecone namespace.
    """
    try:
        print(f"🟢 Deleting from Pinecone: {resume_id} namespace={namespace}")
        index.delete(ids=[resume_id], namespace=namespace)
        print(f"✅ Deleted from Pinecone: {resume_id}")
    except Exception as e:
        print(f"🛑 Error deleting from Pinecone: {resume_id} - {e}")
