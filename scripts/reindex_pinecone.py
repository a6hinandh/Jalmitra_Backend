"""
reindex_pinecone.py — one-shot re-index for a changed embedding model.

Switching EMBED_MODEL changes the vector dimension, which the Pinecone index is
locked to. This script (re)creates the index at the model's dimension and
re-embeds the local data. Run it locally once after changing EMBED_MODEL.

Reads config from env / .env (NO hardcoded keys):
    PINECONE_API_KEY   (required)
    PINECONE_INDEX     (default "gw-index")
    EMBED_MODEL        (default "all-MiniLM-L6-v2")

Usage:
    python scripts/reindex_pinecone.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from core.embeddings import load_data

load_dotenv()

API_KEY    = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "gw-index")
MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

if not API_KEY:
    sys.exit("PINECONE_API_KEY is required (set it in your environment or .env)")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data to embed: state-level (india.json) + Kerala districts (the only district
# coverage the app exposes). Add more state folders here as coverage grows.
DATA_FILES = [
    os.path.join(BACKEND_DIR, "data", "output", "india.json"),
    os.path.join(BACKEND_DIR, "data", "states", "KERALA.json"),
]


def main():
    print(f"🧠 Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"   -> dimension = {dim}")

    pc = Pinecone(api_key=API_KEY)
    existing = pc.list_indexes()

    # If the index exists at the wrong dimension, delete and recreate it.
    match = next((i for i in existing if i["name"] == INDEX_NAME), None)
    if match is not None and match["dimension"] != dim:
        print(f"♻️  Index '{INDEX_NAME}' has dimension {match['dimension']} != {dim}; deleting.")
        pc.delete_index(INDEX_NAME)
        match = None
        time.sleep(5)

    if match is None:
        print(f"📦 Creating index '{INDEX_NAME}' (dim={dim}, cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until the index is ready.
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    index = pc.Index(INDEX_NAME)

    total = 0
    for path in DATA_FILES:
        if not os.path.exists(path):
            print(f"⚠️  Skipping missing file: {path}")
            continue
        print(f"📂 Embedding {os.path.basename(path)} ...")
        ids, texts, metadatas = load_data(path)
        embeddings = model.encode(texts, show_progress_bar=True)

        vectors = []
        for i in range(len(embeddings)):
            meta = metadatas[i].copy()
            meta["text"] = texts[i]
            # Pinecone rejects None metadata values — drop them.
            meta = {k: v for k, v in meta.items() if v is not None}
            vectors.append((ids[i], embeddings[i].tolist(), meta))

        # Upsert in batches of 100.
        for b in range(0, len(vectors), 100):
            index.upsert(vectors[b:b + 100])
        total += len(vectors)
        print(f"   -> upserted {len(vectors)} vectors")

    print(f"🎉 Done. {total} vectors in '{INDEX_NAME}' (dim={dim}).")


if __name__ == "__main__":
    main()
