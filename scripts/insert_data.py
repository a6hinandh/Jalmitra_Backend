# insert_data.py

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embeddings import load_data, create_embeddings
from pinecone_setup import initialize_pinecone

def insert_embeddings(index, ids, texts,metadatas, embeddings):
    # Insert embeddings in batches for better performance
    for i in range(len(embeddings)):
        # Merge metadata with text
        meta = metadatas[i].copy()
        meta["text"] = texts[i]

        vector = embeddings[i].tolist()
        index.upsert([(ids[i], vector, meta)])
        print(f"✅ Inserted {ids[i]}")

    print(f"🎉 Successfully inserted {len(embeddings)} total entries into Pinecone.")

if __name__ == "__main__":
    API_KEY = "pcsk_3Fx1Dr_T9VPQFX4py9BfoeZhDgEJJ74s5SuNQYVgtnQWufp9mwsDQWrAkSsoBWDHbd5wm7"
    json_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "states", "KERALA.json")

    print("🚀 Starting data insertion pipeline...")

    # Initialize Pinecone
    index = initialize_pinecone(API_KEY)

    # Load and process data
    print("📂 Loading data...")
    ids, texts, metadatas = load_data(json_file)
    print(f"📊 Loaded {len(texts)} entries")

    # Create embeddings
    print("🧠 Creating embeddings...")
    embeddings = create_embeddings(texts)

    # Insert into Pinecone
    print("📤 Inserting into Pinecone...")
    insert_embeddings(index, ids, texts,metadatas, embeddings)
