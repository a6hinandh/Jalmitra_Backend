import os
import torch
from sentence_transformers import SentenceTransformer

def download():
    model_name = os.getenv("EMBED_MODEL", "all-mpnet-base-v2")
    print(f"📥 Downloading and caching model '{model_name}' during build phase...")
    SentenceTransformer(model_name, model_kwargs={"dtype": torch.bfloat16})
    print("✅ Model downloaded and cached successfully.")

if __name__ == "__main__":
    download()
