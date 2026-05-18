# backend/embedder.py
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

def embed_text(text: str) -> list:
    return model.encode(text).tolist()

def embed_texts(texts: list) -> list:
    return model.encode(texts, batch_size=32, show_progress_bar=True).tolist()