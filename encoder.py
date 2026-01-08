from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_text_embedding(text: str):
    emb = model.encode([text])[0]
    return np.array(emb)
