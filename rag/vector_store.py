import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatL2(dimension)

documents = []
embeddings = []

def add_to_store(vec, text):
    embeddings.append(vec)
    documents.append(text)
    index.add(np.array([vec]).astype("float32"))

def search(vec):
    D, I = index.search(np.array([vec]).astype("float32"), k=3)
    return [documents[i] for i in I[0]]
