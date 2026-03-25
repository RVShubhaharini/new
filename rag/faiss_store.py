import faiss
import numpy as np
import pickle
import os

dimension = 384
INDEX_FILE = "faiss_index.bin"
MEMORY_FILE = "faiss_memory.pkl"

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(dimension)

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "rb") as f:
        memory = pickle.load(f)
else:
    memory = []

def save_state():
    faiss.write_index(index, INDEX_FILE)
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)

def store_event(text):
    vector = np.random.rand(dimension).astype("float32")
    index.add(np.array([vector]))
    memory.append(text)
    save_state()

def retrieve_context():
    return "\n".join(memory[-10:])  # Return last 10 events for better context
