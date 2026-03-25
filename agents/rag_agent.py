from rag.vector_store import search
from rag.llm_client import ask_llm

def explain(query, vec):
    docs = search(vec)
    prompt = f"{query}\nContext:{docs}"
    return ask_llm(prompt)
