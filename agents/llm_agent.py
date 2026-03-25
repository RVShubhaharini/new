from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME
from rag.faiss_store import retrieve_context

from config import GROQ_API_KEY, MODEL_NAME
from groq import Groq

client = Groq(api_key=GROQ_API_KEY)

def explain_system(question):

    context = retrieve_context()

    if not context.strip():
        context = "No system logs available yet. The system has not performed any unlearning operations."

    prompt = f"""
You are a factual Banking AI Compliance Assistant. Your job is to explain the system status based ONLY on the provided logs.

STRICT RULES:
1. ONLY use information present in the "System Logs" section below.
2. DO NOT invent, hallucinate, or make up any numbers, file sizes (GB/MB), or model names (Model 001, etc.) that are not in the logs.
3. If the logs do not contain the specific information asked (like "total size"), state clearly: "I do not have that information in my current logs."
4. Do not roleplay as having performed actions you didn't do.

System Logs:
{context}

User Question:
{question}

Answer:
"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Reduce creativity to minimum
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"
