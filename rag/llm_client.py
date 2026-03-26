import os
from groq import Groq

# Ensure API key is available
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key, timeout=15.0) if api_key else None

def ask_llm(prompt):
    if not client:
        return "Error: GROQ_API_KEY not found in environment."
        
    chat = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat.choices[0].message.content
