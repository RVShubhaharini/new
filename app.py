import os
import asyncio
import threading
from fastapi import FastAPI
from telegram_bot import main as run_bot_main

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Telegram Bot Running"}

def run_bot():
    # Python-Telegram-Bot v20 heavily relies on an asyncio event loop.
    # Because we are inside a new background thread, there isn't one by default!
    # We must create and set the event loop before calling the bot's main loop.
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    run_bot_main()

# Start the bot in a background daemon thread instantly when Uvicorn loads this file
threading.Thread(target=run_bot, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
