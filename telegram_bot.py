import os
import asyncio
import matplotlib
matplotlib.use('Agg')
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load the main pipeline
from orchestrator.langgraph_flow import graph

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# Optional: To restrict who can use the bot, set an AUTHORIZED_CHAT_ID in .env
AUTHORIZED_CHAT_ID = os.getenv("AUTHORIZED_CHAT_ID")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 👋\n\n"
        f"I am the Agentic Machine Unlearning Bot.\n\n"
        f"Use /delete <customer_id> to completely remove a user's data from our AI models.\n\n"
        f"Example: /delete 123"
    )
    await update.message.reply_text(welcome_msg)

async def check_auth(update: Update) -> bool:
    """Check if the user is authorized to use the bot."""
    if not AUTHORIZED_CHAT_ID:
        return True # If no ID is set, allow anyone (for testing)
        
    chat_id = str(update.effective_chat.id)
    if chat_id != AUTHORIZED_CHAT_ID:
        await update.message.reply_text("⛔ Unauthorized. You do not have permission to trigger unlearning pipelines.")
        return False
    return True

async def delete_record(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trigger the LangGraph unlearning pipeline."""
    
    # 1. Authorization check
    if not await check_auth(update):
        return

    # 2. Extract Customer ID
    if not context.args:
        await update.message.reply_text("⚠️ Please provide a Customer ID.\nUsage: /delete <customer_id>")
        return
        
    customer_id = context.args[0]
    
    # 3. Acknowledge Receipt
    status_msg = await update.message.reply_text(
        f"🔍 Initiating unlearning sequence for Customer ID: <b>{customer_id}</b>...\n\n"
        f"<i>Please wait, this will take a moment as the multi-agent pipeline processes the request.</i>",
        parse_mode='HTML'
    )
    
    # 4. Run the Pipeline
    try:
        # Run graph in a separate thread so it doesn't block the async event loop
        loop = asyncio.get_event_loop()
        
        # Determine if we should force ML strategy via an argument (e.g., /delete 123 ml)
        force_ml = False
        if len(context.args) > 1 and context.args[1].lower() == 'ml':
            force_ml = True
            
        def run_graph_sync():
            return graph.invoke({
                "forget_size": 1, 
                "customer_id": customer_id,
                "email": "shubhaharinirv@gmail.com", # Defaulting to the admin email in .env for certificates
                "force_ml": force_ml
            })
            
        # Add a timeout to prevent the thread from hanging indefinitely if Render kills it
        task = loop.run_in_executor(None, run_graph_sync)
        
        # We wait up to 300 seconds (5 mins) for the pipeline to finish 
        # (Render free tier kills idle/heavy loops aggressively)
        try:
            result = await asyncio.wait_for(task, timeout=300.0)
        except asyncio.TimeoutError:
            await status_msg.edit_text("⚠️ <b>Timeout Error:</b> The server took too long to process this request. The pipeline may have run out of memory or crashed on the cloud server.", parse_mode='HTML')
            return
        
        # 5. Format the Result
        final_status = result.get('status_message', 'Completed')
        model_used = result.get('model_type', 'Unknown')
        accuracy = result.get('accuracy', 'N/A')
        
        if "FAILED" in final_status or "NOT FOUND" in final_status or "BLOCKED" in final_status:
            icon = "❌"
        elif "ALREADY" in final_status:
            icon = "ℹ️"
        else:
            icon = "✅"
            
        reply_msg = (
            f"{icon} <b>Pipeline Completed</b>\n\n"
            f"<b>Customer ID:</b> {customer_id}\n"
            f"<b>Model Strategy:</b> {model_used}\n"
            f"<b>Status:</b> {final_status}\n"
        )
        
        if icon == "✅":
            reply_msg += f"<b>Post-Unlearning Accuracy:</b> {float(accuracy):.2f}%\n"
            reply_msg += f"<i>A detailed compliance certificate has been emailed to the administrator.</i>"
            
        await status_msg.edit_text(reply_msg, parse_mode='HTML')
        
    except Exception as e:
        error_msg = f"‼️ <b>System Error during unlearning:</b>\n<code>{str(e)}</code>\n\n<i>Note: If running on a free cloud tier, this usually means the server ran out of memory while loading the models.</i>"
        await status_msg.edit_text(error_msg, parse_mode='HTML')
        print(f"Pipeline Error: {e}")

def main() -> None:
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN is not set in your .env file!")
        return
        
    # Create the Application and pass it your bot's token with higher timeouts for Render.
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .build()
    )

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("delete", delete_record))

    # Run the bot until the user presses Ctrl-C
    print("🤖 Telegram Bot is running! Waiting for messages...")
    
    # We must allow the thread to run without catching signals 
    # since signals can only be caught in the main thread!
    application.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None)

if __name__ == "__main__":
    main()
