import os
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Import your working brain, summary, AND vision functions!
from rag_engine import answer_query, summarize_conversation, analyze_contract_image

from dotenv import load_dotenv

# Load the hidden environment variables
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = (
        "👋 Welcome to the AVIVO Contract Intelligence Bot!\n\n"
        "I can help you review and analyze legal documents. Here is what I can do:\n"
        "🔹 /ask <query> — Ask questions about our stored contracts\n"
        "🔹 Upload an image — Send a scanned contract for risk tagging\n"
        "🔹 /summarize — Get an executive summary of our session\n\n"
        "How can I assist you today?"
    )
    await update.message.reply_text(welcome_msg)

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    chat_id = update.effective_chat.id 
    
    if not query:
        await update.message.reply_text("Please provide a query. Example: /ask What is the liability limit?")
        return
    
    status_message = await update.message.reply_text(f"🔍 Searching the knowledge base for: '{query}'...\n🧠 Consulting DeepSeek-R1...")

    try:
        answer = await asyncio.to_thread(answer_query, query, chat_id)
        await status_message.edit_text(answer)
    except Exception as e:
        await status_message.edit_text(f"❌ An error occurred in the AI engine: {str(e)}")

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    status_message = await update.message.reply_text("📝 Reading our session memory and generating a summary...")
    
    try:
        summary = await asyncio.to_thread(summarize_conversation, chat_id)
        await status_message.edit_text(f"**Executive Summary:**\n\n{summary}")
    except Exception as e:
        await status_message.edit_text(f"❌ Error generating summary: {str(e)}")

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1. Send immediate feedback
    status_message = await update.message.reply_text("🖼️ Image received! Downloading...")
    
    try:
        # 2. Grab the highest resolution version of the photo sent
        photo_file = await update.message.photo[-1].get_file()
        
        # 3. Create a safe temporary folder if it doesn't exist
        os.makedirs("temp_images", exist_ok=True)
        image_path = os.path.join("temp_images", f"{update.message.message_id}.jpg")
        
        # Download it to your PC
        await photo_file.download_to_drive(image_path)
        await status_message.edit_text("👁️ Contract downloaded. Analyzing with Llama-3.2-Vision...\n*(This takes ~15 seconds as it swaps models in your VRAM)*")
        
        # 4. Call the brain!
        analysis = await asyncio.to_thread(analyze_contract_image, image_path)
        
        # 5. Clean up the hard drive & send the result
        os.remove(image_path)
        await status_message.edit_text(f"**🔍 Vision Risk Analysis:**\n\n{analysis}")
        
    except Exception as e:
        await status_message.edit_text(f"❌ Error analyzing image: {str(e)}")

def main():
    print("Starting AVIVO Contract Bot...")
    app = Application.builder().token(BOT_TOKEN).build()

    # Register the command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("summarize", summarize_command))
    
    # Register the image handler (triggers on photo uploads)
    app.add_handler(MessageHandler(filters.PHOTO, image_handler))

    # Start the bot
    print("Bot is polling... Press Ctrl+C to stop.")
    app.run_polling(poll_interval=1.0)

if __name__ == "__main__":
    main()