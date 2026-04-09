import os
import asyncio
import pypdf
import docx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Import your working brain functions, now including the new document ingestion
from rag_engine import answer_query, summarize_conversation, analyze_contract_image, process_and_store_document
from dotenv import load_dotenv

# Load the hidden environment variables
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fired when the user first starts the bot."""
    keyboard = [
        [
            InlineKeyboardButton("🇬🇧 English", callback_data='lang_en'),
            InlineKeyboardButton("🇮🇳 हिन्दी (Hindi)", callback_data='lang_hi')
        ],
        [
            InlineKeyboardButton("🇮🇳 मराठी (Marathi)", callback_data='lang_mr'),
            InlineKeyboardButton("🇮🇳 ગુજરાતી (Gujarati)", callback_data='lang_gu')
        ],
        [
            InlineKeyboardButton("🇮🇳 ಕನ್ನಡ (Kannada)", callback_data='lang_kn'),
            InlineKeyboardButton("🇮🇳 தமிழ் (Tamil)", callback_data='lang_ta')
        ],
        [
            InlineKeyboardButton("🇮🇳 తెలుగు (Telugu)", callback_data='lang_te')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    welcome_msg = (
        "👋 Welcome to the Contract Buddy!\n"
        "Contract Buddy में आपका स्वागत है।\n\n"
        "Please select your preferred language / कृपया अपनी भाषा चुनें:"
    )
    await update.message.reply_text(welcome_msg, reply_markup=reply_markup)

async def language_selection_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Catches the button click and saves the language."""
    query = update.callback_query
    await query.answer() 
    
    selected_lang = query.data.split('_')[1] 
    context.user_data['language'] = selected_lang
    
    # Confirm selection and prompt for contract upload/query
    if selected_lang == 'en':
        await query.edit_message_text(text="✅ Language set to English.\n\nHere is what I can do:\n🔹 /ask <query> — Ask questions\n🔹 Upload a document/image — Send a contract for processing\n🔹 /summarize — Get a session summary")
    elif selected_lang == 'hi':
        await query.edit_message_text(text="✅ भाषा हिंदी सेट कर दी गई है।\n\nमैं ये कर सकता हूँ:\n🔹 /ask <प्रश्न> — प्रश्न पूछें\n🔹 दस्तावेज़/छवि अपलोड करें — अनुबंध भेजें\n🔹 /summarize — सारांश प्राप्त करें")
    elif selected_lang == 'mr':
         await query.edit_message_text(text="✅ भाषा मराठी सेट केली आहे.\n\nमी हे करू शकतो:\n🔹 /ask <प्रश्न> — प्रश्न विचारा\n🔹 दस्तऐवज/प्रतिमा अपलोड करा — करार पाठवा\n🔹 /summarize — सारांश मिळवा")
    elif selected_lang == 'gu':
         await query.edit_message_text(text="✅ ભાષા ગુજરાતી સેટ થઈ ગઈ છે.\n\nહું આ કરી શકું છું:\n🔹 /ask <પ્રશ્ન> — પ્રશ્નો પૂછો\n🔹 દસ્તાવેજ/છબી અપલોડ કરો — કરાર મોકલો\n🔹 /summarize — સારાંશ મેળવો")
    elif selected_lang == 'kn':
         await query.edit_message_text(text="✅ ಭಾಷೆಯನ್ನು ಕನ್ನಡಕ್ಕೆ ಹೊಂದಿಸಲಾಗಿದೆ.\n\nನಾನು ಏನು ಮಾಡಬಲ್ಲೆ:\n🔹 /ask <ಪ್ರಶ್ನೆ> — ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ\n🔹 ಡಾಕ್ಯುಮೆಂಟ್/ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ — ಒಪ್ಪಂದವನ್ನು ಕಳುಹಿಸಿ\n🔹 /summarize — ಸಾರಾಂಶವನ್ನು ಪಡೆಯಿರಿ")
    elif selected_lang == 'ta':
         await query.edit_message_text(text="✅ மொழி தமிழ் என அமைக்கப்பட்டுள்ளது.\n\nநான் என்ன செய்ய முடியும்:\n🔹 /ask <கேள்வி> — கேள்விகளைக் கேளுங்கள்\n🔹 ஆவணம்/படத்தை பதிவேற்றவும் — ஒப்பந்தத்தை அனுப்பவும்\n🔹 /summarize — சுருக்கத்தைப் பெறவும்")
    elif selected_lang == 'te':
         await query.edit_message_text(text="✅ భాష తెలుగుగా సెట్ చేయబడింది.\n\nనేను ఏమి చేయగలను:\n🔹 /ask <ప్రశ్న> — ప్రశ్నలు అడగండి\n🔹 పత్రం/చిత్రాన్ని అప్‌లోడ్ చేయండి — ఒప్పందాన్ని పంపండి\n🔹 /summarize — సారాంశాన్ని పొందండి")

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    chat_id = update.effective_chat.id 
    
    if not query:
        await update.message.reply_text("Please provide a query. Example: /ask What is the liability limit?")
        return
    
    # Retrieve the user's preferred language (defaulting to English)
    user_lang = context.user_data.get('language', 'en')
    
    status_message = await update.message.reply_text(f"🔍 Processing query...\n🧠 Consulting AI Engine...")

    try:
        # Pass the language tag into the RAG engine so it can translate if needed
        answer = await asyncio.to_thread(answer_query, query, chat_id, user_lang)
        await status_message.edit_text(answer)
    except Exception as e:
        await status_message.edit_text(f"❌ An error occurred in the AI engine: {str(e)}")

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    # Grab the language preference from memory
    user_lang = context.user_data.get('language', 'en')
    
    status_message = await update.message.reply_text("📝 Processing summary... / सारांश तैयार किया जा रहा है...")
    
    try:
        # Pass the user_lang into the summarize_conversation function!
        summary = await asyncio.to_thread(summarize_conversation, chat_id, user_lang)
        await status_message.edit_text(f"**Executive Summary:**\n\n{summary}")
    except Exception as e:
        await status_message.edit_text(f"❌ Error generating summary: {str(e)}")

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_message = await update.message.reply_text("🖼️ Image received! Downloading...")
    
    try:
        photo_file = await update.message.photo[-1].get_file()
        
        os.makedirs("temp_images", exist_ok=True)
        image_path = os.path.join("temp_images", f"{update.message.message_id}.jpg")
        
        await photo_file.download_to_drive(image_path)
        await status_message.edit_text("👁️ Analyzing document... (Extracting text and translating if needed)")
        
        # Retrieve user language and call the brain!
        user_lang = context.user_data.get('language', 'en')
        analysis = await asyncio.to_thread(analyze_contract_image, image_path, user_lang)
        
        os.remove(image_path)
        await status_message.edit_text(f"**🔍 Contract Analysis:**\n\n{analysis}")
        
    except Exception as e:
        await status_message.edit_text(f"❌ Error analyzing image: {str(e)}")

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Catches document uploads (.pdf, .docx, .md, .txt), reads them, and pushes to Cloud."""
    telegram_user_id = update.effective_chat.id
    document = update.message.document
    
    file_name = document.file_name.lower()
    allowed_extensions = ['.md', '.txt', '.pdf', '.docx']
    
    # 1. Gatekeeper Check
    if not any(file_name.endswith(ext) for ext in allowed_extensions):
        await update.message.reply_text("❌ Please upload a PDF, Word (.docx), Markdown (.md), or Text (.txt) file.")
        return

    status_message = await update.message.reply_text(f"📥 Received '{document.file_name}'. Downloading...")
    
    try:
        # 2. Download the file from Telegram servers
        file_obj = await document.get_file()
        os.makedirs("temp_docs", exist_ok=True)
        file_path = os.path.join("temp_docs", document.file_name)
        
        await file_obj.download_to_drive(file_path)
        await status_message.edit_text(f"⚙️ Extracting text from {document.file_name}...")
        
        # 3. Dynamic Text Extraction based on file type
        content = ""
        
        if file_name.endswith('.md') or file_name.endswith('.txt'):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
        elif file_name.endswith('.pdf'):
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    content += extracted_text + "\n"
                    
        elif file_name.endswith('.docx'):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                content += para.text + "\n"
        
        # 4. Safety Check (e.g., if a PDF is just scanned images with no embedded text)
        if len(content.strip()) < 10:
             await status_message.edit_text("❌ Could not extract text. Please ensure the PDF/Word file contains actual text, not just scanned images.")
             os.remove(file_path)
             return

        await status_message.edit_text("☁️ Text extracted! Slicing and securing in your Cloud database...")

        # 5. Push to Supabase using our multi-tenant function
        result = await asyncio.to_thread(process_and_store_document, content, document.file_name, telegram_user_id)
        
        # 6. Clean up the local temp file
        os.remove(file_path)
        
        await status_message.edit_text(result)
        
    except Exception as e:
        await status_message.edit_text(f"❌ Error processing document: {str(e)}")

class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is awake and running!")

def run_dummy_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), DummyHandler)
    print(f"🌐 Heartbeat server started on port {port}")
    server.serve_forever()

def main():
    print("Starting AVIVO Contract Bot...")
    
    # 1. Start the Render Heartbeat in the background
    threading.Thread(target=run_dummy_server, daemon=True).start()
    
    # 2. Start the Telegram Bot
    app = Application.builder().token(BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("summarize", summarize_command))
    app.add_handler(CallbackQueryHandler(language_selection_handler))
    
    # Media and Document Handlers
    app.add_handler(MessageHandler(filters.PHOTO, image_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))

    # Start the bot
    print("Bot is polling... Press Ctrl+C to stop.")
    app.run_polling(poll_interval=1.0)

if __name__ == "__main__":
    main()