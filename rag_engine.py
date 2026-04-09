import os
import re
import time
import requests
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from langdetect import detect
from langchain_text_splitters import MarkdownTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------
# 1. CLOUD SECRETS & INITIALIZATION
# ---------------------------------------------------------
load_dotenv()
SUPABASE_URI = os.getenv("SUPABASE_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

if not all([SUPABASE_URI, GROQ_API_KEY, HF_TOKEN, OCR_SPACE_API_KEY]):
    print("⚠️ WARNING: One or more Cloud API keys are missing from your .env file!")

print("🔌 Booting Cloud Connections...")

# Initialize Groq Cloud LLMs (Lightning Fast)
# deepseek-r1-distill-llama-70b is vastly more powerful than the local 8b model
deepseek_llm = ChatGroq(api_key=GROQ_API_KEY, model_name="openai/gpt-oss-120b", temperature=0)
# We use Llama-3.3-70b for fast, highly accurate translations
translator_llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)

# --- MEMORY BUFFER ---
chat_histories = {}
MAX_HISTORY = 4

LANGUAGE_MAP = {
    'hi': 'Hindi', 'mr': 'Marathi', 'gu': 'Gujarati',
    'kn': 'Kannada', 'ta': 'Tamil', 'te': 'Telugu'
}

# ---------------------------------------------------------
# 2. CLOUD API HELPER FUNCTIONS
# ---------------------------------------------------------
def get_cloud_embedding(text):
    """Fetches vector embeddings from HuggingFace Cloud API."""
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    response = requests.post(api_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
    
    if response.status_code != 200:
        print(f"⚠️ HF API Error: {response.text}")
        return [0.0] * 384 # Fallback to empty vector on error
        
    res = response.json()
    # Handle HF returning a list of lists
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        return res[0]
    return res

def extract_text_from_image_cloud(image_path):
    """Sends image to OCR.space for lightweight text extraction."""
    print("🖼️ Sending image to OCR.space Cloud...")
    payload = {
        'isOverlayRequired': False,
        'apikey': OCR_SPACE_API_KEY,
        'language': 'eng',
        'OCREngine': 2 # Engine 2 is highly optimized for document text
    }
    with open(image_path, 'rb') as f:
        response = requests.post('https://api.ocr.space/parse/image', files={image_path: f}, data=payload)
    
    result = response.json()
    if result.get('IsErroredOnProcessing'):
        print(f"❌ OCR Error: {result.get('ErrorMessage')}")
        return ""
        
    parsed_results = result.get('ParsedResults', [])
    extracted_text = " ".join([res.get('ParsedText', '') for res in parsed_results])
    return extracted_text.strip()

def translate_text(text, target_language="English"):
    """Uses Groq Llama-3 for high-speed legal translation."""
    prompt = f"""You are a highly accurate legal translator.
    Translate the following text into {target_language}.
    Do not summarize, do not add commentary. Just provide the exact translation.

    TEXT:
    {text}

    TRANSLATION:"""
    response = translator_llm.invoke(prompt)
    return response.content.strip()

# ---------------------------------------------------------
# 3. CORE CLOUD DB OPERATIONS (MULTI-TENANT)
# ---------------------------------------------------------
def process_and_store_document(text, document_name, telegram_user_id):
    """Chunks live text from Telegram and stores it in Supabase with the User's ID."""
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    
    try:
        conn = psycopg2.connect(SUPABASE_URI)
        register_vector(conn)
        cursor = conn.cursor()
        
        print(f"📦 Chunking and storing {document_name} for User {telegram_user_id}...")
        chunks = splitter.create_documents([text])
        
        insert_query = """
            INSERT INTO contract_chunks (document_name, chunk_text, chunk_embedding, telegram_user_id)
            VALUES (%s, %s, %s, %s)
        """
        
        success_count = 0
        for chunk in chunks:
            chunk_text = chunk.page_content
            # NOW USING CLOUD EMBEDDING
            embedding = get_cloud_embedding(chunk_text) 
            
            try:
                cursor.execute(insert_query, (document_name, chunk_text, embedding, str(telegram_user_id)))
                success_count += 1
            except Exception as e:
                print(f"⚠️ Error uploading chunk: {e}")
                conn.rollback()
                continue
                
        conn.commit()
        cursor.close()
        conn.close()
        return f"✅ Successfully securely stored {success_count} chunks from {document_name}."
        
    except Exception as e:
        print(f"❌ Cloud DB Storage Error: {e}")
        return "❌ Failed to store document securely."

def retrieve_chunks(query, telegram_user_id, top_k=3):
    """Embeds the query and fetches the top-k most relevant chunks for a SPECIFIC USER."""
    # NOW USING CLOUD EMBEDDING
    query_embedding = get_cloud_embedding(query)
    
    vector_string = f"[{','.join(map(str, query_embedding))}]"
    
    try:
        conn = psycopg2.connect(SUPABASE_URI)
        register_vector(conn)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT document_name, chunk_text 
            FROM contract_chunks 
            WHERE telegram_user_id = %s
            ORDER BY chunk_embedding <=> %s::vector 
            LIMIT %s
        """, (str(telegram_user_id), vector_string, top_k))
        
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ Cloud DB Retrieval Error: {e}")
        return []

# ---------------------------------------------------------
# 4. BOT PIPELINE FUNCTIONS
# ---------------------------------------------------------
def answer_query(query, chat_id="default", user_lang="en"):
    """Retrieves context from Cloud, injects memory, and translates."""
    
    if user_lang != 'en':
        print(f"🔄 Translating query from {user_lang} to English...")
        english_query = translate_text(query, "English")
    else:
        english_query = query

    retrieved_docs = retrieve_chunks(english_query, chat_id)
    print(f"[DEBUG] Retrieved {len(retrieved_docs)} chunks from CLOUD for user {chat_id}.")
    
    if not retrieved_docs:
        return "I couldn't find any relevant information in your uploaded contracts."
        
    context_parts = []
    for row in retrieved_docs:
        doc_name = row[0]
        text = row[1]
        context_parts.append(f"[Source: {doc_name}]\n{text}\n")
        
    context_string = "\n".join(context_parts)
    
    history = chat_histories.get(chat_id, [])
    history_string = ""
    if history:
        history_string = "Previous Conversation Context:\n"
        for msg in history:
            history_string += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "query"],
        template=(
            "You are an expert Contract Intelligence Assistant.\n"
            "Use the provided contract excerpts and our previous conversation history to answer the user's question. "
            "You MUST cite the exact source document name in your answer based on the provided [Source: ...]. "
            "If the context does not contain the answer, state that explicitly.\n\n"
            "{chat_history}\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Expert Answer:"
        )
    )
    
    prompt = prompt_template.format(chat_history=history_string, context=context_string, query=english_query)
    print(f"🧠 Thinking... (Querying Groq DeepSeek-70b for user {chat_id})")
    
    # LangChain ChatGroq returns an AIMessage object, so we extract .content
    raw_response = deepseek_llm.invoke(prompt).content
    clean_english_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    chat_histories[chat_id].append({"role": "user", "content": english_query})
    chat_histories[chat_id].append({"role": "assistant", "content": clean_english_response})
    
    if len(chat_histories[chat_id]) > MAX_HISTORY * 2:
        chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY * 2:]
        
    if user_lang != 'en':
        target_lang = LANGUAGE_MAP.get(user_lang, 'English')
        print(f"🔄 Translating final answer back to {target_lang}...")
        return translate_text(clean_english_response, target_lang)

    return clean_english_response

def analyze_contract_image(image_path, user_lang="en"):
    """Extracts text via Cloud OCR, normalizes, analyzes, and translates back."""
    extracted_text = extract_text_from_image_cloud(image_path)
    
    if len(extracted_text) < 10:
        return "❌ Could not extract enough text from this image. Please ensure the photo is clear and well-lit."

    try:
        doc_lang = detect(extracted_text[:500])
        print(f"🌐 Document Language Detected: {doc_lang}")
    except:
        doc_lang = 'en'
        
    indic_langs = ['hi', 'mr', 'gu', 'kn', 'ta', 'te', 'bn', 'ml', 'or', 'pa']
    
    if doc_lang in indic_langs:
        print("🔄 Translating document to English via Groq Llama-3...")
        english_contract = translate_text(extracted_text, "English")
    else:
        english_contract = extracted_text

    print("🧠 Groq DeepSeek-70b analyzing contract liabilities...")
    analysis_prompt = f"""You are an expert AI legal paralegal.
    Review the following contract text extracted from an image.
    Identify any major liabilities, unusual clauses, or risks.
    Format your response with clear headings and bullet points.

    CONTRACT TEXT:
    {english_contract}

    RISK ANALYSIS:"""
    
    raw_response = deepseek_llm.invoke(analysis_prompt).content
    clean_english_analysis = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

    if user_lang != 'en':
        target_lang = LANGUAGE_MAP.get(user_lang, 'English')
        print(f"🔄 Translating analysis back to {target_lang}...")
        return translate_text(clean_english_analysis, target_lang)

    return clean_english_analysis

def summarize_conversation(chat_id="default", user_lang="en"):
    """Reads memory buffer, generates a summary via Groq, and translates it."""
    history = chat_histories.get(chat_id, [])
    
    if not history:
        error_msg = "We haven't discussed anything yet! Use /ask to query your contracts first."
        if user_lang != 'en':
            target_lang = LANGUAGE_MAP.get(user_lang, 'English')
            return translate_text(error_msg, target_lang)
        return error_msg
        
    history_string = ""
    for msg in history:
        history_string += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    prompt = (
        "You are an AI assistant. Please provide a brief, professional executive summary of the following conversation "
        "regarding legal contracts. Highlight the main questions asked and the key findings.\n\n"
        f"Conversation:\n{history_string}\n\n"
        "Executive Summary:"
    )
    
    print("📝 Generating Session Summary via Groq...")
    raw_response = deepseek_llm.invoke(prompt).content
    clean_english_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    
    if user_lang != 'en':
        target_lang = LANGUAGE_MAP.get(user_lang, 'English')
        print(f"🔄 Translating summary to {target_lang}...")
        return translate_text(clean_english_response, target_lang)
    
    return clean_english_response