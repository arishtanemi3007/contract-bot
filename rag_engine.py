import os
import re
import time
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
import easyocr
from langdetect import detect
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Load Cloud Secrets
load_dotenv()
SUPABASE_URI = os.getenv("SUPABASE_URI")

if not SUPABASE_URI:
    print("⚠️ WARNING: SUPABASE_URI is missing from your .env file!")

# ---------------------------------------------------------
# 1. INITIALIZATION & GLOBALS
# ---------------------------------------------------------
print("Booting local embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Booting local LLMs...")
deepseek_llm = OllamaLLM(model="deepseek-r1:8b")
# Initialize Sarvam for Indic translations
sarvam_llm = OllamaLLM(model="mashriram/sarvam-1", temperature=0.1)

print("Loading EasyOCR Vision Models (English, Hindi, Marathi)...")
reader = easyocr.Reader(['en', 'hi', 'mr'])

DATA_DIR = "data"

# --- MEMORY BUFFER ---
chat_histories = {}
MAX_HISTORY = 4

LANGUAGE_MAP = {
    'hi': 'Hindi', 'mr': 'Marathi', 'gu': 'Gujarati',
    'kn': 'Kannada', 'ta': 'Tamil', 'te': 'Telugu'
}

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def translate_with_sarvam(text, target_language="English"):
    """Uses Sarvam-1 for strict, zero-hallucination legal translation."""
    prompt = f"""You are a highly accurate legal translator.
    Translate the following text into {target_language}.
    Do not summarize, do not add commentary. Just provide the exact translation.

    TEXT:
    {text}

    TRANSLATION:"""
    raw_response = sarvam_llm.invoke(prompt)
    return raw_response.strip()

# ---------------------------------------------------------
# 3. CORE CLOUD DB OPERATIONS (SUPABASE)
# ---------------------------------------------------------
def ingest_documents():
    """Reads Markdown files, chunks them, and stores embeddings DIRECTLY IN CLOUD."""
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Could not find the '{DATA_DIR}' folder.")
        return

    try:
        conn = psycopg2.connect(SUPABASE_URI)
        register_vector(conn)
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ Could not connect to Cloud DB for ingestion: {e}")
        return

    insert_query = """
        INSERT INTO contract_chunks (document_name, chunk_text, chunk_embedding)
        VALUES (%s, %s, %s)
    """

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".md"):
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            print(f"Processing {filename} to Cloud...")
            chunks = splitter.create_documents([content])
            
            for chunk in chunks:
                text = chunk.page_content
                embedding = embedding_model.encode(text).tolist()
                
                try:
                    cursor.execute(insert_query, (filename, text, embedding))
                except Exception as e:
                    print(f"⚠️ Error uploading chunk: {e}")
                    conn.rollback()
                    continue
            conn.commit()
    
    print("✅ All new documents successfully pushed to Supabase!")
    cursor.close()
    conn.close()

def retrieve_chunks(query, top_k=3):
    """Embeds the query and fetches the top-k most relevant chunks from the CLOUD."""
    query_embedding = embedding_model.encode(query).tolist()
    
    # 🚨 FIX: Format the list exactly how Postgres expects a vector string
    vector_string = f"[{','.join(map(str, query_embedding))}]"
    
    try:
        # Connect to Cloud
        conn = psycopg2.connect(SUPABASE_URI)
        register_vector(conn)
        cursor = conn.cursor()
        
        # 🚨 FIX: Added ::vector to explicitly cast the string
        cursor.execute("""
            SELECT document_name, chunk_text 
            FROM contract_chunks 
            ORDER BY chunk_embedding <=> %s::vector 
            LIMIT %s
        """, (vector_string, top_k))
        
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
    """Retrieves context from Cloud, injects memory, and translates if necessary."""
    
    # 1. Translate incoming Indic query to English for the DeepSeek brain
    if user_lang != 'en':
        print(f"🔄 Translating query from {user_lang} to English...")
        english_query = translate_with_sarvam(query, "English")
    else:
        english_query = query

    retrieved_docs = retrieve_chunks(english_query)
    print(f"[DEBUG] Successfully retrieved {len(retrieved_docs)} chunks from CLOUD.")
    
    if not retrieved_docs:
        return "I couldn't find any relevant information in the uploaded contracts."
        
    context_parts = []
    for row in retrieved_docs:
        doc_name = row[0]
        text = row[1]
        context_parts.append(f"[Source: {doc_name}]\n{text}\n")
        
    context_string = "\n".join(context_parts)
    
    # Format Chat History
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
    print("🧠 Thinking... (Querying deepseek-r1:8b with memory)")
    
    raw_response = deepseek_llm.invoke(prompt)
    clean_english_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    
    # Save English logic to memory to prevent cross-language confusion in future turns
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    chat_histories[chat_id].append({"role": "user", "content": english_query})
    chat_histories[chat_id].append({"role": "assistant", "content": clean_english_response})
    
    if len(chat_histories[chat_id]) > MAX_HISTORY * 2:
        chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY * 2:]
        
    # 2. Translate English answer back to the user's selected language
    if user_lang != 'en':
        target_lang = LANGUAGE_MAP.get(user_lang, 'English')
        print(f"🔄 Translating final answer back to {target_lang}...")
        return translate_with_sarvam(clean_english_response, target_lang)

    return clean_english_response

def analyze_contract_image(image_path, user_lang="en"):
    """Extracts text via OCR, normalizes to English, analyzes, and translates back."""
    print("🖼️ Running EasyOCR extraction...")
    
    results = reader.readtext(image_path, detail=0)
    extracted_text = " ".join(results)
    
    if len(extracted_text.strip()) < 10:
        return "❌ Could not extract enough text from this image. Please ensure the photo is clear and well-lit."

    try:
        doc_lang = detect(extracted_text[:500])
        print(f"🌐 Document Language Detected: {doc_lang}")
    except:
        doc_lang = 'en'
        
    indic_langs = ['hi', 'mr', 'gu', 'kn', 'ta', 'te', 'bn', 'ml', 'or', 'pa']
    
    # Normalize to English for DeepSeek
    if doc_lang in indic_langs:
        print("🔄 Translating document to English via Sarvam-1...")
        english_contract = translate_with_sarvam(extracted_text, "English")
    else:
        english_contract = extracted_text

    # DeepSeek Risk Analysis
    print("🧠 DeepSeek-R1 analyzing contract liabilities...")
    analysis_prompt = f"""You are an expert AI legal paralegal.
    Review the following contract text extracted from an image.
    Identify any major liabilities, unusual clauses, or risks.
    Format your response with clear headings and bullet points.

    CONTRACT TEXT:
    {english_contract}

    RISK ANALYSIS:"""
    
    raw_response = deepseek_llm.invoke(analysis_prompt)
    clean_english_analysis = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

    # Translate back to the user's preferred language if necessary
    if user_lang != 'en':
        target_lang = LANGUAGE_MAP.get(user_lang, 'English')
        print(f"🔄 Translating analysis back to {target_lang}...")
        return translate_with_sarvam(clean_english_analysis, target_lang)

    return clean_english_analysis

def summarize_conversation(chat_id="default", user_lang="en"):
    """Reads the user's memory buffer, generates a summary, and translates it."""
    history = chat_histories.get(chat_id, [])
    
    if not history:
        error_msg = "We haven't discussed anything yet! Use /ask to query your contracts first."
        if user_lang != 'en':
            target_lang = LANGUAGE_MAP.get(user_lang, 'English')
            return translate_with_sarvam(error_msg, target_lang)
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
    
    print("📝 Generating Session Summary...")
    raw_response = deepseek_llm.invoke(prompt)
    clean_english_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    
    # Translate back to the user's preferred language if necessary
    if user_lang != 'en':
        target_lang = LANGUAGE_MAP.get(user_lang, 'English')
        print(f"🔄 Translating summary to {target_lang}...")
        return translate_with_sarvam(clean_english_response, target_lang)
    
    return clean_english_response

if __name__ == "__main__":
    print("\n--- Testing RAG Pipeline (CLOUD) ---")
    test_question = "What is the liability limit?"
    print(f"Test Question: {test_question}")
    
    start_time = time.time()
    answer = answer_query(test_question)
    end_time = time.time()
    
    print(f"\n🤖 Final Output:\n{answer}")
    print(f"\n⏱️ Response Time: {end_time - start_time:.2f} seconds")