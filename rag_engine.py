import os
import re
import time
import base64
import sqlite3
import sqlite_vec
import struct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownTextSplitter
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# 1. Initialize the embedding model
print("Loading embedding model (this may take a moment the first time)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

DB_PATH = "contracts.db"
DATA_DIR = "data"

# --- MEMORY BUFFER ---
# Stores conversation history keyed by Telegram chat_id
chat_histories = {}
MAX_HISTORY = 4 # Keep the last 4 interactions to save VRAM

def init_db():
    """Initializes the SQLite database with the sqlite-vec extension."""
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS contract_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT,
            chunk_text TEXT
        )
    ''')
    
    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_embedding float[384]
        )
    ''')
    conn.commit()
    return conn

def serialize_f32(vector):
    """Converts a list of floats into the raw bytes required by sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)

def ingest_documents():
    """Reads Markdown files, chunks them, and stores embeddings in the DB."""
    conn = init_db()
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Could not find the '{DATA_DIR}' folder.")
        return

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".md"):
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            print(f"Processing {filename}...")
            chunks = splitter.create_documents([content])
            
            for chunk in chunks:
                text = chunk.page_content
                embedding = embedding_model.encode(text).tolist()
                
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO contract_chunks (document_name, chunk_text) VALUES (?, ?)",
                    (filename, text)
                )
                row_id = cursor.lastrowid
                
                cursor.execute(
                    "INSERT INTO vec_chunks (rowid, chunk_embedding) VALUES (?, ?)",
                    (row_id, serialize_f32(embedding))
                )
            conn.commit()
    
    print("✅ All documents successfully embedded and stored in contracts.db!")
    conn.close()

def retrieve_chunks(query, top_k=3):
    """Embeds the query and fetches the top-k most relevant chunks from the database."""
    conn = init_db()
    query_embedding = embedding_model.encode(query).tolist()
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM contract_chunks")
    
    cursor.execute('''
        SELECT 
            contract_chunks.document_name, 
            contract_chunks.chunk_text
        FROM vec_chunks
        JOIN contract_chunks ON contract_chunks.id = vec_chunks.rowid
        WHERE vec_chunks.chunk_embedding MATCH ?
        AND k = ?
    ''', (serialize_f32(query_embedding), top_k))
    
    results = cursor.fetchall()
    conn.close()
    return results

def answer_query(query, chat_id="default"):
    """Retrieves context, injects memory, and generates an answer."""
    retrieved_docs = retrieve_chunks(query)
    print(f"[DEBUG] Successfully retrieved {len(retrieved_docs)} relevant chunks for the prompt.")
    
    if not retrieved_docs:
        return "I couldn't find any relevant information in the uploaded contracts."
        
    context_parts = []
    for row in retrieved_docs:
        doc_name = row[0]
        text = row[1]
        context_parts.append(f"[Source: {doc_name}]\n{text}\n")
        
    context_string = "\n".join(context_parts)
    
    # 1. Format the Chat History
    history = chat_histories.get(chat_id, [])
    history_string = ""
    if history:
        history_string = "Previous Conversation Context:\n"
        for msg in history:
            history_string += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    llm = OllamaLLM(model="deepseek-r1:8b")
    
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
    
    prompt = prompt_template.format(chat_history=history_string, context=context_string, query=query)
    print(f"🧠 Thinking... (Querying deepseek-r1:8b with memory)")
    
    raw_response = llm.invoke(prompt)
    clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    
    # 2. Save the new interaction to Memory
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    chat_histories[chat_id].append({"role": "user", "content": query})
    chat_histories[chat_id].append({"role": "assistant", "content": clean_response})
    
    # 3. Prune old memory so we don't crash the GPU
    if len(chat_histories[chat_id]) > MAX_HISTORY * 2:
        chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY * 2:]
        
    return clean_response

def summarize_conversation(chat_id="default"):
    """Reads the user's memory buffer and generates an executive summary."""
    history = chat_histories.get(chat_id, [])
    
    if not history:
        return "We haven't discussed anything yet! Use /ask to query your contracts first."
        
    history_string = ""
    for msg in history:
        history_string += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    prompt = (
        "You are an AI assistant. Please provide a brief, professional executive summary of the following conversation "
        "regarding legal contracts. Highlight the main questions asked and the key findings.\n\n"
        f"Conversation:\n{history_string}\n\n"
        "Executive Summary:"
    )
    
    llm = OllamaLLM(model="deepseek-r1:8b")
    print("📝 Generating Session Summary...")
    raw_response = llm.invoke(prompt)
    clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    
    return clean_response

def analyze_contract_image(image_path):
    {
                "type": "text", 
                "text": (
                    "You are an expert legal assistant. Read the text in this image. "
                    "FIRST, determine if this is a legal contract or formal agreement. "
                    "If it is NOT a legal document (e.g., it is a social media post, casual text, or unrelated photo), "
                    "reply with: '⚠️ This does not appear to be a legal document.' and provide a brief 1-sentence summary of what it actually is. "
                    "IF it IS a legal document, extract the key terms and clearly flag any high-risk clauses. Keep it concise."
                )
            },
    
    # 1. Convert the physical image into AI-readable code (Base64)
    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        
    # 2. Call the Vision model (ChatOllama handles multimodal inputs)
    chat_model = ChatOllama(model="llama3.2-vision")
    
    # 3. LangChain message format for multimodal inputs
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": "You are an expert legal assistant. Read this image of a contract or document. Extract the key terms and clearly flag any high-risk clauses (like indemnification, extreme liabilities, or strict deadlines). Keep it professional and concise."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
    )
    
    print("👁️ Thinking... (Analyzing image with llama3.2-vision)")
    response = chat_model.invoke([message])
    return response.content

if __name__ == "__main__":
    print("\n--- Testing RAG Pipeline ---")
    test_question = "What is the liability limit?"
    print(f"Test Question: {test_question}")
    
    start_time = time.time()
    answer = answer_query(test_question)
    end_time = time.time()
    
    print(f"\n🤖 Final Output:\n{answer}")
    print(f"\n⏱️ Response Time: {end_time - start_time:.2f} seconds")