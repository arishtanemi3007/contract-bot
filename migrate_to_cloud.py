import os
import sqlite3
import sqlite_vec
import struct
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# Load secrets from .env
load_dotenv()

# --- CONFIGURATION ---
LOCAL_DB_PATH = "contracts.db"
SUPABASE_URI = os.getenv("SUPABASE_URI")

if not SUPABASE_URI:
    raise ValueError("❌ SUPABASE_URI is missing from your .env file!")

def deserialize_f32(blob):
    """Converts local sqlite-vec bytes back into a Python list of floats."""
    return list(struct.unpack(f"{len(blob)//4}f", blob))

def migrate():
    print("🚀 Starting Secure Data Teleportation to Supabase...")

    # 1. Connect to Local SQLite
    print("📦 Reading local database...")
    local_conn = sqlite3.connect(LOCAL_DB_PATH)
    local_conn.enable_load_extension(True)
    sqlite_vec.load(local_conn)
    local_conn.enable_load_extension(False)
    local_cursor = local_conn.cursor()

    # Get local data (Join the chunks with their vectors)
    local_cursor.execute('''
        SELECT c.document_name, c.chunk_text, v.chunk_embedding 
        FROM contract_chunks c
        JOIN vec_chunks v ON c.id = v.rowid
    ''')
    rows = local_cursor.fetchall()
    print(f"✅ Found {len(rows)} chunks to migrate.")

    # 2. Connect to Cloud Supabase
    print("☁️ Connecting to Supabase...")
    try:
        cloud_conn = psycopg2.connect(SUPABASE_URI)
        register_vector(cloud_conn) # Enable pgvector support
        cloud_cursor = cloud_conn.cursor()
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return

    # 3. Teleport Data
    print("⚡ Uploading data to the cloud...")
    insert_query = """
        INSERT INTO contract_chunks (document_name, chunk_text, chunk_embedding)
        VALUES (%s, %s, %s)
    """
    
    success_count = 0
    for row in rows:
        doc_name, chunk_text, vector_blob = row
        vector_list = deserialize_f32(vector_blob)
        
        try:
            cloud_cursor.execute(insert_query, (doc_name, chunk_text, vector_list))
            success_count += 1
        except Exception as e:
            print(f"⚠️ Error uploading a chunk: {e}")
            cloud_conn.rollback() # reset transaction on error
            continue
            
    cloud_conn.commit()
    
    # 4. Cleanup
    cloud_cursor.close()
    cloud_conn.close()
    local_conn.close()
    
    print(f"🎉 Teleportation Complete! {success_count}/{len(rows)} chunks successfully pushed to Supabase.")

if __name__ == "__main__":
    migrate()