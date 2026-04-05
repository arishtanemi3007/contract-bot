# ⚖️ Contract Buddy

An asynchronous, multimodal, and fully localized AI legal assistant deployed via Telegram. Designed with privacy and performance in mind, this project demonstrates advanced Agentic AI architecture, combining Retrieval-Augmented Generation (RAG) with local multimodal vision models to analyze, query, and summarize legal contracts. 

## 🧠 Architecture Overview

Built to run entirely locally, ensuring zero data leakage for sensitive legal documents. The system dynamically manages VRAM by swapping between reasoning and translation models on the fly.

### 🚀 Core Features (Updated for Milestone 1)

* **The Universal Translator Pipeline:** Natively processes legal documents in **English** and **6 Indic Languages** (Hindi, Marathi, Gujarati, Kannada, Tamil, Telugu).
* **Smart OCR Extraction:** Bypasses standard vision models in favor of `EasyOCR`, successfully extracting raw Devanagari and Latin scripts from physical document images and handwritten notes.
* **Dynamic AI Routing:** Uses a "Translate-Reason-Translate" workflow:
  * `langdetect` identifies the document language.
  * `Sarvam-1` (an Indic-optimized LLM) translates regional text into English to prevent hallucination.
  * `DeepSeek-R1` acts as the core legal brain, running risk analysis on the normalized English text.
  * `Sarvam-1` translates the final legal analysis back into the user's selected UI language.
* **Local RAG Pipeline:** Utilizes `sqlite-vec` for ultra-fast, lightweight vector storage and `sentence-transformers` for embedding contract chunks.
* **Stateful Memory:** Implements a custom chat_id-keyed conversational memory buffer, enabling contextual follow-up questions and executive session summaries.
* **Asynchronous Routing:** Built with `python-telegram-bot` and `asyncio.to_thread` to prevent API timeouts during heavy local GPU inference.
* **Output Sanitization:** Engineered regex layers to cleanly parse and remove internal `<think>` tags from reasoning models.

## 🛠️ Tech Stack

* **LLM Engine:** Ollama (Local)
* **Models:** `deepseek-r1:8b` (Reasoning/RAG), `mashriram/sarvam-1` (Indic Translation)
* **Vision/Extraction:** `EasyOCR`
* **Language Detection:** `langdetect`
* **Orchestration:** LangChain (Agentic routing and memory management)
* **Vector Database:** SQLite (`sqlite-vec` extension)
* **Embeddings:** `SentenceTransformers` (`all-MiniLM-L6-v2`)
* **API Framework:** `python-telegram-bot`
* **Language:** Python

## 🔬 Critical Evaluation: Pipeline Guardrails

During stress testing, a known failure mode was identified: "Prompt Override / Hallucination." When presented with non-legal text (e.g., a handwritten Hindi school essay), the AI could attempt to fulfill the "flag high-risk clauses" system prompt by hallucinating standard contract terms.

**The Fix:** Engineered a strict instructional prompt alongside the Universal Translator pipeline, forcing the model to evaluate the context before attempting extraction. 

**Result:** The model successfully identifies non-contracts across multiple languages. For example, when fed a handwritten Hindi essay, EasyOCR extracted the text, Sarvam translated it, DeepSeek realized it wasn't a contract and refused analysis, and Sarvam successfully translated that refusal back to the user ("मैं जोखिम विश्लेषण नहीं कर सकता।").

## 🚀 Local Setup & Installation

**1. Clone the repository:**
git clone [https://github.com/arishtanemi3007/contract-bot.git](https://github.com/arishtanemi3007/contract-bot.git)
cd contract-bot

**2. Set up the virtual environment:**
python -m venv venv
.\venv\Scripts\activate

**3. Install dependencies:**
pip install python-telegram-bot langchain langchain-ollama langchain-text-splitters sentence-transformers sqlite-vec easyocr langdetect python-dotenv

**4. Environment Variables:**
Create a .env file in the root directory:
TELEGRAM_BOT_TOKEN=your_token_here
To generate Telegram token, go to https://web.telegram.org/a/ , search BotFather and follow instructions

**5. Pull the local Ollama models:**
ollama pull deepseek-r1:8b
ollama pull mashriram/sarvam-1

**6. Start the Bot:**
python bot.py

(Note: The first run will automatically download the EasyOCR weights and the SentenceTransformer embedding model.)

Author: **Ishan Chourey**

Portfolio: https://ishan-chourey-profile.netlify.app

LinkedIn: https://www.linkedin.com/in/ishanchourey-1m20051993/