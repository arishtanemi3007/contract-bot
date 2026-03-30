⚖️ Avivo Contract Intelligence Bot
An asynchronous, multimodal, and fully localized AI legal assistant deployed via Telegram. Designed with privacy and performance in mind, this project demonstrates advanced Agentic AI architecture, combining Retrieval-Augmented Generation (RAG) with local multimodal vision models to analyze, query, and summarize legal contracts.

🧠 Architecture Overview
Built to run entirely locally, ensuring zero data leakage for sensitive legal documents. The system dynamically manages VRAM by swapping between reasoning and vision models on the fly.

Core Features
Local RAG Pipeline: Utilizes sqlite-vec for ultra-fast, lightweight vector storage and sentence-transformers for embedding contract chunks.

Stateful Memory: Implements a custom chat_id-keyed conversational memory buffer, enabling contextual follow-up questions and executive session summaries.

Asynchronous Routing: Built with python-telegram-bot and asyncio.to_thread to prevent API timeouts during heavy local GPU inference.

Multimodal Vision: Integrates llama3.2-vision to OCR and analyze physical contract photos directly from the chat.

Output Sanitization: Engineered regex layers to cleanly parse and remove internal <think> tags from reasoning models.

🛠️ Tech Stack
LLM Engine: Ollama (Local)

Models: deepseek-r1:8b (Reasoning/RAG), llama3.2-vision (Multimodal)

Orchestration: LangChain (Agentic routing and memory management)

Vector Database: SQLite (sqlite-vec extension)

API Framework: python-telegram-bot

Language: Python

🔬 Critical Evaluation: Vision Model Guardrails
During stress testing, a known failure mode was identified: "Prompt Override / Hallucination." When presented with non-legal text (e.g., a social media post), the vision model attempted to fulfill the "flag high-risk clauses" system prompt by hallucinating standard contract terms.

The Fix: Engineered an "Escape Hatch" prompt, forcing the model to classify the document type before attempting extraction.

Result: The model successfully identifies non-contracts and refuses to generate legal advice.

Test cases demonstrating this multimodal routing behavior are available in the test_suite/ directory.

🚀 Local Setup & Installation
1. Clone the repository:

git clone https://github.com/YOUR_GITHUB_USERNAME/avivo-contract-bot.git
cd avivo-contract-bot

2. Set up the virtual environment:

python -m venv venv
.\venv\Scripts\activate

3. Install dependencies:

pip install python-telegram-bot langchain langchain-ollama sentence-transformers sqlite-vec python-dotenv

4. Environment Variables:
Create a .env file in the root directory:

TELEGRAM_BOT_TOKEN=your_token_here

5. Run the local AI Engine:

ollama pull deepseek-r1:8b
ollama pull llama3.2-vision

6. Start the Bot:

python bot.py

Author: Ishan Chourey
https://ishan-chourey-profile.netlify.app
https://www.linkedin.com/in/ishanchourey-1m20051993/