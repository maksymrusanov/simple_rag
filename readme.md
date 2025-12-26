
# RAG with LangChain + Ollama (Local LLM)

This project implements a **Retrieval-Augmented Generation (RAG)** system using **LangChain** and **Ollama** to run large language models locally without relying on any cloud API.

---

## 🚀 Overview

RAG improves language model responses by:

1. Retrieving relevant documents from a knowledge base.  
2. Injecting retrieved context into the prompt.  
3. Generating answers grounded in the retrieved information.

This setup uses:

- LangChain for document loading, splitting, embedding, and retrieval.  
- A vector store (Chroma) for fast similarity search.  
- Ollama to run LLMs locally (e.g., 'llama3:8b').

No external API keys or cloud services are required.

---

## 🧰 Requirements

- Python 3.12  
- Ollama installed and running (`ollama serve`)  
- Downloaded Ollama model (e.g., `ollama pull llama3:8b`)

---

## ⚙️ Installation

```bash
# Install correct version of Python

# macOS
brew install python@3.12

# Ubuntu / Debian
sudo apt update
sudo apt install python3.12

# Clone repo
git clone https://github.com/maksymrusanov/simple_rag.git
cd simple_rag

# Activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

1. In the project root, create a folder named `data` and put all your files that will be used as the knowledge base there.

2. Run the main script:

```bash
python main.py
```
