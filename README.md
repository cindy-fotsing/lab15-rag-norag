# lab15-rag-norag

This workspace contains 4 Python applications to compare **NoRAG** (full-context prompting) versus **RAG** (retrieval-augmented generation), using both **Google Gemini** and **NVIDIA-hosted models**.

## Applications

1. `ask_norag_google.py`
- NoRAG with Google Gemini (`gemini-2.5-flash-lite`).
- Loads `docs/CrimeAndPunishment.txt`, injects a "needle" passage, and runs two direct large-context questions:
	- a baseline question about Marmeladov (present in the original book)
	- a needle question about Denis (the injected passage)

2. `ask_norag_nvidia.py`
- NoRAG with NVIDIA-hosted model (`mistralai/mistral-large-3-675b-instruct-2512`).
- Same dual-question NoRAG needle-in-a-haystack approach without retrieval.

3. `ask_rag_google.py`
- RAG with Google Gemini for generation.
- Uses local embeddings (`all-MiniLM-L6-v2`) + local ChromaDB (`chroma_db/`) to retrieve relevant chunks before asking the model.
- Default retrieval uses `n_results=5` for the question: `What does this say about Denis?`

4. `ask_rag_nvidia.py`
- RAG with NVIDIA-hosted model for generation.
- Same local retrieval pipeline (Sentence Transformers + ChromaDB), then sends retrieved context to the model.
- Default retrieval uses `n_results=2` for the question: `What does this say about Denis?`

## Project Structure

- `docs/CrimeAndPunishment.txt`: main source text used by the scripts.
- `chroma_db/`: persistent local vector database (used by RAG scripts).
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.10+ (recommended)
- API keys:
- `GOOGLE_API_KEY` for Google scripts
- `NVIDIA_API_KEY` for NVIDIA scripts

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Then edit `.env` and set your real keys.

## Run Each Application

With your virtual environment activated:

### 1) NoRAG + Google

```bash
python ask_norag_google.py
```

### 2) NoRAG + NVIDIA

```bash
python ask_norag_nvidia.py
```

### 3) RAG + Google

```bash
python ask_rag_google.py
```

### 4) RAG + NVIDIA

```bash
python ask_rag_nvidia.py
```

## Notes

- The RAG apps index text chunks into `chroma_db/` and reuse them across runs.
- First RAG run may take longer due to indexing.
- NoRAG scripts have debug prompt printing disabled by default (`_debug = False`).
- RAG scripts print retrieved context; the NVIDIA RAG script also prints extra prompt/debug metadata.
- Streaming output is enabled in all four applications.

# Free books from Project Gutenberg
- https://www.gutenberg.org/
# Crime and Punishment by Fyodor Dostoevsky
- https://www.gutenberg.org/ebooks/2554
# Huckleberry Finn by Mark Twain
- https://www.gutenberg.org/ebooks/76
