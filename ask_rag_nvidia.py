import os
import sys
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# This script demonstrates a simple "Needle in a Haystack" scenario, 
# where we have a large context of text (the "haystack") 
# and we want to see if the model can find a specific piece of information (the "needle") 
# when asked a question about it.

# In that scenario, we will use the full text of 'Crime and Punishment' by Fyodor Dostoevsky as our haystack, 
# in which we have already injected a specific piece of information about a character named Denis.


# For this to work, you need to set the NVIDIA_API_KEY environment variable 
# with your API key from https://build.nvidia.com/models

# --- SETUP ---
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    # Try loading api_key from .env file if not set in environment variables
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY is not set.")
        print("Set it before running, for example:")
        print("  export NVIDIA_API_KEY='your_api_key_here'")
        sys.exit(1)

#model_name = "qwen/qwen3-coder-480b-a35b-instruct"
#model_name = os.getenv("NVIDIA_MODEL", "qwen/qwen3-coder-480b-a35b-instruct")
model_name = "mistralai/mistral-large-3-675b-instruct-2512" 

stream_mode = True

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key,
    timeout=25, # Set a reasonable timeout for the request
    max_retries=0
)

# Setup ChromaDB with LOCAL Embeddings
# The model used to generate embeddings is "all-MiniLM-L6-v2", which is a small but effective model for many tasks.
# See https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 for more details.
# This model runs on your machine. No API calls, no quotas, and very fast.
local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

USE_PERSISTENT_DB = True

if USE_PERSISTENT_DB:
    print("Using persistent ChromaDB client. Data will be saved to disk and reused across runs.")
    # We'll use a persistent client so you don't have to re-index every time you run the script
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
else:
    print("Using in-memory ChromaDB client. Data will be lost after the script finishes.")
    chroma_client = chromadb.Client()

# Create or get the collection where we will store our document chunks and their embeddings
collection = chroma_client.get_or_create_collection(
    name="literary_analysis_local", 
    embedding_function=local_ef
)

# RAG Logic: Processing the file
# The idea is to index the text of 'Crime and Punishment' 
# into a local ChromaDB collection, and then perform a similarity search 
# to retrieve relevant snippets based on a question, which will then be sent to the model 
# for generating a response.
def index_document(filepath):
    if collection.count() > 0:
        print("Collection already indexed. Skipping...")
        return

    print(f"Indexing {filepath} locally (this will take a moment)...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Simple paragraph chunking
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 100]
    
    # Indexing locally is much faster—no batching for quotas needed!
    collection.add(
        documents=chunks,
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    print(f"Indexed {len(chunks)} chunks locally.")

# get_rag_context(query, n_results=5) performs a similarity search in the ChromaDB collection 
# to retrieve the most relevant snippets based on the query.
def get_rag_context(query, n_results=5):
    # Search the vector DB for the most relevant snippets
    results = collection.query(query_texts=[query], n_results=n_results)
    # Flatten the list of documents retrieved
    return "\n---\n".join(results['documents'][0])

# Generation Logic
# Build a prompt that includes the retrieved context and the question, 
# and send it to the model to generate a response.
def generate_response(question, context):

    full_prompt = f"""
    You are a literary analyst. Using only the following excerpts from the book, 
    answer the question provided. If the information is not in the excerpts, say you don't know.

    EXCERPTS:
    {context}

    QUESTION:
    {question}
    """

    # Print the prompt length in characters and tokens for debugging
    print(f"Prompt length: {len(full_prompt)} characters")
    # A rough estimate of token count (1 token ≈ 4 characters)
    estimated_token_count = len(full_prompt) // 4
    print(f"Estimated token count: {estimated_token_count} tokens")
    # Print full prompt for debugging (optional, can be very long)
    print(f"Full prompt:\n{full_prompt}\n{'-'*50}")

    print("Sending request...", flush=True)
    print(f"Primary model: {model_name} | stream={stream_mode}", flush=True)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=1,
        top_p=0.95,
        max_tokens=512,
#        extra_body={"chat_template_kwargs": {"thinking": False}},
        stream=True
    )

    got_any_token = False
    for chunk in completion:
        if not getattr(chunk, "choices", None):
          continue
        if chunk.choices and chunk.choices[0].delta.content is not None:
           got_any_token = True
        print(chunk.choices[0].delta.content, end="", flush=True)

        if not got_any_token:
            print("No tokens received.", flush=True)

    print("\nDone.", flush=True)

# Execution
def main():
    # Only index once (in-memory for this demo)
    if collection.count() == 0:
        index_document("docs/CrimeAndPunishment.txt")

    question = "What does this say about Denis?"
    
    print(f"\n--- RAG RETRIEVAL START ---")
    context = get_rag_context(question, n_results=2)

    # Debug - print the retrieved context to verify it's working
    print(f"Retrieved context:\n{context}")
    print("\n--- RAG RETRIEVAL END ---\n")

    continue_processing = True
    if continue_processing:
        print(f"\n\n Retrieved relevant snippets. Sending to model...")
        print(f"Question: {question}")
        print(f"Context:\n{context}")

        print(f"--- Waiting for RESPONSE: ---\n")
    
        generate_response(question, context)

# Note: You can run this script multiple times without re-indexing, 
# since the ChromaDB client is persistent and checks if the collection already has data before indexing again.

if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f"An error occurred: {e}")
