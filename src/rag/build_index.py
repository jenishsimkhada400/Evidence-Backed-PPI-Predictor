import json
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def build_index(
    corpus_path="data/processed/evidence_corpus.jsonl",
    index_path="data/rag_index"
):
    logging.info("Loading corpus...")
    docs = []
    with open(corpus_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Convert to LangChain Document format
            # We add specific keywords to 'page_content' to ensure retrieval hits on IDs
            searchable_text = f"{item['content']} {item['metadata']['uA']} {item['metadata']['uB']}"
            
            docs.append(Document(
                page_content=searchable_text,
                metadata=item['metadata']
            ))
            
    logging.info(f"Loaded {len(docs)} documents. Initializing embedding model...")
    
    # Use a small, fast local model (no OpenAI API needed yet)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    logging.info("Building FAISS index (this may take a few minutes)...")
    # In a real app, we'd batch this. For <1M docs, it fits in RAM usually.
    # If it crashes, slice docs[:50000] to test.
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    logging.info(f"Saving index to {index_path}...")
    vectorstore.save_local(index_path)
    logging.info("Done! Index saved.")

if __name__ == "__main__":
    build_index()