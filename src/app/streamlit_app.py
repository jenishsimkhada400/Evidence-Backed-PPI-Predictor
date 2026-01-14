import streamlit as st
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import time

# Import our project modules
from src.models.two_tower import TwoTowerPPI
from src.rag.retrieve import EvidenceRetriever

# --- CONFIG ---
st.set_page_config(page_title="PPI Predictor + Evidence", layout="wide")

# Determine device (MPS for Mac, CUDA, or CPU)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- CACHED LOADERS (Speed up app) ---

@st.cache_resource
def load_esm_model():
    """Loads the protein language model (ESM-2) to embed inputs on the fly."""
    print(f"Loading ESM-2 Model on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(DEVICE)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_ppi_model():
    """Loads your trained Two-Tower PyTorch model."""
    print("Loading Trained PPI Model...")
    model = TwoTowerPPI().to(DEVICE)
    
    # Load weights
    model_path = Path("models/best_model.pt")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(" Model weights loaded.")
    else:
        st.error("Model file (models/best_model.pt) not found! Did you run training?")
    
    model.eval()
    return model

@st.cache_resource
def load_retriever():
    """Loads the FAISS index for searching evidence."""
    print("Loading RAG Retriever...")
    return EvidenceRetriever()

# --- HELPER: Embed Sequence ---
def get_embedding(tokenizer, model, sequence):
    """Embeds a single protein sequence string into a (1, 320) vector."""
    inputs = tokenizer(
        [sequence], 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over the sequence
        last_hidden = outputs.last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        embedding = sum_emb / sum_mask
        
    return embedding 

# --- UI LAYOUT ---

st.title(" Evidence-Backed PPI Predictor")
st.markdown("""
**Bioinformatics Portfolio Project** Predicts if two proteins interact using a **Siamese Neural Network (ESM-2 embeddings)** and retrieves experimental evidence via **RAG (Retrieval Augmented Generation)**.
""")

# Load Global Resources
tokenizer, esm_model = load_esm_model()
ppi_model = load_ppi_model()
retriever = load_retriever()

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Input Proteins")

# Default values: P53 (P04637) and BARD1 (Q99728) - a known interaction
prot_a_id = st.sidebar.text_input("Protein A (UniProt ID)", value="P04637").strip()
prot_b_id = st.sidebar.text_input("Protein B (UniProt ID)", value="Q99728").strip()

# Helper to look up sequences from your local file (so user doesn't have to paste them)
@st.cache_data
def load_sequences_map():
    # Only loads the sequences we already fetched in Phase 4
    df = pd.read_csv("data/processed/protein_sequences.csv")
    return pd.Series(df.sequence.values, index=df.uniprot_id).to_dict()

seq_map = load_sequences_map()

# --- MAIN LOGIC ---

if st.sidebar.button("Analyze Interaction"):
    
    # 1. Lookup Sequences
    seq_a = seq_map.get(prot_a_id)
    seq_b = seq_map.get(prot_b_id)
    
    # Error handling if ID is invalid
    if not seq_a or not seq_b:
        st.error(f" Sequence not found locally for {prot_a_id} or {prot_b_id}. (Try using IDs from your dataset, e.g., P04637)")
    else:
        # 2. Embed & Predict
        with st.spinner(" Embedding proteins & Running Neural Network..."):
            emb_a = get_embedding(tokenizer, esm_model, seq_a)
            emb_b = get_embedding(tokenizer, esm_model, seq_b)
            
            with torch.no_grad():
                logits = ppi_model(emb_a, emb_b)
                prob = torch.sigmoid(logits).item()
        
        # 3. Retrieve Evidence
        with st.spinner(" Searching 1.2M BioGRID records..."):
            # We fetch top 10 to ensure we catch relevant ones
            docs = retriever.retrieve(prot_a_id, prot_b_id)
            report = retriever.generate_evidence_card(prot_a_id, prot_b_id, docs, prob)

        # --- DISPLAY RESULTS ---
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        # LEFT COLUMN: Prediction Score
        with col1:
            st.subheader("Model Prediction")
            st.metric("Interaction Probability", f"{prob:.1%}")
            
            # Interpretation Logic
            if prob > 0.8:
                st.success(" High Confidence Interaction")
            elif prob < 0.2:
                st.error(" Low Probability")
            else:
                st.warning(" Uncertain / Possible")
                
            st.info(f"Verdict: **{report['verdict']}**")
            st.caption(f"Based on {report['summary']}")

        # RIGHT COLUMN: Evidence Card
        with col2:
            st.subheader("Experimental Evidence (RAG)")
            
            if report['citations']:
                st.success(f"Found {len(report['citations'])} confirmed citations in BioGRID.")
                
                # Display Citations nicely
                for pmid in report['citations']:
                    st.markdown(f" **PubMed {pmid}** — [Link](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                
                if report['methods']:
                    st.markdown("**Methods Detected:** " + ", ".join(report['methods']))
            
            else:
                st.warning("🔍 No direct experimental evidence found in the index.")
                st.markdown("This might be a **novel prediction** or a missing record in this specific dataset version.")
                
            # Show Raw Context (Debug View for Recruiters)
            with st.expander("🔎 View Raw Retrieved Context Snippets"):
                for i, txt in enumerate(report['retrieved_snippets']):
                    st.text(f"Doc {i+1}: {txt}")