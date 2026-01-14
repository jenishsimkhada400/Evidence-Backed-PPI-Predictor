import pandas as pd
import torch
import logging
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def embed_sequences(
    input_csv="data/processed/protein_sequences.csv",
    output_path="data/embeddings/protein_embeddings.pt",
    model_name="facebook/esm2_t6_8M_UR50D",
    batch_size=16
):
    """
    Converts amino acid sequences to fixed-size vectors using ESM-2.
    Saves as a PyTorch dictionary: {uniprot_id: tensor_vector}
    """
    # 1. Setup Device (MPS for Mac, CUDA for Nvidia, else CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon) acceleration! 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA acceleration!")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU (might be slow)...")

    # 2. Load Model & Tokenizer
    logging.info(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval() # Set to inference mode

    # 3. Load Data
    df = pd.read_csv(input_csv)
    # Filter out extremely long sequences to avoid OOM (ESM max is usually ~1024 context)
    # We truncate to 1022 (+2 special tokens)
    ids = df['uniprot_id'].tolist()
    seqs = df['sequence'].tolist()
    
    logging.info(f"Embedding {len(seqs)} sequences...")
    
    embeddings = {}
    
    # 4. Processing Loop
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size)):
            batch_ids = ids[i : i+batch_size]
            batch_seqs = seqs[i : i+batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_seqs, 
                padding=True, 
                truncation=True, 
                max_length=1024, 
                return_tensors="pt"
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Mean Pooling: Average all amino acid vectors in the sequence
            # outputs.last_hidden_state shape: (batch, seq_len, 320)
            # We use the attention mask to ignore padding tokens in the average
            
            last_hidden_state = outputs.last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            
            batch_embeddings = sum_embeddings / sum_mask # (batch, 320)
            
            # Move to CPU and store
            batch_embeddings = batch_embeddings.cpu()
            
            for pid, emb in zip(batch_ids, batch_embeddings):
                embeddings[pid] = emb

    # 5. Save
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Saving dictionary with {len(embeddings)} entries to {output_path}...")
    torch.save(embeddings, output_path)
    logging.info("Done!")

if __name__ == "__main__":
    embed_sequences()