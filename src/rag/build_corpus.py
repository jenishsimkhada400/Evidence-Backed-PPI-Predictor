import pandas as pd
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def build_corpus(
    raw_path="data/raw/BIOGRID-ALL-LATEST.tab3.txt",
    output_path="data/processed/evidence_corpus.jsonl",
    tax_id=9606
):
    """
    Reads raw BioGRID data and creates a text corpus for RAG.
    Each row becomes a JSON 'document' with search metadata.
    """
    logging.info(f"Reading raw data from {raw_path}...")
    
    # CORRECTED Column Names based on your header check
    col_pub_source = "Publication Source" # Was "Pubmed ID"
    col_prot_A = "SWISS-PROT Accessions Interactor A"
    col_prot_B = "SWISS-PROT Accessions Interactor B"
    
    use_cols = [
        "Official Symbol Interactor A",
        "Official Symbol Interactor B",
        "Experimental System Type",
        "Experimental System", 
        col_pub_source,
        "Organism ID Interactor A",
        "Organism ID Interactor B",
        col_prot_A, 
        col_prot_B
    ]
    
    # Load Data
    df = pd.read_csv(raw_path, sep="\t", usecols=use_cols, low_memory=False)
    
    # Filter (Same rules as Phase 1)
    df = df[
        (df["Organism ID Interactor A"] == tax_id) & 
        (df["Organism ID Interactor B"] == tax_id) &
        (df["Experimental System Type"] == "physical")
    ]
    
    # Drop rows without IDs
    df = df.dropna(subset=[col_prot_A, col_prot_B])
    
    logging.info(f"Found {len(df):,} evidence records.")
    
    documents = []
    
    logging.info("Formatting documents...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # clean IDs
        ua = str(row[col_prot_A]).split('|')[0]
        ub = str(row[col_prot_B]).split('|')[0]
        sym_a = str(row["Official Symbol Interactor A"])
        sym_b = str(row["Official Symbol Interactor B"])
        method = str(row["Experimental System"])
        
        # Clean PubMed ID (remove "PUBMED:" prefix if present)
        raw_pmid = str(row[col_pub_source])
        pmid = raw_pmid.replace("PUBMED:", "") if "PUBMED:" in raw_pmid else raw_pmid
        
        # Canonicalize pair for searching (sort A/B)
        if ua > ub:
            ua, ub = ub, ua
            sym_a, sym_b = sym_b, sym_a
            
        # The text content the LLM will see
        text_content = (
            f"INTERACTION: {sym_a} ({ua}) interacts with {sym_b} ({ub}). "
            f"DETECTED_BY: {method}. "
            f"SOURCE: PubMed {pmid}."
        )
        
        doc = {
            "content": text_content,
            "metadata": {
                "uA": ua,
                "uB": ub,
                "symA": sym_a,
                "symB": sym_b,
                "pmid": pmid,
                "method": method
            }
        }
        documents.append(doc)
        
    # Save as JSONL
    logging.info(f"Saving {len(documents):,} documents to {output_path}...")
    with open(output_path, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
            
    logging.info("Done! Corpus ready.")

if __name__ == "__main__":
    build_corpus()