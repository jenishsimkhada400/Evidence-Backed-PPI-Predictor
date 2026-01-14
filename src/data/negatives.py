import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def generate_negatives(positive_path, output_path, ratio=1.0, seed=42):
    """
    Generates random negative pairs (A, B) such that (A, B) is NOT in the positive set.
    """
    np.random.seed(seed)
    
    logging.info(f"Loading positives from {positive_path}...")
    pos_df = pd.read_csv(positive_path)
    
    # 1. Build a set of existing positives for O(1) lookup
    # Store as tuples (min, max) to handle undirected nature
    pos_set = set(zip(pos_df['uA'], pos_df['uB']))
    
    # 2. Get list of all proteins
    unique_proteins = pd.unique(pos_df[['uA', 'uB']].values.ravel('K'))
    logging.info(f"Pool of {len(unique_proteins):,} proteins to sample from.")
    
    # 3. Calculate how many negatives we need
    num_negatives = int(len(pos_df) * ratio)
    logging.info(f"Generating {num_negatives:,} negative pairs (Ratio {ratio}:1)...")
    
    negatives = []
    
    # 4. Sampling Loop
    while len(negatives) < num_negatives:
        # Sample in batches for speed
        needed = num_negatives - len(negatives)
        batch_size = int(needed * 1.2) + 100  # Over-sample slightly to account for collisions
        
        # Pick random A and random B
        prots_A = np.random.choice(unique_proteins, batch_size)
        prots_B = np.random.choice(unique_proteins, batch_size)
        
        for a, b in zip(prots_A, prots_B):
            if a == b: continue # No self-loops
            
            p1, p2 = min(a, b), max(a, b)
            
            if (p1, p2) not in pos_set:
                negatives.append({'uA': p1, 'uB': p2, 'label': 0})
                pos_set.add((p1, p2)) # Prevent duplicates in negatives
                
            if len(negatives) >= num_negatives:
                break
    
    # 5. Save Combined Dataset
    neg_df = pd.DataFrame(negatives)
    logging.info(f"Created {len(neg_df):,} negatives.")
    
    all_df = pd.concat([pos_df, neg_df], ignore_index=True)
    
    # Shuffle the dataset
    all_df = all_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_path, index=False)
    logging.info(f"Saved merged dataset ({len(all_df):,} rows) to {output_path}")

if __name__ == "__main__":
    generate_negatives(
        "data/processed/ppis_positive.csv", 
        "data/processed/ppis_all.csv", 
        ratio=5.0  # Balanced dataset
    )