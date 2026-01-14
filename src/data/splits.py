import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def make_protein_disjoint_split(input_path, output_dir, test_size=0.15, val_size=0.10, seed=42):
    """
    Splits data such that proteins in Test set are completely unseen in Train.
    This is a 'Strict Cold Start' evaluation.
    """
    logging.info(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Identify all unique proteins
    unique_proteins = pd.unique(df[['uA', 'uB']].values.ravel('K'))
    logging.info(f"Total unique proteins: {len(unique_proteins):,}")
    
    # 2. Split PROTEINS, not edges
    train_prots, test_prots = train_test_split(unique_proteins, test_size=test_size, random_state=seed)
    
    # Further split train proteins to get validation proteins
    train_prots, val_prots = train_test_split(train_prots, test_size=val_size, random_state=seed)
    
    train_set = set(train_prots)
    val_set = set(val_prots)
    test_set = set(test_prots)
    
    logging.info(f"Split Proteins -> Train: {len(train_set):,}, Val: {len(val_set):,}, Test: {len(test_set):,}")
    
    # 3. Assign Edges based on protein membership
    # Rule:
    # If BOTH proteins in Train Set -> Train Split
    # If BOTH proteins in Val Set   -> Val Split
    # If BOTH proteins in Test Set  -> Test Split
    # Mixed edges (e.g. Train-Test) -> Drop (to prevent leakage)
    
    def assign_split(row):
        uA, uB = row['uA'], row['uB']
        if uA in train_set and uB in train_set:
            return 'train'
        elif uA in val_set and uB in val_set:
            return 'val'
        elif uA in test_set and uB in test_set:
            return 'test'
        else:
            return 'drop'
            
    logging.info("Assigning edges to splits (this may take a moment)...")
    df['split'] = df.apply(assign_split, axis=1)
    
    # 4. Filter and Save
    train_df = df[df['split'] == 'train'].drop(columns=['split'])
    val_df   = df[df['split'] == 'val'].drop(columns=['split'])
    test_df  = df[df['split'] == 'test'].drop(columns=['split'])
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    
    logging.info(f"Saved Splits to {output_dir}")
    logging.info(f"  Train edges: {len(train_df):,}")
    logging.info(f"  Val edges:   {len(val_df):,}")
    logging.info(f"  Test edges:  {len(test_df):,}")
    logging.info(f"  Dropped (mixed) edges: {len(df) - len(train_df) - len(val_df) - len(test_df):,}")

if __name__ == "__main__":
    make_protein_disjoint_split(
        "data/processed/ppis_all.csv",
        "data/processed/splits/protein_disjoint"
    )