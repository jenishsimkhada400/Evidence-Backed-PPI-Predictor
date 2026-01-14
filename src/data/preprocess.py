import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure clean logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_biogrid(
    input_path: str, 
    output_path: str, 
    tax_id: int = 9606  # Default: Human
):
    """
    Parses BioGRID TAB3 format.
    Filters for:
    1. Specific Taxon ID (e.g., 9606 for Human)
    2. Physical interactions only
    3. Valid UniProt (SWISS-PROT) IDs
    """
    input_file = Path(input_path)
    if not input_file.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    logging.info(f"Loading raw data from {input_path}...")
    
    # EXACT Column names from your file header check
    col_system_type = "Experimental System Type"
    col_org_A = "Organism ID Interactor A"
    col_org_B = "Organism ID Interactor B"
    col_prot_A = "SWISS-PROT Accessions Interactor A"  # Fixed capitalization
    col_prot_B = "SWISS-PROT Accessions Interactor B"  # Fixed capitalization
    
    use_cols = [col_system_type, col_org_A, col_org_B, col_prot_A, col_prot_B]
    
    # Load data
    df = pd.read_csv(input_path, sep="\t", usecols=use_cols, low_memory=False)
    
    original_len = len(df)
    logging.info(f"Raw rows loaded: {original_len:,}")

    # 1. Filter by Organism (TaxID)
    df = df[
        (df[col_org_A] == tax_id) & 
        (df[col_org_B] == tax_id)
    ]
    logging.info(f"Rows after TaxID {tax_id} filter: {len(df):,}")

    # 2. Filter for Physical interactions
    df = df[df[col_system_type] == "physical"]
    logging.info(f"Rows after Physical filter: {len(df):,}")

    # 3. Handle UniProt IDs
    # Drop rows where IDs are missing or "-"
    df = df[
        (df[col_prot_A] != "-") & 
        (df[col_prot_B] != "-") &
        (df[col_prot_A].notna()) & 
        (df[col_prot_B].notna())
    ]
    
    # BioGRID sometimes lists multiple IDs like "P12345|Q99999". Take the first one.
    df["protA"] = df[col_prot_A].apply(lambda x: str(x).split("|")[0])
    df["protB"] = df[col_prot_B].apply(lambda x: str(x).split("|")[0])
    
    # 4. Canonicalize Pairs (Sort A/B so order doesn't matter)
    df["uA"] = df.apply(lambda row: min(row["protA"], row["protB"]), axis=1)
    df["uB"] = df.apply(lambda row: max(row["protA"], row["protB"]), axis=1)
    
    # Remove self-loops
    df = df[df["uA"] != df["uB"]]

    # Deduplicate
    final_df = df[["uA", "uB"]].drop_duplicates()
    final_df["label"] = 1
    
    logging.info(f"Final clean positive pairs: {len(final_df):,}")
    
    # Save
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logging.info(f"Saved to {output_path}")
    
    # Save unique proteins
    unique_prots = pd.unique(final_df[['uA', 'uB']].values.ravel('K'))
    pd.DataFrame(unique_prots, columns=["uniprot_id"]).to_csv(out_dir / "proteins.csv", index=False)
    logging.info(f"Saved unique protein list ({len(unique_prots):,} proteins) to proteins.csv")

if __name__ == "__main__":
    RAW_FILE = "data/raw/BIOGRID-ALL-LATEST.tab3.txt" 
    OUTPUT_FILE = "data/processed/ppis_positive.csv"
    process_biogrid(RAW_FILE, OUTPUT_FILE)