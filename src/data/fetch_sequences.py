import pandas as pd
import requests
import time
import logging
from io import StringIO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_sequences(input_csv, output_csv, batch_size=500):
    """
    Reads a list of UniProt IDs and fetches their sequences via UniProt REST API.
    """
    df = pd.read_csv(input_csv)
    ids = df['uniprot_id'].unique().tolist()
    
    logging.info(f"Found {len(ids):,} unique proteins to fetch.")
    
    # Check if output exists to resume partial downloads
    if Path(output_csv).exists():
        existing_df = pd.read_csv(output_csv)
        existing_ids = set(existing_df['uniprot_id'])
        ids = [i for i in ids if i not in existing_ids]
        logging.info(f"Resuming... {len(ids)} left to fetch.")
    else:
        # Create empty file with header
        pd.DataFrame(columns=['uniprot_id', 'sequence']).to_csv(output_csv, index=False)

    if not ids:
        logging.info("All sequences already fetched!")
        return

    # UniProt API URL
    url = "https://rest.uniprot.org/uniprotkb/accessions"
    
    processed_count = 0
    
    # Process in batches
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        
        try:
            # UniProt accepts comma-separated IDs
            response = requests.get(
                url,
                params={'accessions': ','.join(batch), 'format': 'tsv', 'fields': 'accession,sequence'}
            )
            response.raise_for_status()
            
            # Parse TSV response
            batch_data = pd.read_csv(StringIO(response.text), sep='\t')
            
            # Rename columns to match our schema
            # UniProt returns: "Entry", "Sequence"
            batch_data = batch_data.rename(columns={"Entry": "uniprot_id", "Sequence": "sequence"})
            
            # Append to CSV immediately (safe against crashes)
            batch_data.to_csv(output_csv, mode='a', header=False, index=False)
            
            processed_count += len(batch_data)
            logging.info(f"Fetched {processed_count}/{len(ids) + (len(existing_ids) if 'existing_ids' in locals() else 0)}")
            
            # Be nice to the API
            time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"Error fetching batch {i}: {e}")
            time.sleep(2) # Wait a bit before retry if needed

    logging.info(f"Done! Sequences saved to {output_csv}")

if __name__ == "__main__":
    fetch_sequences("data/processed/proteins.csv", "data/processed/protein_sequences.csv") 