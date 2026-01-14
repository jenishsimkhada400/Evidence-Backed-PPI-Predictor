import logging
import pandas as pd
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class EvidenceRetriever:
    def __init__(self, corpus_path="data/processed/evidence_corpus.jsonl"):
        """
        Loads the evidence corpus into memory for O(1) exact retrieval.
        """
        print(f"Loading Evidence Corpus from {corpus_path}...")
        
        # Load the JSONL data directly into a DataFrame
        data = []
        with open(corpus_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                # Flatten the metadata for easy filtering
                entry = doc['metadata']
                entry['content'] = doc['content']
                data.append(entry)
        
        self.df = pd.DataFrame(data)
        
        # Create a "pair_id" column for fast lookup: "minID_maxID"
        # This handles the A-B vs B-A problem instantly
        self.df['pair_key'] = self.df.apply(
            lambda row: f"{min(row['uA'], row['uB'])}_{max(row['uA'], row['uB'])}", 
            axis=1
        )
        
        print(f"✅ Loaded {len(self.df):,} evidence records into memory.")

    def retrieve(self, protein_a_id, protein_b_id):
        """
        Retrieves evidence using EXACT Dictionary Lookup.
        """
        pA = str(protein_a_id).strip()
        pB = str(protein_b_id).strip()
        
        # Generate the lookup key
        search_key = f"{min(pA, pB)}_{max(pA, pB)}"
        
        # Filter the DataFrame
        matches = self.df[self.df['pair_key'] == search_key]
        
        # Convert back to list of dicts/objects behaving like Documents
        results = []
        for _, row in matches.iterrows():
            # Create a mock object that mimics LangChain Document
            class MockDoc:
                pass
            d = MockDoc()
            d.page_content = row['content']
            d.metadata = row.to_dict()
            results.append(d)
            
        return results

    def generate_evidence_card(self, protein_a, protein_b, docs, pred_score):
        """
        Synthesizes the report.
        """
        found_pmids = set()
        methods = set()
        retrieved_snippets = []

        for d in docs:
            meta = d.metadata
            found_pmids.add(meta.get('pmid', 'Unknown'))
            m = meta.get('method', 'Unknown')
            if m: methods.add(m)
            retrieved_snippets.append(d.page_content)

        # Verdict Logic
        if len(found_pmids) > 0:
            verdict = "Strong Support" if len(found_pmids) > 1 else "Supportive Evidence"
            summary = f"Found {len(found_pmids)} direct experimental citations."
        elif pred_score > 0.8:
            verdict = "Predicted (Novel)"
            summary = "High model confidence, but no direct experimental records in this dataset."
        elif pred_score < 0.2:
            verdict = "Unlikely Interaction"
            summary = "Low model confidence and no evidence."
        else:
            verdict = "Uncertain"
            summary = "Model is unsure and no evidence found."

        return {
            "protein_a": protein_a,
            "protein_b": protein_b,
            "model_score": f"{pred_score:.4f}",
            "verdict": verdict,
            "summary": summary,
            "methods": list(methods),
            "citations": list(found_pmids)[:5], # Top 5
            "retrieved_snippets": retrieved_snippets[:5]
        }