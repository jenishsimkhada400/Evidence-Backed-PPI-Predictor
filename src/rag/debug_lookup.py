import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

def debug_lookup(target_a="P04637", target_b="Q99728"):
    print(f"🔍 DEBUGGING PAIR: {target_a} <-> {target_b}")
    
    corpus_path = "data/processed/evidence_corpus.jsonl"
    if not Path(corpus_path).exists():
        print("❌ Error: evidence_corpus.jsonl not found!")
        return

    print(f"📂 Scanning {corpus_path} (this takes a moment)...")
    
    found_count = 0
    
    with open(corpus_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            meta = data['metadata']
            
            # Check for match (ignoring direction)
            uA = meta.get('uA', '')
            uB = meta.get('uB', '')
            
            # Check 1: Exact match
            if (uA == target_a and uB == target_b) or (uA == target_b and uB == target_a):
                print(f"\n✅ FOUND HIT at line {i}:")
                print(f"   Metadata: {meta}")
                print(f"   Content:  {data['content']}")
                found_count += 1
                if found_count >= 3: 
                    print("   (Stopping after 3 hits)")
                    break
            
            # Check 2: Loose Check (Is it there but with weird formatting?)
            if target_a in str(meta) and target_b in str(meta):
                if found_count == 0:
                    print(f"\n⚠️ FOUND LOOSE MATCH (formatting issue?):")
                    print(f"   Stored as: uA='{uA}', uB='{uB}'")

    if found_count == 0:
        print("\n❌ PAIR NOT FOUND in the corpus file.")
        print("   This means the data was filtered out during Phase 7 (Build Corpus).")
        print("   Common reasons: TaxID filter, missing method, or missing IDs.")

if __name__ == "__main__":
    debug_lookup()