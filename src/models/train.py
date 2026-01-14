import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from src.models.two_tower import TwoTowerPPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- 1. Custom Dataset ---
class PPIDataset(Dataset):
    def __init__(self, csv_path, embeddings_path):
        self.data = pd.read_csv(csv_path)
        self.embeddings = torch.load(embeddings_path, map_location="cpu") # Load dict once
        
        # Filter out rows where we don't have embeddings (rare edge case)
        initial_len = len(self.data)
        self.data = self.data[
            self.data['uA'].isin(self.embeddings.keys()) & 
            self.data['uB'].isin(self.embeddings.keys())
        ]
        if len(self.data) < initial_len:
            logging.warning(f"Dropped {initial_len - len(self.data)} rows due to missing embeddings.")
            
        self.pairs = self.data[['uA', 'uB']].values
        self.labels = self.data['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prot_a, prot_b = self.pairs[idx]
        label = self.labels[idx]
        
        # Fetch embeddings (fast dictionary lookup)
        emb_a = self.embeddings[prot_a]
        emb_b = self.embeddings[prot_b]
        
        return emb_a, emb_b, torch.tensor(label)

# --- 2. Training Function ---
def train_model():
    # Config
    BATCH_SIZE = 256
    LR = 0.001
    EPOCHS = 10
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Training on {DEVICE}")
    
    # Paths
    TRAIN_PATH = "data/processed/splits/protein_disjoint/train.csv"
    VAL_PATH = "data/processed/splits/protein_disjoint/val.csv"
    EMBED_PATH = "data/embeddings/protein_embeddings.pt"
    SAVE_DIR = Path("models/")
    SAVE_DIR.mkdir(exist_ok=True)
    
    # Load Data
    logging.info("Loading Datasets...")
    train_ds = PPIDataset(TRAIN_PATH, EMBED_PATH)
    val_ds = PPIDataset(VAL_PATH, EMBED_PATH)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Init Model
    model = TwoTowerPPI().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_prauc = 0.0
    
    # Loop
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for emb_a, emb_b, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            emb_a, emb_b, labels = emb_a.to(DEVICE), emb_b.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(emb_a, emb_b)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATE ---
        model.eval()
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for emb_a, emb_b, labels in val_loader:
                emb_a, emb_b = emb_a.to(DEVICE), emb_b.to(DEVICE)
                logits = model(emb_a, emb_b)
                probs = torch.sigmoid(logits)
                
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(labels.numpy())
        
        # Metrics
        val_probs = np.array(val_probs)
        val_targets = np.array(val_targets)
        
        roc_auc = roc_auc_score(val_targets, val_probs)
        pr_auc = average_precision_score(val_targets, val_probs)
        
        logging.info(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f} | Val ROC-AUC={roc_auc:.4f} | Val PR-AUC={pr_auc:.4f}")
        
        # Save Best
        if pr_auc > best_val_prauc:
            best_val_prauc = pr_auc
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
            logging.info(f"🔥 New Best Model Saved! (PR-AUC: {best_val_prauc:.4f})")

if __name__ == "__main__":
    train_model()