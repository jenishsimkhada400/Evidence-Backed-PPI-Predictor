import torch
import torch.nn as nn

class TwoTowerPPI(nn.Module):
    def __init__(self, embed_dim=320, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # 1. Projectors (Optional: refine the raw ESM embeddings)
        # We assume input is already 320-dim from ESM-2 (t6_8M)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Interaction Combiner
        # We combine vector A and B into a single feature vector:
        # [A, B, |A - B|, A * B]
        # 320 + 320 + 320 + 320 = 1280 dimensions
        combined_dim = embed_dim * 4
        
        # 3. Classifier (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1) # Output: single logit
        )
        
    def forward(self, emb_a, emb_b):
        # Refine embeddings
        feat_a = self.projector(emb_a)
        feat_b = self.projector(emb_b)
        
        # Feature Engineering inside the model
        # Absolute difference captures "distance"
        # Element-wise product captures "similarity/alignment"
        diff = torch.abs(feat_a - feat_b)
        prod = feat_a * feat_b
        
        # Concatenate
        combined = torch.cat([feat_a, feat_b, diff, prod], dim=1)
        
        # Predict
        logits = self.classifier(combined)
        return logits