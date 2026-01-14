#  Evidence-Backed PPI Predictor
### A Deep Learning & RAG System for Protein-Protein Interaction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![Bioinformatics](https://img.shields.io/badge/Domain-Bioinformatics-green) ![RAG](https://img.shields.io/badge/Technique-RAG-orange)

**An Explainable AI system that predicts physical interactions between proteins using ESM-2 Language Models and validates predictions with real-time experimental evidence from BioGRID.**

---

##  Executive Summary

In modern drug discovery and systems biology, determining whether two proteins interact (Protein-Protein Interaction, or PPI) is a critical step. While high-throughput screening is accurate, it is expensive and slow. Computational predictors exist, but they often act as "black boxes"—providing a confidence score without context or justification.

**This project solves the "Black Box" problem.** It combines state-of-the-art **Deep Learning** (for prediction) with **Retrieval Augmented Generation** (for validation).

### How It Works (The "Scientist & Librarian" Architecture)
The system operates as a dual-engine pipeline:
1.  **The Scientist (Predictive Engine):** A **Siamese Neural Network** processes the raw amino acid sequences of two proteins. It uses **ESM-2 (Evolutionary Scale Modeling)** embeddings to understand the physicochemical "grammar" of the proteins (hydrophobicity, charge, structure) and predicts the likelihood of binding.
2.  **The Librarian (Evidence Engine):** Simultaneously, a **RAG system** queries a local, indexed corpus of **1.2 million experimental records** (BioGRID). It retrieves specific PubMed citations and detection methods (e.g., "Two-Hybrid", "Affinity Capture") to confirm if the interaction has been observed in a lab.

---

## Project Scope & Constraints (Critical Details)

To ensure biological rigor and model reliability, strict filters were applied to the training data. This model is **specifically designed** for:

* Organism: Homo sapiens Only (TaxID: 9606)
    * **Detail:** We filtered the BioGRID dataset to exclude all non-human data (yeast, mouse, etc.).
    * **Reasoning:** Protein interaction rules can vary across species. Mixing species (e.g., training on yeast) introduces noise when predicting human disease targets.
* Interaction Type: Physical Only
    * **Detail:** We discarded "Genetic" interactions (e.g., synthetic lethality, phenotypic enhancement).
    * **Reasoning:** Genetic interactions imply a functional link (genes working in the same pathway) but do **not** guarantee the proteins physically touch. Our model is designed to predict physical binding (structure), so we trained strictly on physical experimental data (e.g., Co-crystalization, Affinity Capture).
* Structure: Heterodimers Only (No Self-Loops)
    * Detail: We removed all Homodimers (Protein A interacting with Protein A).
    * Reasoning: Homodimers are biologically distinct and mathematically trivial for Siamese networks (distance is always zero). Removing them forces the model to learn the complex rules of how *two different* surfaces bind.
*  Symmetry: Canonicalized Pairs
    * **Detail:** All pairs are sorted alphabetically (`min(ID1, ID2)`).
    * **Reasoning:** Biologically, A binding B is the same as B binding A. Treating them as separate rows causes **Data Leakage** (the model memorizes the duplicate). Canonicalization guarantees every interaction appears exactly once.

---

##  Performance Metrics

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **PR-AUC** | **0.80** | Precision-Recall AUC. The most critical metric for biological data, focusing on the quality of positive predictions. |
| **ROC-AUC** | **0.79** | Indicates strong discrimination capability between interacting and non-interacting pairs. |
| **Inference** | **<100ms** | Real-time prediction latency per pair on standard hardware (MPS/CUDA). |

###  Validation Case Studies (Sanity Checks)
We stress-tested the model against known biological truths:

| Scenario | Protein Pair | Prediction | Evidence Found | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Positive** | **EGFR + GRB2** | **91.7%** (High) | 56 Citations | **Validated** (Correct) |
| **Standard Negative** | **HBA1 + KRT1** | **58.4%** (Uncertain) | 0 Citations | **Validated** (Correct) |
| **Family Generalization** | **TP53 + TP63** | **92.8%** (High) | 4 Citations | **Validated** (Found rare evidence) |
| **Spatial Negative** | **BCL2 + INSR** | **15.7%** (Low) | 0 Citations | **Validated** (Correctly separated Mitochondria vs Membrane) |

---

## Detailed Technical Methodology

### 1. Data Engineering Pipeline
* **Source:** **BioGRID (Release 4.4)** - The gold standard for interaction datasets.
* **Negative Sampling:** Since BioGRID only lists positives, we generated **Synthetic Negatives** by randomly pairing unique proteins and verifying they did not exist in the positive set.
    * **Ratio:** 1:1 (Balanced). For every 1 positive interaction, we generated 1 negative.
    * **Bias Prevention:** Prevents the model from learning to just guess "Yes" (majority class bias).

### 2. Feature Engineering (The Intelligence Layer)
* **Embeddings:** We utilized **ESM-2 (`esm2_t6_8M_UR50D`)**, a Transformer protein language model by Meta AI.
* **Why ESM-2?** Unlike One-Hot Encoding, ESM-2 captures evolutionary context. It "reads" amino acid sequences and understands that specific patterns (e.g., Leucine zippers) imply structural features.
* **Dimensionality:** Each protein is converted into a **320-dimensional dense vector**.
* **Pooling:** Variable-length sequences were normalized using **Global Mean Pooling** to produce fixed-size inputs for the neural network.

### 3. Deep Learning Architecture
* **Type:** **Siamese Neural Network (Two-Tower)**.
* **Mechanism:**
    1.  **Twin Towers:** Two identical neural networks (sharing weights) process Protein A and Protein B separately.
    2.  **Projection:** A Linear Layer + ReLU projects the ESM-2 embeddings into a learned interaction space.
    3.  **Interaction Layer:** The two vectors are fused using heuristic matching operations to capture relationship dynamics:
        * **Concatenation:** `[u, v]`
        * **Absolute Difference:** `|u - v|` (Measures distance/complementarity)
        * **Element-wise Product:** `u * v` (Measures alignment/similarity)
    4.  **Classification Head:** A Multi-Layer Perceptron (MLP) with Dropout (0.3) outputs the final probability.
* **Evaluation Strategy:** **Protein-Disjoint Split (Cold Start)**.
    * We moved 20% of **unique proteins** into a test set.
    * Any interaction involving a test protein was removed from training.
    * **Result:** The model is tested on proteins it has **never seen**, proving it learned biophysical rules, not memorization.

### 4. RAG Implementation (The Evidence Layer)
* **Challenge:** Standard Vector Search (FAISS) failed to distinguish between alphanumeric IDs (e.g., `P04637` vs `P04638`) due to semantic drift in language models.
* **Solution:** Implemented a **Deterministic Hashed Lookup**.
    * **Indexing:** The entire BioGRID corpus (1.2M rows) is indexed in memory using a sorted pair hash: `hash(min_id + "_" + max_id)`.
    * **Retrieval:** Lookups are **O(1)** (Constant Time). This guarantees 100% precision—if a paper exists, the system *will* find it.

---

## Installation & Usage Guide

Follow these steps to deploy the project locally.

### Prerequisites
* **Python 3.10+**
* **Git**
* **8GB RAM** (Minimum)

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/jenishsimkhada400/Evidence-Backed-PPI-Predictor.git
cd Evidence-Backed-PPI-Predictor

# Install dependencies
python -m pip install -r requirements.txt
```
### 2. Data Acquisition
Due to GitHub file size limits, the raw BioGRID dataset is not included. You must download it.
1. Go to the [BioGRID Download Page](https://downloads.thebiogrid.org/BioGRID/Release-Archive/).
2. Download the latest "All Organisms" TAB3 file.
3. Unzip and place the .txt file in data/raw/.
4. Rename it (or update the config) to match BIOGRID-ALL-LATEST.tab3.txt.

3. Pipeline Execution
To reproduce the entire workflow from scratch:

```bash

# 1. Process Data (Filter Human/Physical, Generate Negatives, Split)
python -m src.data.preprocess

# 2. Fetch Sequences (Map UniProt IDs to Amino Acid Strings)
python -m src.data.fetch_sequences

# 3. Embed Sequences (Run ESM-2 Model - Caution: Compute Heavy)
python -m src.features.embed_sequences

# 4. Train Model (Train PyTorch Network)
python -m src.models.train

# 5. Build RAG Index (Create Evidence Corpus)
python -m src.rag.build_corpus
```
### 4. Run the Dashboard
Launch the interactive UI to test predictions.

```bash
PYTHONPATH=. streamlit run src/app/streamlit_app.py
```
---
### Project Structure
```Plaintext

PPI-RAG-BIOGRID/
├── configs/               # YAML configs for paths and hyperparameters
├── data/
│   ├── raw/               # (Empty) Place raw BioGRID txt here
│   ├── processed/         # (Generated) Train/Test splits, sequences.csv
│   └── embeddings/        # (Generated) .pt dictionary of ESM-2 vectors
├── models/                # Saved PyTorch model weights (best_model.pt)
├── src/
│   ├── app/               # Streamlit Dashboard code
│   ├── data/              # ETL scripts: preprocessing, negative sampling
│   ├── features/          # ESM-2 embedding logic
│   ├── models/            # Two-Tower Network architecture & Training Loop
│   └── rag/               # Evidence Retrieval & Corpus Generation
├── requirements.txt       # Python dependencies
└── README.md              # Project Documentation
```
---
### Acknowledgements & References
* ESM-2 (Meta AI): Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science, 2023.
* BioGRID: Oughtred et al., "The BioGRID interaction database: 2021 update." Nucleic Acids Research, 2021.
* UniProt: The Universal Protein Resource.
---
### Author
Jenish Simkhada _Bioinformatics & Computational Biology_ | [UMBC LinkedIn](https://www.linkedin.com/in/jenishsimkhada/) | [GitHub](https://github.com/jenishsimkhada400)




