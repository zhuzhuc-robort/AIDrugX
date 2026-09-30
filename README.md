# AIDrugX

**AIDrugX: Ligand-Conditioned Protein Binding-Site Prediction with Multi-View Fusion.**

AIDrugX predicts, at single-residue resolution, whether each amino acid in a protein sequence is likely to contact a given small-molecule ligand. It fuses protein language model embeddings, 1D convolutional sequence features, ligand language model embeddings, and molecular graph representations via cross-modal attention, and produces a per-residue binding score that can be aggregated into a protein–ligand interaction score.

<img width="2560" height="1440" alt="Snipaste_2026-09-30_16-05-39" src="https://github.com/user-attachments/assets/e21ca3de-3b19-4021-adbd-c04d356c6b0d" />




## Table of Contents

- [Background and Motivation](#background-and-motivation)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Downloading Pretrained Weights](#downloading-pretrained-weights)
- [Quick Start](#quick-start)
- [Input Format](#input-format)
- [Output Files](#output-files)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Background and Motivation

Identifying which residues on a protein surface interact with a given small molecule is a fundamental problem in:

- **Structure-based drug design** — locating binding pockets before docking
- **Virtual screening** — ranking candidate ligands by predicted interaction likelihood
- **Protein function annotation** — inferring unknown binding sites from sequence alone
- **Mutagenesis planning** — prioritizing residues for experimental validation

Most existing pocket predictors rely on 3D structure and are sensitive to conformational changes. AIDrugX instead works directly on **protein sequence + ligand SMILES**, making it applicable when only a sequence is available — the common case for newly discovered proteins, engineered variants, or AlphaFold-predicted structures.

The model is trained on ~150,000 protein–ligand pairs derived from the PDB, with residue-level labels defined by a 5 Å heavy-atom contact criterion.

---

## How It Works

AIDrugX is a **per-residue binary classifier** with four parallel branches:

```
                Protein                              Ligand
   ┌──────────────────────────┐          ┌──────────────────────────┐
   │  Sequence                │          │  SMILES                  │
   └──────────┬───────────────┘          └──────────┬───────────────┘
              │                                     │
     ┌────────┴────────┐                  ┌─────────┴──────────┐
     │ ESM-2 (650M)    │                  │ MolFormer          │
     │ chunked encoder │                  │ (LM_Mol)           │
     └────────┬────────┘                  └─────────┬──────────┘
              │                                     │
     ┌────────┴────────┐                  ┌─────────┴──────────┐
     │ 1D ResNet CNN   │                  │ GAT (DGL)          │
     │ (one-hot)       │                  │ molecular graph    │
     └────────┬────────┘                  └─────────┬──────────┘
              │                                     │
              └──────────────┬──────────────────────┘
                             │
                 Cross-modal attention fusion
                             │
                  Per-residue MLP decoder
                             │
                    (L,) binding scores
```

**Key ideas:**

1. **Protein side** — ESM-2 provides rich contextual embeddings; a 1D CNN captures local sequence motifs. Both are kept at full length so every residue gets its own representation.
2. **Ligand side** — MolFormer encodes SMILES at the token level; a Graph Attention Network encodes the molecular graph. Both are pooled into a single ligand vector.
3. **Fusion** — The ligand vector is broadcast to every residue position, then cross-attended with protein features through multiple attention blocks. This lets each residue "ask" whether the ligand is compatible with its local environment.
4. **Output** — For a protein of length *L*, the model outputs *L* scores in [0, 1]. Aggregating the top-k scores (default k = 20) gives a protein–ligand interaction score.

**Performance** (internal validation set):

| Metric | Value |
|---|---|
| Residue-level AUC | ~0.92 |
| Residue-level PRC | ~0.54 |
| Top-20 recall | ~0.58 |
| Top-50 recall | ~0.80 |

---

## Repository Structure

```
AIDrugX/
├── predict.py                                # Main inference script
├── environment.yml                           # Conda environment
├── README.md                                 # This file
├── trained_model_v5_residue/
│   └── best_model.pth                        # ← download separately (see below)
├── LM_Mol/
│   ├── bert_vocab.txt
│   ├── tokenizer.py
│   ├── rotate_builder.py
│   ├── ...
│   └── check_points/                         # ← download separately (see below)
│       └── N-Step-Checkpoint_3_30000.ckpt
└── examples/
    └── input_example.csv                     # Demo input
```

---

## Installation

### Prerequisites

- Ubuntu 20.04 / 22.04 (or WSL2 on Windows)
- CUDA-capable GPU recommended (CPU inference works but is slow)
- ~15 GB disk space (ESM-2 650M model + weights)

### Step 1 — Create conda environment

```bash
conda create -n AIDrugX python=3.8.19 -y
conda activate AIDrugX
```

### Step 2 — Install PyTorch with CUDA 12.1

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

> If your CUDA version is different, adjust the wheel URL accordingly.  
> For CPU-only: `pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1`

### Step 3 — Install compiler toolchain (needed to build DGL)

```bash
conda install -c conda-forge gcc=11 gxx=11
```

### Step 4 — Install remaining dependencies

```bash
conda env update -n AIDrugX -f environment.yml
```

This installs `dgl`, `dgllife`, `transformers`, `fast-transformers`, `pandas`, `numpy`, `tqdm`, `scikit-learn`, `biopython`, `matplotlib`, and other utilities.

### Step 5 — Verify installation

```bash
python -c "import torch, dgl, dgllife, transformers; \
           print('torch:', torch.__version__); \
           print('cuda available:', torch.cuda.is_available()); \
           print('dgl:', dgl.__version__)"
```

If `cuda available: True`, you are ready.

---

## Downloading Pretrained Weights

The two model weight files are **too large for GitHub** and are hosted separately.

### Download link

> **Baidu / SJTU Cloud:**  
> Link: https://pan.sjtu.edu.cn/web/share/e40d33cf8a57b0a3b37b54e8c8a14e30  
> Access code: **1122**

### Files to download

| File | Size | Destination |
|---|---|---|
| `best_model.pth` | ~100 MB | `AIDrugX/trained_model_v5_residue/best_model.pth` |
| `N-Step-Checkpoint_3_30000.ckpt` | ~500 MB | `AIDrugX/LM_Mol/check_points/N-Step-Checkpoint_3_30000.ckpt` |

### Correct placement (critical)

After downloading, your directory tree should look **exactly** like this:

```
AIDrugX/
├── predict.py
├── trained_model_v5_residue/
│   └── best_model.pth                        ✅ here
└── LM_Mol/
    ├── bert_vocab.txt
    ├── tokenizer.py
    └── check_points/
        └── N-Step-Checkpoint_3_30000.ckpt    ✅ here
```

If the paths differ, edit the top of `predict.py`:

```python
MODEL_PATH      = "trained_model_v5_residue/best_model.pth"
CHECKPOINT_PATH = "LM_Mol/check_points/N-Step-Checkpoint_3_30000.ckpt"
```

### ESM-2 model

`facebook/esm2_t33_650M_UR50D` (~2.6 GB) is downloaded automatically from HuggingFace on first run.

If you are in mainland China, the script already sets `HF_ENDPOINT=https://hf-mirror.com`. To pre-download manually:

```bash
huggingface-cli download facebook/esm2_t33_650M_UR50D \
    --cache-dir ~/.cache/huggingface
```

---

## Quick Start

```bash
# 1. Activate environment
conda activate AIDrugX

# 2. Run the built-in demo
python predict.py --input examples/input_example.csv --output predictions.csv
```

You should see progress logs for ESM-2 encoding, MolFormer encoding, and per-protein inference, followed by:

```
======================================================================
  Prediction Complete
======================================================================
  Total pairs          : 5
  Score range          : 0.4123 ~ 0.6234
  ...
```

All results will be in `predictions.csv` and `predictions_plots/`.

---

## Input Format

Your input file must be a CSV with **at minimum** two columns: `smiles` and `sequence`.

```csv
smiles,sequence
CC(=O)Oc1ccccc1C(=O)O,MKTAYSDKLPGE...
c1ccccc1,MEEPQSDF...
O=C(N)c1ccc[n+](c1)C2CC(C(O)C2O)COP(=O)(O)OP(=O)(O)OCC5OC(n4cnc3c(ncnc34)N)C(O)C5O,GSENVEVFTAEGKGRGLKATKEFWA...
```

**Column name requirements:**

| Column | Meaning | Default name |
|---|---|---|
| SMILES | Small-molecule structure | `smiles` |
| Sequence | Protein amino-acid sequence (1-letter) | `sequence` |

If your CSV uses different names (e.g., `SMILES`, `Protein`), use:

```bash
python predict.py --input my_data.csv --smiles-col SMILES --sequence-col Protein
```

Additional columns (e.g., `pdb_id`, `ligand`, `label`) are preserved in the output.

---

## Output Files

Running `predict.py` produces **four** output artifacts:

### 1. `predictions.csv` — main result table

One row per (SMILES, sequence) pair. Adds these columns to your input:

| Column | Description |
|---|---|
| `prot_len` | Protein sequence length |
| `score` | Aggregate binding score (top-20 mean by default) |
| `top1_score` | Highest per-residue score |
| `top5_mean` | Mean of top-5 residue scores |
| `top20_mean` | Mean of top-20 residue scores |
| `top50_mean` | Mean of top-50 residue scores |
| `n_pos_residues` | Number of residues with score > 0.5 |

### 2. `predictions.residue_scores.csv` — per-residue scores

One row per residue per pair:

```csv
row_id,smiles,sequence_len,position,residue,score
0,O=C(N)c1ccc...,351,0,K,0.2134
0,O=C(N)c1ccc...,351,1,S,0.1987
0,O=C(N)c1ccc...,351,2,K,0.3012
...
```

Useful for downstream analysis: `groupby("row_id")`, `nlargest(10, "score")`, etc.

### 3. `predictions.residue_scores.npz` — compressed NumPy archive

Faster to load than CSV. Keys are `row_0`, `row_1`, ...

```python
import numpy as np
data = np.load("predictions.residue_scores.npz")
scores_row0 = data["row_0"]   # shape (L,)
```

### 4. `predictions_plots/` — one PNG per pair

Each plot shows:
- **Top panel** — per-residue score curve; red dots mark top-k residues; top-1 position annotated
- **Bottom panel** — cumulative contribution curve

---

## Usage Examples

### Basic — predict on a CSV

```bash
python predict.py --input my_data.csv --output results.csv
```

### Only the main table (skip plots and residue scores)

```bash
python predict.py --input my_data.csv --output results.csv \
    --no-plot --no-save-residue-scores
```

This is **much faster** and uses **much less disk space** for large inputs.

### Custom top-k in plots

```bash
python predict.py --input my_data.csv --output results.csv \
    --plot-top-k 10 --plot-threshold 0.3
```

### Limit the number of plots

```bash
python predict.py --input my_data.csv --output results.csv \
    --plot-max 50
```

Only the first 50 pairs are plotted — useful when handling thousands of pairs.

### Larger batch size (faster GPU inference)

```bash
python predict.py --input my_data.csv --output results.csv --batch-size 64
```

### Force CPU

```bash
python predict.py --input my_data.csv --output results.csv --device cpu
```

### Full help

```bash
python predict.py --help
```

---

## Interpreting the Scores

The model output is a **per-residue confidence** in the range (0, 1). It is **not a calibrated probability**.

| Score range | Interpretation |
|---|---|
| > 0.7 | High confidence the residue contacts the ligand |
| 0.5 – 0.7 | Moderate confidence; usually still in the true pocket |
| 0.3 – 0.5 | Weak signal — probably background |
| < 0.3 | Almost certainly not in contact |

**Best practices:**

- Use `score` or `top20_mean` to **rank** ligands. Higher = more likely to bind.
- Use `n_pos_residues` to see how many residues the model "fires" on.
- Do **not** read `score = 0.55` as "55% probability of binding." The training loss (focal) does not produce calibrated probabilities.
- For a fixed protein, compare scores across ligands — the *relative* ranking is what matters.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: trained_model_v5_residue/best_model.pth` | Weights not downloaded or placed wrong | See [Downloading Pretrained Weights](#downloading-pretrained-weights) |
| `ModuleNotFoundError: LM_Mol` | `LM_Mol/` directory missing | Make sure it is at `AIDrugX/LM_Mol/` |
| `ModuleNotFoundError: fast_transformers` | Missing dependency | `pip install fast-transformers` |
| `CUDA out of memory` | Sequence too long | Reduce `--batch-size`; or set `ESM_MAX_LEN=1000` in `predict.py` |
| `expected scalar type Half but found Float` | You modified `ESM2Encoder` to use `torch_dtype=torch.float16` | Do **not** set `torch_dtype` when loading ESM-2 |
| Slow inference | Large protein or many ligands | Increase `--batch-size`, use GPU, disable plots |
| Empty output / all NaN | All SMILES failed graph construction | Check SMILES validity |

---

## Citation

If you use AIDrugX in your research, please cite:

```bibtex
@software{AIDrugX2024,
  title  = {AIDrugX: Deep Learning for Protein--Ligand Binding Site Prediction},
  author = {<Your Name>},
  year   = {2024},
  url    = {https://github.com/<your-org>/AIDrugX}
}
```

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgments

- [ESM-2](https://github.com/facebookresearch/esm) by Meta AI
- [MolFormer / LM_Mol](https://github.com/IBM/molformer) by IBM Research
- [DGL-LifeSci](https://github.com/awslabs/dgl-lifesci) by AWS
- Training data derived from the [RCSB PDB](https://www.rcsb.org/)

For questions or bug reports, please open a GitHub issue.
