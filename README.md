## 1. Project Snapshot

**Protein Design with ProGen2**  
An end-to-end workflow that fine-tunes the ProGen2 transformer on three Pfam families (PF00257, PF00069, PF00072) to generate novel, biologically plausible protein sequences. Covers:

- **Data acquisition & preprocessing**  
- **Model fine-tuning**  
- **Sequence generation**  
- **Comprehensive evaluation** (perplexity, sequence identity, HMMER recovery, attention analysis)

# Protein Design with ProGen2: Fine-tuning for Protein Family Generation

## Project Structure
```protein-design-progen2/
├── downloads/                  # Downloaded Pfam family sequences
├── checkpoints/                # Saved model checkpoints
├── generated_samples/          # Generated protein sequences
├── evaluation_results/         # Evaluation metrics and visualizations
│   ├── perplexity/             # Perplexity measurements 
│   ├── identity/               # Sequence identity analysis
│   └── attention/              # Attention pattern visualizations
├── models/                     # ProGen model architecture (imported)
└── src/                        # Source code
    ├── download_pfam.py        # Downloads protein families from Pfam
    ├── prepare_data.py         # Preprocesses sequences for training
    ├── finetune.py             # Fine-tunes ProGen2 on protein families
    ├── sample.py               # Interactive sequence generation
    ├── generate_sequences.py   # Batch sequence generation
    ├── evaluate_perplexity.py  # Measures model perplexity
    ├── evaluate_identity.py    # Computes sequence identity 
    ├── plot_attention.py       # Visualizes attention patterns
    └── plot_identity_comparison.py  # Compares model performance
```



## 2. Key Results at a Glance

- **Mean Sequence Identity**  
  - Single-family (PF00257): ~22%  
  - Multi-family (PF00257, PF00069, PF00072): ~25%

- **Perplexity Decrease**  
  - From ~1.7 (epoch 1) down to ~1.3 (epoch 3)

- **HMMER Recovery Rate**  
  - > 93% of generated sequences recognized by Pfam HMMs

- **Attention Patterns**  
  - Top heads in layer 12 capture biologically meaningful long-range couplings

## 3. Project Overview

This project implements an end-to-end pipeline for fine-tuning ProGen2, a state-of-the-art protein language model, to generate novel protein sequences with biological relevance. The workflow encompasses:

1. **Data acquisition** - Downloading protein family sequences from Pfam database
2. **Data preprocessing** - Preparing sequences with appropriate tokens for model training
3. **Model fine-tuning** - Adapting ProGen2 to specific protein families
4. **Sequence generation** - Creating novel protein sequences using the fine-tuned models
5. **Evaluation** - Comprehensive assessment of generated sequences using multiple metrics

The implementation supports both single-family and multi-family training, with mechanisms to compare performance between these approaches. The project demonstrates how transformer-based language models can effectively learn the underlying patterns and constraints of protein sequences, potentially accelerating protein engineering and design.

## 4. Installation & Dependencies

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU recommended for training

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/protein-design-progen2.git
   cd protein-design-progen2```
2. Create and activate a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```


3. Install dependencies:
```bash
pip install torch==1.10.0+cu113 
pip install transformers==4.18.0
pip install tokenizers==0.12.1
pip install biopython
pip install numpy pandas matplotlib tqdm
```

4.Download the base ProGen2 model
```bash
# The script will automatically pull the model from HuggingFace
# Default: hugohrban/progen2-small
```

## 5. Usage Guide

This section provides step-by-step instructions for running the complete protein design pipeline.

### 5.1 Data Acquisition

Download protein family sequences from Pfam using the `download_pfam.py` script:

```bash
python download_pfam.py PF00257 PF00069 PF00072
```

This will download the specified Pfam family sequences and save them as FASTA files in a `downloads` directory.

### 5.2 Data Preprocessing
Prepare the downloaded sequences for training using `prepare_data.py`:
```bash
python prepare_data.py --input_files downloads/PF00257.fasta downloads/PF00069.fasta downloads/PF00072.fasta --train_split_ratio 0.8 --output_file_train train_data.txt --output_file_test test_data.txt
```

Optional flags:

- --bidirectional: Enable bidirectional sequence representation
- --seed: Set random seed for reproducibility

### 5.3 Model Fine-tuning
Fine-tune the ProGen2 model on the prepared data:
```bash
python finetune.py --model hugohrban/progen2-small --train_file train_data.txt --test_file test_data.txt --epochs 3 --lr 1e-4 --batch_size 16 --device cuda
```
Key parameters:

- --epochs: Number of training epochs
- --lr: Learning rate
- --batch_size: Batch size for training
- --accumulation_steps: Gradient accumulation steps (default: 4)
- --checkpoint_rate: Save model checkpoint every N epochs (default: 5)
- --decay: Learning rate decay type (choices: cosine, linear, constant; default: cosine)

### 5.4 Sequence Generation
Generate novel protein sequences using the fine-tuned model:
```bash
python generate_sequences.py --model checkpoints/progen2-small-finetuned/e3 --families PF00257 PF00069 PF00072 --epoch 3 --batch_size 64 --iters 1
```
Options:

- --k: Top-k sampling parameter (default: 15)
- --t: Sampling temperature (default: 1.0)
- --max_length: Maximum sequence length (default: 1024)
- --device: Computing device (default: auto)

## 5.5 Evaluation
### 5.5.1 Perplexity
Evaluate model perplexity on test sequences:
```bash
python evaluate_perplexity.py --model checkpoints/progen2-small-finetuned/e3 --test_file test_data.txt --epoch 3 --tag single
```
### 5.5.2 Sequence Identity
Calculate sequence identity between generated and reference sequences:
```bash
python evaluate_identity.py --generated_fasta generated_samples/generated/PF00257_epoch3.fa --test_txt test_data.txt --family PF00257 --epoch 3 --tag single
```
Compare identity between single and multi-family models:
```bash
python plot_identity_comparison.py --single evaluation_results/identity/PF00257_epoch3_single_identities.npy --multi evaluation_results/identity/PF00257_epoch3_multi_identities.npy --family PF00257 --epoch 3
```
### 5.5.3 Attention Analysis
Visualize attention patterns in the model:
```bash
python plot_attention.py --model checkpoints/progen2-small-finetuned/e3 --test_file test_data.txt --sequence 0 --global_top_k 2 --max_display 100
```