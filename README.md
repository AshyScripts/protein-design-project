## 1. Project Snapshot

**Protein Design with ProGen2**  
An end-to-end workflow that fine-tunes the ProGen2 transformer on three Pfam families (PF00257, PF00069, PF00072) to generate novel, biologically plausible protein sequences. Covers:

- **Data acquisition & preprocessing**  
- **Model fine-tuning**  
- **Sequence generation**  
- **Comprehensive evaluation** (perplexity, sequence identity, HMMER recovery, attention analysis)

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

