#!/usr/bin/env python3
"""
Generate and clean sequences for multiple Pfam families from a fine-tuned ProGen2 checkpoint.
Sequences are saved under a root directory 'generated_samples'.
"""
import os
import argparse
import torch
from sample import sample, truncate
from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM

# Disable Rust-tokenizer parallelism if needed
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Root directory for all generated samples
ROOT_DIR = "generated_samples"

def main():
    parser = argparse.ArgumentParser(
        description="Generate cleaned sequences for multiple families")
    parser.add_argument("--model", required=True,
                        help="Path to checkpoint directory (e.g. .../e3)")
    parser.add_argument("--families", nargs='+', required=True,
                        help="List of Pfam family codes, e.g. PF00257 PF00069")
    parser.add_argument("--epoch", required=True,
                        help="Epoch label to tag outputs (e.g. 1)")
    parser.add_argument("--output_dir", default="generated",
                        help="Subdirectory under generated_samples to save FASTA files")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="# sequences per batch per iteration")
    parser.add_argument("--iters", type=int, default=1,
                        help="# sampling iterations per family (total = iters * batch_size)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum generation length including prompt")
    parser.add_argument("--k", type=int, default=15,
                        help="Top-k sampling parameter (0 = no top-k)")
    parser.add_argument("--t", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, mps, or cpu")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Set seeds
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Load model and tokenizer
    model = ProGenForCausalLM.from_pretrained(args.model).to(device).eval()
    tok_path = os.path.join(args.model, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        tokenizer = Tokenizer.from_pretrained(args.model)
    tokenizer.no_padding()

    # Prepare root output directory
    full_root = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(full_root, exist_ok=True)

    # Generate for each family
    for fam in args.families:
        fam_lower = fam.lower()
        prompt = f"<|{fam_lower}|>1"
        all_sequences = []
        # sampling loop
        for _ in range(args.iters):
            seqs = sample(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_length=args.max_length,
                num_return_sequences=args.batch_size,
                temperature=args.t,
                top_k=(None if args.k == 0 else args.k),
            )
            all_sequences.extend(seqs)

        # Clean sequences (remove tokens)
        cleaned = [truncate(s) for s in all_sequences]

        # Write to FASTA under generated_samples/<output_dir>/
        fam_dir = os.path.join(full_root)
        out_path = os.path.join(
            fam_dir,
            f"{fam}_epoch{args.epoch}.fa"
        )
        with open(out_path, 'w') as f:
            for idx, seq in enumerate(cleaned):
                f.write(f">{fam}_seq{idx}\n")
                f.write(seq + "\n")

        print(f"Saved {len(cleaned)} sequences for {fam} to {out_path}")

if __name__ == "__main__":
    main()
