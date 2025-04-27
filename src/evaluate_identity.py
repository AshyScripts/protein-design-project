#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO, pairwise2

# Disable Rust-tokenizer parallelism if needed
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def clean_test_seq(line: str, family: str) -> str:
    """
    Extract raw amino-acid sequence from a test .txt line.
    Removes <|family|> prefix and terminus tokens 1/2; returns None if no match.
    """
    prefix = f"<|{family.lower()}|>"
    if not line.startswith(prefix):
        return None
    seq = line[len(prefix):]
    if seq and seq[0] in ['1', '2']:
        seq = seq[1:]
    if seq and seq[-1] in ['1', '2']:
        seq = seq[:-1]
    return seq or None


def compute_identity(seq1: str, seq2: str) -> float:
    """
    Global alignment identity; returns 0.0 if sequences empty or no alignment.
    """
    if not seq1 or not seq2:
        return 0.0
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True,
                                          penalize_end_gaps=False)
    if not alignments:
        return 0.0
    a, b, score, start, end = alignments[0]
    return score / len(a) if len(a) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sequence identity between generated and reference sequences."
    )
    parser.add_argument("--generated_fasta", required=True,
                        help="FASTA of generated sequences")
    parser.add_argument("--test_txt", required=True,
                        help="Mixed test .txt with one preprocessed sequence per line")
    parser.add_argument("--family", required=True,
                        help="Pfam code, e.g. PF00257 or PF00069_truncated")
    parser.add_argument("--epoch", required=True,
                        help="Epoch label for output naming")
    parser.add_argument("--tag", default="",
                        help="Optional tag to differentiate runs (e.g. single, multi)")
    parser.add_argument("--output_dir", default="evaluation_results/identity",
                        help="Root dir to save stats and plots")
    args = parser.parse_args()

    # Define a hierarchical output path: output_dir/family/epoch_tag
    family_dir = os.path.join(args.output_dir, args.family)
    # Build run directory name
    run_name = f"epoch_{args.epoch}" + (f"_{args.tag}" if args.tag else "")
    run_dir = os.path.join(family_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Load and clean test sequences
    test_seqs = []
    with open(args.test_txt) as f:
        for line in f:
            clean = clean_test_seq(line.strip(), args.family)
            if clean:
                test_seqs.append(clean)
    if not test_seqs:
        raise ValueError(f"No test sequences for family {args.family} in {args.test_txt}")

    # Load generated sequences, filter out empties
    gen_records = [rec for rec in SeqIO.parse(args.generated_fasta, "fasta") if len(rec.seq) > 0]
    if not gen_records:
        raise ValueError(f"No non-empty sequences in {args.generated_fasta}")

    # Compute identity for each generated vs. reference (cycled)
    identities = []
    for i, rec in enumerate(gen_records):
        gen_seq = str(rec.seq)
        ref_seq = test_seqs[i % len(test_seqs)]
        identities.append(compute_identity(gen_seq, ref_seq))
    ids = np.array(identities)
    
    # dump identities for later comparison
    out_npy = os.path.join(
       args.output_dir,
       f"{args.family}_epoch{args.epoch}_{args.tag}_identities.npy"
    )
    np.save(out_npy, ids)
    print(f"Saved raw identities to {out_npy}")

    # Summary statistics
    stats = {
        'num_samples':       len(ids),
        'mean_identity':     float(ids.mean()),
        'median_identity':   float(np.median(ids)),
        'std_identity':      float(ids.std()),
        'frac_ge_50':        float((ids >= 0.50).mean()),
        'frac_ge_80':        float((ids >= 0.80).mean()),
    }
    stats_df = pd.DataFrame(stats.items(), columns=['metric', 'value'])
    stats_file = os.path.join(run_dir, f"identity_stats.csv")
    stats_df.to_csv(stats_file, index=False)

    # Plot histogram
    hist_file = os.path.join(run_dir, f"identity_hist.png")
    plt.figure()
    plt.hist(ids, bins=20, range=(0,1), edgecolor='black')
    plt.xlabel('Sequence identity')
    plt.ylabel('Count')
    plt.title(f'{args.family} Identity Histogram (epoch {args.epoch}{" "+args.tag if args.tag else ""})')
    plt.savefig(hist_file)
    plt.close()

    # Plot cumulative distribution
    cdf_file = os.path.join(run_dir, f"identity_cdf.png")
    sorted_ids = np.sort(ids)
    cdf = np.arange(1, len(sorted_ids)+1) / len(sorted_ids)
    plt.figure()
    plt.plot(sorted_ids, cdf, marker='.', linestyle='none')
    plt.xlabel('Sequence identity')
    plt.ylabel('Cumulative fraction')
    plt.title(f'{args.family} Identity CDF (epoch {args.epoch}{" "+args.tag if args.tag else ""})')
    plt.savefig(cdf_file)
    plt.close()

    print(f"Saved identity stats to: {stats_file}")
    print(f"Saved histogram to:     {hist_file}")
    print(f"Saved CDF to:           {cdf_file}")

if __name__ == "__main__":
    main()
