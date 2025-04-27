import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

# Paths
generated_fasta = "generated_samples/e3/samples_ctx<|pf00257|>1_k15_t1.0.fa"
pfam_hmm = "pfam_profiles/PF00257.hmm"
hmmer_output = "evaluation_results/hmmer_results.txt"
hmmer_tblout = "evaluation_results/hmmer_tblout.txt"

# Create output directory
os.makedirs("evaluation_results", exist_ok=True)

# Filter out empty sequences
valid_sequences_file = "evaluation_results/valid_sequences.fa"
valid_count = 0

with open(valid_sequences_file, "w") as outfile:
    for record in SeqIO.parse(generated_fasta, "fasta"):
        if len(record.seq.strip()) > 0:
            SeqIO.write(record, outfile, "fasta")
            valid_count += 1

print(f"Filtered {valid_count} valid sequences")

# Run HMMER
try:
    # Run hmmsearch with output to a text file and table format
    subprocess.run(
        f"hmmsearch --tblout {hmmer_tblout} {pfam_hmm} {valid_sequences_file} > {hmmer_output}",
        shell=True,
        check=True,
    )
    print(f"HMMER search completed successfully")
    
    # Parse HMMER results
    matches = []
    with open(hmmer_tblout, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) > 0:
                seq_id = parts[0]
                e_value = float(parts[4])
                score = float(parts[5])
                matches.append((seq_id, e_value, score))
    
    # Calculate statistics
    match_count = len(matches)
    match_percentage = (match_count / valid_count) * 100
    
    print(f"\nHMMER Analysis Results:")
    print(f"Total valid sequences analyzed: {valid_count}")
    print(f"Sequences matching PF00257 family: {match_count}")
    print(f"Percentage of sequences in family: {match_percentage:.2f}%")
    
    # Create a DataFrame for further analysis
    if matches:
        results_df = pd.DataFrame(matches, columns=["sequence_id", "e_value", "score"])
        
        # Save to CSV
        results_df.to_csv("evaluation_results/hmmer_results.csv", index=False)
        
        # Plot E-value distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df["e_value"].apply(lambda x: -1 * np.log10(x + 1e-300)), bins=20)
        plt.xlabel("-log10(E-value)")
        plt.ylabel("Count")
        plt.title("Distribution of HMMER E-values")
        plt.savefig("evaluation_results/hmmer_evalue_distribution.png")
        plt.show()
        
        # Plot score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df["score"], bins=20)
        plt.xlabel("HMMER Bit Score")
        plt.ylabel("Count")
        plt.title("Distribution of HMMER Bit Scores")
        plt.savefig("evaluation_results/hmmer_score_distribution.png")
        plt.show()
        
        print("\nTop 5 sequences with highest scores:")
        print(results_df.sort_values("score", ascending=False).head(5))
    
    # Visualize match percentage
    plt.figure(figsize=(8, 8))
    plt.pie(
        [match_count, valid_count - match_count],
        labels=["Matches PF00257", "No match"],
        autopct="%1.1f%%",
        colors=["#66b3ff", "#ff9999"],
        explode=(0.1, 0),
        shadow=True,
    )
    plt.title("HMMER Analysis: Generated Sequences Matching PF00257 Family")
    plt.savefig("evaluation_results/hmmer_match_percentage.png")
    plt.show()
    
except Exception as e:
    print(f"Error running HMMER: {e}")
    print("Please ensure HMMER is correctly installed and the HMM file exists")