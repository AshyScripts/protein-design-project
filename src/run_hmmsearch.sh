#!/usr/bin/env bash
set -euo pipefail

# Families as they appear in your generated FASTA filenames
FAMILIES=(
  PF00257
  PF00069_truncated
  PF00072_truncated
)

PROFILE_DIR="pfam_profiles"
GEN_DIR="generated_samples/multi-family"
OUT_DIR="."
SUMMARY="$OUT_DIR/hmm_summary.csv"

# CSV header
echo "family,epoch,total_seqs,matches,pct_match" > "$SUMMARY"

for fam in "${FAMILIES[@]}"; do
  # map truncated names back to HMM filenames
  base="${fam%_truncated}"
  hmmfile="$PROFILE_DIR/${base}.hmm"
  fasta="$GEN_DIR/${fam}_epoch3.fa"
  tblout="$OUT_DIR/${fam}_multiE3.tbl"

  # sanity checks
  [[ -f "$hmmfile" ]] || { echo "ERROR: missing $hmmfile" >&2; exit 1; }
  [[ -f "$fasta"  ]] || { echo "ERROR: missing $fasta"  >&2; exit 1; }

  # press/index if needed
  if [[ ! -f "${hmmfile}.h3m" ]]; then
    echo ">>> Pressing $hmmfile"
    hmmpress "$hmmfile"
  fi

  echo ">>> Running hmmsearch for $fam → $tblout"
  hmmsearch --noali --tblout "$tblout" "$hmmfile" "$fasta" >/dev/null

  # count total vs. unique matches
  total_seqs=$(grep -c '^>' "$fasta")
  matches=$(grep -v '^#' "$tblout" | awk '{print $1}' | sort -u | wc -l)

  # compute percentage with bc and format
  raw_pct=$(echo "scale=4; $matches*100/$total_seqs" | bc)
  pct=$(printf "%.1f" "$raw_pct")

  echo "  → $fam: $matches / $total_seqs sequences matched ($pct%)"
  echo "${base},3,${total_seqs},${matches},${pct}" >> "$SUMMARY"
done

echo
echo "All done — summary written to $SUMMARY"
