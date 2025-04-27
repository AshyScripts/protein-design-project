#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--single",  required=True,
                    help=".npy file of single-family identities")
parser.add_argument("--multi",   required=True,
                    help=".npy file of multi-family identities")
parser.add_argument("--family",  required=True)
parser.add_argument("--epoch",   required=True)
parser.add_argument("--out_dir", default="evaluation_results/identity")
args = parser.parse_args()

# load
ids_single = np.load(args.single)
ids_multi  = np.load(args.multi)

# histogram
plt.figure()
bins = np.linspace(0,1,21)
plt.hist(ids_single, bins=bins, alpha=0.6, color="C0", label="Single")
plt.hist(ids_multi,  bins=bins, alpha=0.6, color="C1", label="Multi")
plt.xlabel("Sequence identity")
plt.ylabel("Count")
plt.title(f"{args.family} Identity Histogram (epoch {args.epoch})")
plt.legend()
hist_png = os.path.join(args.out_dir, f"{args.family}_epoch{args.epoch}_identity_hist_compare.png")
plt.savefig(hist_png, bbox_inches="tight")
plt.close()

# CDF
plt.figure()
for ids, label, col in [(ids_single,"Single","C0"), (ids_multi,"Multi","C1")]:
    s = np.sort(ids)
    cdf = np.arange(1, len(s)+1) / len(s)
    plt.plot(s, cdf, marker=".", linestyle="none", color=col, label=label)
plt.xlabel("Sequence identity")
plt.ylabel("Cumulative fraction")
plt.title(f"{args.family} Identity CDF (epoch {args.epoch})")
plt.legend()
cdf_png = os.path.join(args.out_dir, f"{args.family}_epoch{args.epoch}_identity_cdf_compare.png")
plt.savefig(cdf_png, bbox_inches="tight")
plt.close()

print("Wrote:")
print(" ", hist_png)
print(" ", cdf_png)
