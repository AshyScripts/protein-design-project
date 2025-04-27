#!/usr/bin/env python3
"""
Plot the top ‘non-local’ attention heads for a protein sequence, optionally limiting
visualization to the first N residues to highlight local structure.
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM

# disable tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def load_attention(model_path, test_file, sequence_idx, device):
    model = ProGenForCausalLM.from_pretrained(model_path).to(device).eval()
    tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
    tokenizer.no_padding()

    lines = open(test_file).read().splitlines()
    seq = lines[sequence_idx]
    ids = torch.tensor(tokenizer.encode(seq).ids, dtype=torch.long).unsqueeze(0).to(device)
    seq_len = ids.shape[-1]

    with torch.no_grad():
        outputs = model(ids, output_attentions=True)
    attns = torch.stack([a.squeeze(0) for a in outputs.attentions], dim=0)
    return attns, seq_len


def compute_energy_layer(layer_attn, window):
    _, S, _ = layer_attn.shape
    idx = torch.arange(S, device=layer_attn.device)
    di = idx.view(-1,1) - idx.view(1,-1)
    mask = (di.abs() > window)
    masked = layer_attn * mask.unsqueeze(0)
    return masked.sum(dim=(1,2))


def find_top_heads(attns, top_k, window, layer_idx=None):
    n_layers, H, _, _ = attns.shape
    results = []
    if layer_idx is not None:
        energy = compute_energy_layer(attns[layer_idx], window)
        for h in range(H):
            results.append((energy[h].item(), layer_idx, h))
    else:
        for l in range(n_layers):
            energy = compute_energy_layer(attns[l], window)
            for h in range(H):
                results.append((energy[h].item(), l, h))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def plot_and_save(attns, seq_len, head_info, output_dir, max_display):
    energy, layer, head = head_info
    # determine display length
    N = min(seq_len, max_display)
    data = attns[layer, head][:N, :N].cpu().numpy()

    masked = np.ma.masked_equal(data, 0.0)
    if masked.count() > 0:
        vmin = masked.min()
        vmax = masked.max()
    else:
        vmin, vmax = 1e-8, 1.0
    cmap = plt.cm.Blues
    cmap.set_bad(color='white')

    plt.figure(figsize=(5,5))
    im = plt.imshow(masked, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), aspect='equal')
    plt.title(f"Layer {layer+1}, Head {head} (Energy={energy:.1f})")
    plt.xlabel('Key idx')
    plt.ylabel('Query idx')
    cbar = plt.colorbar(im)
    cbar.set_label('Attention weight (log scale)')

    os.makedirs(output_dir, exist_ok=True)
    fname = f"attention_L{layer+1}_H{head}_top{N}.png"
    path = os.path.join(output_dir, fname)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved heatmap: {path}")


def main():
    p = argparse.ArgumentParser(description="Plot top-K non-local attention heads, optionally limited to first N residues")
    p.add_argument("--model",       required=True, help="Checkpoint dir (e.g. .../e3)")
    p.add_argument("--test_file",   required=True, help="One-sequence-per-line preprocessed file")
    p.add_argument("--sequence",    type=int, default=0, help="Sequence index to analyze")
    p.add_argument("--global_top_k", type=int, default=2, help="Number of heads to plot globally if no layer is specified")
    p.add_argument("--layer",       type=int, help="1-indexed layer to restrict head search")
    p.add_argument("--window",      type=int, default=2, help="Ignore +/- window around diagonal")
    p.add_argument("--max_display", type=int, default=100, help="Max residues (rows/cols) to plot")
    p.add_argument("--output_dir",  default="evaluation_results/attention", help="Directory to save heatmaps")
    p.add_argument("--device",      default="auto", help="Device: auto|cuda|mps|cpu")
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    attns, seq_len = load_attention(args.model, args.test_file, args.sequence, device)

    layer_idx = None
    if args.layer is not None:
        if 1 <= args.layer <= attns.shape[0]:
            layer_idx = args.layer - 1
        else:
            raise ValueError(f"Layer {args.layer} out of range [1, {attns.shape[0]}]")

    top_heads = find_top_heads(attns, args.global_top_k, args.window, layer_idx)

    print("Top heads (energy, layer, head):")
    for e, l, h in top_heads:
        print(f"  Layer {l+1}, Head {h}, Energy {e:.1f}")

    for head_info in top_heads:
        plot_and_save(attns, seq_len, head_info, args.output_dir, args.max_display)

if __name__ == "__main__":
    main()
