#!/usr/bin/env python3
import os
import argparse
import math
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from models.progen.modeling_progen import ProGenForCausalLM
from torch.nn.utils.rnn import pad_sequence

class SequenceDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        self.lines = open(path).read().splitlines()
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        enc = self.tok.encode(self.lines[idx])
        ids = enc.ids[: self.max_length]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save perplexity for a ProGen2 checkpoint"
    )
    parser.add_argument("--model",      required=True, help="Path to checkpoint directory (e.g. .../e3)")
    parser.add_argument("--test_file",  required=True, help="Test file (.txt) with one sequence per line")
    parser.add_argument("--device",     default="cuda", help="Device: auto, cuda, mps, or cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch",      required=True, help="Epoch identifier for naming outputs")
    parser.add_argument("--tag",        default="", help="Optional run tag (e.g. single, multi)")
    parser.add_argument("--output_dir", default="evaluation_results/perplexity",
                        help="Root dir to save results")
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

    # Load model
    model = ProGenForCausalLM.from_pretrained(args.model).to(device).eval()

    # Load tokenizer
    tok_path = os.path.join(args.model, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    tokenizer.enable_padding(direction="right", pad_id=0, pad_token="<|pad|>", length=1024)

    # Prepare data loader
    ds = SequenceDataset(args.test_file, tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, collate_fn=collate_fn)

    # Compute total loss and token count
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            out = model(batch, labels=batch)
            loss = out.loss
            n_toks = (batch != 0).sum().item()
            total_loss  += loss.item() * n_toks
            total_tokens += n_toks

    # Calculate metrics
    avg_loss   = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Print results
    print(f"Avg cross‑entropy: {avg_loss:.4f}")
    print(f"Perplexity:       {perplexity:.2f}")

    # Organize outputs
    model_dir = args.model.rstrip(os.sep)
    model_name = os.path.basename(os.path.dirname(model_dir))
    run_name = f"epoch_{args.epoch}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.output_dir, model_name, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save to CSV
    csv_file = os.path.join(out_dir, "perplexity.csv")
    with open(csv_file, "w") as f:
        f.write("metric,value\n")
        f.write(f"avg_cross_entropy,{avg_loss:.4f}\n")
        f.write(f"perplexity,{perplexity:.4f}\n")

    print(f"Saved results to: {csv_file}")

if __name__ == "__main__":
    main()
