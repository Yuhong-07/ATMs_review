#!/usr/bin/env python3
"""Evaluate SAMMNet mean accuracy and Top-K accuracy."""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, "utils"))
sys.path.append(os.path.join(PROJECT_DIR, "models"))

from dataset_loader import MoleculeDataset
from multitask_models import MULTIGCN, MULTIGIN, MULTISAGE
from utils import match_nodes


DATASET_CONFIGS = {
    "random": {
        "dataset_dir": os.path.join(PROJECT_DIR, "dataset", "random"),
        "checkpoint": os.path.join(
            CURRENT_DIR, "multitask_results", "GCN", "random", "best_model.pth"
        ),
    },
    "EC_split_N2": {
        "dataset_dir": os.path.join(PROJECT_DIR, "dataset", "EC_split_N2"),
        "checkpoint": os.path.join(
            CURRENT_DIR, "multitask_results", "GCN", "EC_split_N2", "best_model.pth"
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAMMNet mean accuracy and Top-K accuracy."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIGS),
        default=["random", "EC_split_N2"],
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        default=["1", "3", "5", "10"],
        help="Positive Top-K values, for example: --top-k 1 3 5 10.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(CURRENT_DIR, "multitask_topk_metrics"),
    )
    parser.add_argument("--model-name", choices=["GCN", "GIN", "SAGE"], default="GCN")
    parser.add_argument("--in-channels", type=int, default=111)
    parser.add_argument("--out-channels", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-classes", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_top_k(values):
    top_k_values = {1}
    for value in values:
        for part in str(value).split(","):
            if part.strip():
                k = int(part)
                if k < 1:
                    raise ValueError("Top-K values must be positive integers.")
                top_k_values.add(k)
    return sorted(top_k_values)


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def build_model(args):
    model_classes = {"GCN": MULTIGCN, "GIN": MULTIGIN, "SAGE": MULTISAGE}
    return model_classes[args.model_name](
        args.in_channels,
        args.out_channels,
        args.n_classes,
        n_layers=args.n_layers,
    )


def load_model(args, checkpoint_path, device):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }

    model = build_model(args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_dataset(args, dataset_name, top_k_values, device):
    config = DATASET_CONFIGS[dataset_name]
    checkpoint_path = config["checkpoint"]
    dataset_dir = config["dataset_dir"]
    test_path = os.path.join(dataset_dir, "test.csv")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")

    print(f"\n[{dataset_name}] loading checkpoint: {checkpoint_path}", flush=True)
    model = load_model(args, checkpoint_path, device)
    dataset = MoleculeDataset(root=dataset_dir, filename="test.csv")

    # Evaluate one reaction at a time so product-to-reactant labels stay local.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    correct_totals = {k: 0 for k in top_k_values}
    all_hits = {k: [] for k in top_k_values}
    atom_total = 0
    samples = 0
    skipped = 0

    print(f"[{dataset_name}] evaluating {len(dataset)} test reactions", flush=True)
    with torch.no_grad():
        for _, data in enumerate(loader):
            if args.limit and samples >= args.limit:
                break

            data = data.to(device)
            _, h_r = model(data.x_r, data.edge_index_r)
            _, h_p = model(data.x_p, data.edge_index_p)
            soft_matching = match_nodes(h_p, h_r)

            ground_truth = data.p2r_mapper
            valid_mask = ground_truth >= 0
            valid_mask &= ground_truth < soft_matching.size(1)
            if not valid_mask.any():
                skipped += 1
                continue

            scores = soft_matching[valid_mask]
            labels = ground_truth[valid_mask]
            total_atoms = int(labels.numel())

            for k in top_k_values:
                effective_k = min(k, scores.size(1))
                candidates = torch.topk(scores, effective_k, dim=1).indices
                correct = int(
                    (candidates == labels.unsqueeze(1)).any(dim=1).sum().item()
                )
                correct_totals[k] += correct
                all_hits[k].append(correct / total_atoms)

            atom_total += total_atoms
            samples += 1

            if samples % 100 == 0:
                print(f"[{dataset_name}] processed {samples} reactions", flush=True)

    if skipped:
        print(f"[{dataset_name}] skipped {skipped} reactions", flush=True)

    summary = {
        "dataset": dataset_name,
        "mean_accuracy": safe_mean(all_hits[1]) * 100,
    }
    for k in top_k_values:
        summary[f"Hits@{k}"] = (
            correct_totals[k] / atom_total * 100 if atom_total else 0.0
        )
    return summary


def write_summary(output_dir, summaries, top_k_values):
    summary_path = os.path.join(output_dir, "summary.csv")
    fields = ["dataset", "mean_accuracy"]
    fields += [f"Hits@{k}" for k in top_k_values]

    with open(summary_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    return summary_path


def main():
    args = parse_args()
    top_k_values = parse_top_k(args.top_k)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"Using device: {device}", flush=True)
    print(f"Top-K values: {top_k_values}", flush=True)
    summaries = [
        evaluate_dataset(args, dataset_name, top_k_values, device)
        for dataset_name in args.datasets
    ]
    summary_path = write_summary(args.output_dir, summaries, top_k_values)
    print(f"\nSummary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
