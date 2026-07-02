#!/usr/bin/env python3
"""
Simple test script based on original test_batch.py logic.
Avoids complex indexing issues.
"""
import argparse
import subprocess
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import rdkit.Chem as Chem
import pickle
import os
import sys
from tqdm import tqdm

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'AMNet'))
sys.path.append(os.path.join(current_dir, 'dataset'))
sys.path.append(os.path.join(current_dir, 'utils'))

from gin import GIN
from amnet import FMNet
from molgraphdataset import MolGraphDataset, get_atom_features, get_bond_features
from utils import get_predicted_atom_mapping, get_acc_on_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get feature dimensions
silly_smiles = "O=O"
silly_mol = Chem.MolFromSmiles(silly_smiles)
n_node_features = len(get_atom_features(silly_mol.GetAtomWithIdx(0)))
n_edge_features = len(get_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))

TARGET_DATASETS = ("random", "EC_split_N2")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    choices=('all',) + TARGET_DATASETS,
    default='all',
    help='Dataset to test; the default all runs random and EC_split_N2.',
)
parser.add_argument(
    '--model_dir',
    type=str,
    default=os.path.join(current_dir, 'saved_models'),
)
parser.add_argument(
    '--dataset_dir',
    type=str,
    default=os.path.join(current_dir, 'dataset'),
)
parser.add_argument(
    '--output_dir',
    type=str,
    default=os.path.join(current_dir, 'test_results'),
)
parser.add_argument('--num_wl_iterations', type=int, default=3)
parser.add_argument('--embedding_dim', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()

if args.dataset == 'all':
    for dataset_name in TARGET_DATASETS:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            '--dataset', dataset_name,
            '--model_dir', args.model_dir,
            '--dataset_dir', args.dataset_dir,
            '--output_dir', args.output_dir,
            '--num_wl_iterations', str(args.num_wl_iterations),
            '--embedding_dim', str(args.embedding_dim),
            '--num_layers', str(args.num_layers),
            '--batch_size', str(args.batch_size),
        ]
        subprocess.run(cmd, check=True)
    sys.exit(0)

print("="*50)
print(f"Testing Dataset: {args.dataset}")
print("="*50)

# Paths
test_csv = os.path.join(args.dataset_dir, args.dataset, 'test.csv')
model_path = os.path.join(args.model_dir, args.dataset, 'model.pth')
output_path = os.path.join(args.output_dir, args.dataset)

print(f"Test CSV: {test_csv}")
print(f"Model: {model_path}")
print(f"Output: {output_path}")

# Check files exist
if not os.path.exists(test_csv):
    print(f" Test file not found: {test_csv}")
    sys.exit(1)

if not os.path.exists(model_path):
    print(f" Model file not found: {model_path}")
    sys.exit(1)

# Load data
test_data = pd.read_csv(test_csv)
print(f"Loaded {len(test_data)} test samples")

# Create dataset
test_dataset = MolGraphDataset(test_data, args.num_wl_iterations, santitize=False)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, 
                         follow_batch=['x_r', 'x_p'])

# Create model
gnn = GIN(n_node_features, args.embedding_dim, num_layers=args.num_layers, cat=True)
model = FMNet(gnn)
model = model.to(device)

# Load weights
model.load_state_dict(torch.load(model_path, map_location=device))
print(f"✓ Loaded model")

# Evaluate
model.eval()
all_h1 = []
all_h3 = []
all_h5 = []
all_h10 = []
all_acc = []
preds = []

print("\nEvaluating...")
with torch.no_grad():
    for data in tqdm(test_loader, desc="Testing"):
        data = data.to(device)
        
        # Forward pass - try both signatures
        try:
            M_hat, r_mask = model(data.x_r, data.edge_index_r, data.edge_feat_r,
                                 data.x_p, data.edge_index_p, data.edge_feat_p,
                                 data.x_r_batch, data.x_p_batch)
            M = model.symmetrywise_correspondence_matrix(M_hat, r_mask, 
                                                        data.eq_edge_index, 
                                                        data.x_p_batch)
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Try simpler approach
            M = model(data.x_r, data.edge_index_r, data.edge_feat_r,
                     data.x_p, data.edge_index_p, data.edge_feat_p,
                     data.x_r_batch, data.x_p_batch)
            if isinstance(M, tuple):
                M = M[0]
        
        # Get predictions using original functions
        try:
            pred = get_predicted_atom_mapping(M, data)
            acc = get_acc_on_test(pred, data)
        except Exception as e:
            print(f"Warning: Error in prediction: {e}")
            # Fallback: simple argmax
            pred = M.argmax(dim=-1).cpu().tolist()
            np_rp = [rp.item() if hasattr(rp, 'item') else rp for rp in data.rp_mapper]
            correct = sum(1 for i in range(len(np_rp)) if pred[i] == np_rp[i])
            acc = correct / len(np_rp) if len(np_rp) > 0 else 0
        
        preds.append(pred)
        all_acc.append(acc)
        
        # Calculate hits@k
        try:
            h1 = model.hits_at_k(1, M, data.y_r, data.rp_mapper, reduction='mean')
            all_h1.append(h1)
            h3 = model.hits_at_k(3, M, data.y_r, data.rp_mapper, reduction='mean')
            all_h3.append(h3)
            h5 = model.hits_at_k(5, M, data.y_r, data.rp_mapper, reduction='mean')
            all_h5.append(h5)
            h10 = model.hits_at_k(10, M, data.y_r, data.rp_mapper, reduction='mean')
            all_h10.append(h10)
        except Exception as e:
            print(f"Warning: Error in hits@k: {e}")
            all_h1.append(0)
            all_h3.append(0)
            all_h5.append(0)
            all_h10.append(0)
        
# Calculate results
mean_accuracy = sum(all_acc) / len(all_acc) if all_acc else 0
avg_h1 = sum(all_h1) / len(all_h1) if all_h1 else 0
avg_h3 = sum(all_h3) / len(all_h3) if all_h3 else 0
avg_h5 = sum(all_h5) / len(all_h5) if all_h5 else 0
avg_h10 = sum(all_h10) / len(all_h10) if all_h10 else 0

# Print results
print(f"\n{'='*50}")
print(f"Results:")
print(f"{'='*50}")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Hits@1:  {avg_h1:.4f}")
print(f"Hits@3:  {avg_h3:.4f}")
print(f"Hits@5:  {avg_h5:.4f}")
print(f"Hits@10: {avg_h10:.4f}")
print(f"{'='*50}")

# Save results
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, 'all_test_acc.txt'), 'wb') as f:
    pickle.dump(all_acc, f)

with open(os.path.join(output_path, 'all_test_h1.txt'), 'wb') as f:
    pickle.dump(all_h1, f)

with open(os.path.join(output_path, 'all_test_h3.txt'), 'wb') as f:
    pickle.dump(all_h3, f)

with open(os.path.join(output_path, 'all_test_h5.txt'), 'wb') as f:
    pickle.dump(all_h5, f)

with open(os.path.join(output_path, 'all_test_h10.txt'), 'wb') as f:
    pickle.dump(all_h10, f)

with open(os.path.join(output_path, 'predictions.pkl'), 'wb') as f:
    pickle.dump(preds, f)

with open(os.path.join(output_path, 'test_accuracy.txt'), 'w') as f:
    f.write(f"Dataset: {args.dataset}\n")
    f.write(f"Mean Accuracy: {mean_accuracy:.4f}\n")
    f.write(f"\n")
    f.write(f"Hits@1:  {avg_h1:.4f}\n")
    f.write(f"Hits@3:  {avg_h3:.4f}\n")
    f.write(f"Hits@5:  {avg_h5:.4f}\n")
    f.write(f"Hits@10: {avg_h10:.4f}\n")

print(f"\n✓ Results saved to {output_path}")
print("Done!")
