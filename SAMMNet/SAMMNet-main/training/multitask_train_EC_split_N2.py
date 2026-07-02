import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch_geometric.loader import DataLoader
import os
import sys
import torch.distributed as dist
import csv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime

# Add parent directory to path to allow imports from utils and models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))
sys.path.append(os.path.join(parent_dir, 'models'))

from mask_atoms import *
from multitask_models import *
from utils import *
from dataset_loader import *

def main():
    # DDP Setup
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(0)
    np.random.seed(0)

    model_name = 'GCN'
    in_channels = 111  
    out_channels = 512  
    n_layers = 3
    gat_heads = 4
    lr = 0.0001
    step_size = 5
    gamma = 0.1
    num_epochs = 500
    best_valid_loss = float('inf')
    patience = 20
    current_patience = 0
    n_classes = 50
    mask_ratio = 0.15
    batch_size = 4
    
    path = f'multitask_results/{model_name}/EC_split_N2'

    if local_rank == 0:
        if not os.path.exists(path):
            os.makedirs(path)
        print('Loading data...')

    # Ensure directories exist before proceeding
    dist.barrier()

    train_set = MoleculeDataset(root="../dataset/EC_split_N2/", filename="train.csv")
    validation_set = MoleculeDataset(root="../dataset/EC_split_N2/", filename="valid.csv")
    test_set = MoleculeDataset(root="../dataset/EC_split_N2/", filename="test.csv")

    train_sampler = DistributedSampler(train_set, shuffle=True)
    validation_sampler = DistributedSampler(validation_set, shuffle=False)
    test_sampler = DistributedSampler(test_set, shuffle=False)

    # num_workers=12 per GPU * 8 GPUs = 96 workers (approx 80% of 128 cores)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, 
                              num_workers=12, pin_memory=True, follow_batch=['x_r', 'x_p'])
    validation_loader = DataLoader(validation_set, batch_size=batch_size, sampler=validation_sampler, 
                                   num_workers=12, pin_memory=True, follow_batch=['x_r', 'x_p'])
    test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, 
                                   num_workers=12, pin_memory=True, follow_batch=['x_r', 'x_p'])

    results_path = os.path.join(path, 'training_results.csv')
    if local_rank == 0:
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Valid Loss', 'Valid Acc', 'Test Loss', 'Test Acc'])

    if local_rank == 0:
        print(f'Loading model...')

    if model_name == 'SAGE':
        gnn = MULTISAGE(in_channels, out_channels, n_classes, n_layers=n_layers)
    elif model_name == 'GCN':
        gnn = MULTIGCN(in_channels, out_channels, n_classes, n_layers=n_layers)
    elif model_name == 'GIN':
        gnn = MULTIGIN(in_channels, out_channels, n_classes, n_layers=n_layers)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    gnn = gnn.to(device)
    gnn = DDP(gnn, device_ids=[local_rank])

    optimizer = Adam(gnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    if local_rank == 0:
        print('Training...')

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}')
            
        gnn.train()
        total_train_loss = 0.0
        total_train_accuracies = 0.0
        local_steps = 0
        
        for step, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            data = mask_product_atom_labels(data, mask_ratio=mask_ratio)
            
            _, h_r = gnn(data.x_r, data.edge_index_r)
            out_mask, h_p = gnn(data.x_p, data.edge_index_p,
                                data.x_p_masked, edge_index_masked=data.edge_index_p)

            soft_matching = match_nodes(h_p, h_r)
            # predicted_matches = select_matched_nodes(soft_matching)
            ground_truth = data.p2r_mapper
            # symmetry_aware_mapping = get_symmetry_aware_atom_mapping(soft_matching, data)
            
            valid_mask = torch.ones(soft_matching.size(0), dtype=torch.bool, device=device)

            loss_match = F.nll_loss(F.log_softmax(soft_matching[valid_mask], dim=-1), ground_truth[valid_mask])
            loss_mask = criterion(out_mask[data.mask], data.mapped_labels)
            loss = 0.7 * loss_match + 0.3 * loss_mask
            
            loss.backward()
            optimizer.step()
            
            accuracy = calculate_accuracy(soft_matching, data)
            total_train_accuracies += accuracy
            total_train_loss += loss.item()
            local_steps += 1

        # Aggregate Train Metrics
        train_stats = torch.tensor([total_train_loss, total_train_accuracies, local_steps], device=device)
        dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
        global_train_loss = train_stats[0].item()
        global_train_acc = train_stats[1].item()
        global_steps = train_stats[2].item()
        
        avg_train_loss = global_train_loss / global_steps
        avg_train_acc = global_train_acc / global_steps

        # Validation phase
        gnn.eval()
        total_valid_loss = 0.0
        total_valid_accuracies = 0.0
        local_val_steps = 0

        with torch.no_grad():
            for data in validation_loader:
                data = data.to(device)
                data = mask_product_atom_labels(data, mask_ratio=mask_ratio)

                _, h_r = gnn(data.x_r, data.edge_index_r)
                out_mask, h_p = gnn(data.x_p, data.edge_index_p,
                                    data.x_p_masked, edge_index_masked=data.edge_index_p)

                soft_matching = match_nodes(h_p, h_r)
                ground_truth = data.p2r_mapper
                valid_mask = torch.ones(soft_matching.size(0), dtype=torch.bool, device=device)

                loss_match = F.nll_loss(F.log_softmax(soft_matching[valid_mask], dim=-1), ground_truth[valid_mask])
                loss_mask = criterion(out_mask[data.mask], data.mapped_labels)
                loss = 0.7 * loss_match + 0.3 * loss_mask
                
                accuracy = calculate_accuracy(soft_matching, data)
                total_valid_accuracies += accuracy
                total_valid_loss += loss.item()
                local_val_steps += 1

        # Aggregate Valid Metrics
        valid_stats = torch.tensor([total_valid_loss, total_valid_accuracies, local_val_steps], device=device)
        dist.all_reduce(valid_stats, op=dist.ReduceOp.SUM)
        global_valid_loss = valid_stats[0].item()
        global_valid_acc = valid_stats[1].item()
        global_val_steps = valid_stats[2].item()
        
        avg_valid_loss = global_valid_loss / global_val_steps
        avg_valid_acc = global_valid_acc / global_val_steps

        # Test phase
        total_test_loss = 0.0
        total_test_accuracies = 0.0
        local_test_steps = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                data = mask_product_atom_labels(data, mask_ratio=mask_ratio)

                _, h_r = gnn(data.x_r, data.edge_index_r)
                out_mask, h_p = gnn(data.x_p, data.edge_index_p,
                                    data.x_p_masked, edge_index_masked=data.edge_index_p)

                soft_matching = match_nodes(h_p, h_r)
                ground_truth = data.p2r_mapper
                valid_mask = torch.ones(soft_matching.size(0), dtype=torch.bool, device=device)

                loss_match = F.nll_loss(F.log_softmax(soft_matching[valid_mask], dim=-1), ground_truth[valid_mask])
                loss_mask = criterion(out_mask[data.mask], data.mapped_labels)
                loss = 0.7 * loss_match + 0.3 * loss_mask
                
                accuracy = calculate_accuracy(soft_matching, data)
                total_test_accuracies += accuracy
                total_test_loss += loss.item()
                local_test_steps += 1

        # Aggregate Test Metrics
        test_stats = torch.tensor([total_test_loss, total_test_accuracies, local_test_steps], device=device)
        dist.all_reduce(test_stats, op=dist.ReduceOp.SUM)
        global_test_loss = test_stats[0].item()
        global_test_acc = test_stats[1].item()
        global_test_steps = test_stats[2].item()
        
        avg_test_loss = global_test_loss / global_test_steps
        avg_test_acc = global_test_acc / global_test_steps
        
        stop_training = torch.tensor(0, device=device)

        if local_rank == 0:
            print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Valid Loss {avg_valid_loss:.4f}, Test Loss {avg_test_loss:.4f}")
            
            # Save metrics
            with open(results_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, avg_test_loss, avg_test_acc])
                
            # Save Model
            torch.save(gnn.module.state_dict(), f'{path}/model_epoch_{epoch+1}.pth')

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                current_patience = 0
                torch.save(gnn.module.state_dict(), f'{path}/best_model.pth')
            else:
                current_patience += 1
                if current_patience >= patience:
                    print("Early stopping at epoch:", epoch + 1)
                    stop_training = torch.tensor(1, device=device)
        
        dist.broadcast(stop_training, src=0)
        if stop_training.item() == 1:
            break

        scheduler.step()

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
