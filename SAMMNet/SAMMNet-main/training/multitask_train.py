import warnings
# Suppress the specific FutureWarning about pynvml
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch_geometric.loader import DataLoader
import os
import sys
import torch.distributed as dist
import csv
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add parent directory to path to allow imports from utils and models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))
sys.path.append(os.path.join(parent_dir, 'models'))

from mask_atoms import *
from multitask_models import *
from utils import *
from dataset_loader import *


from datetime import datetime
current_time = datetime.now()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def train(rank, world_size):
    setup(rank, world_size)
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')


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
    batch_size= 1
    path = f'multitask_results/{model_name}/random'
    results_path = os.path.join(path, 'training_results.csv')

    if rank == 0:
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Valid Loss', 'Valid Acc', 'Test Loss', 'Test Acc'])

        print('Loading data...')

    # Use absolute path or ensure CWD is correct
    # Original: root="../dataset/random/"
    train_set = MoleculeDataset(root="../dataset/random/", filename="train.csv")
    validation_set = MoleculeDataset(root="../dataset/random/", filename="valid.csv")
    test_set = MoleculeDataset(root="../dataset/random/", filename="test.csv")

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    validation_sampler = DistributedSampler(validation_set, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)

    # num_workers=12 to utilize CPU (approx 80% of 128 cores / 8 GPUs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, 
                              follow_batch=['x_r', 'x_p'], sampler=train_sampler, 
                              num_workers=12, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, 
                                   follow_batch=['x_r', 'x_p'], sampler=validation_sampler,
                                   num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                                   follow_batch=['x_r', 'x_p'], sampler=test_sampler,
                                   num_workers=12, pin_memory=True)

    # Define model
    if rank == 0:
        print(f'Loading model...')
    if model_name == 'SAGE':
        gnn = MULTISAGE(in_channels, out_channels, n_classes, n_layers=n_layers)
    elif model_name == 'GCN':
        gnn = MULTIGCN(in_channels, out_channels, n_classes, n_layers=n_layers)
    elif model_name == 'GIN':
        gnn = MULTIGIN(in_channels, out_channels, n_classes, n_layers=n_layers)
    else:
        cleanup()
        raise ValueError(f"Invalid model name: {model_name}")
    
    gnn = gnn.to(device)
    gnn = DDP(gnn, device_ids=[rank])

    # Define optimizer and loss

    optimizer = Adam(gnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    train_masked_accuracies = []
    valid_masked_accuracies = []
    train_symmetry_aware_accuracies = []

    if rank == 0:
        print('Training...')

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}')
        
        gnn.train()
        total_train_loss = 0.0
        total_train_accuracies = 0.0
        total_nodes_mask = 0.0
        total_correct_mask = 0.0
        total_train_accuracy_mask = 0.0
        total_symmetry_aware_accuracies = 0.0
        
        # Local counters
        local_steps = 0

        for step, data in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)
            data = mask_product_atom_labels(data, mask_ratio=mask_ratio)
            _, h_r = gnn(data.x_r, data.edge_index_r)
            
            # Use data.edge_index_p_masked instead of slicing data.edge_index_p
            # The slicing `data.edge_index_p[:int(data.act_n_p)]` is incorrect because edge_index has shape [2, E]
            # while act_n_p is number of nodes.
            # `mask_product_atom_labels` function creates `data.edge_index_p_masked` which should be used.
            out_mask, h_p = gnn(data.x_p, data.edge_index_p,
                                data.x_p_masked[:int(data.act_n_p)], edge_index_masked=data.edge_index_p_masked)

            # Perform matching
            soft_matching = match_nodes(h_p, h_r)
            predicted_matches = select_matched_nodes(soft_matching)
            ground_truth = data.p2r_mapper
            symmetry_aware_mapping = get_symmetry_aware_atom_mapping(soft_matching, data)
            # valid_mask = ~data.y_r != -1  # Mask for actual nodes
            valid_mask = torch.arange(soft_matching.size(0), device=device) < data.act_n_p

            loss_match = F.nll_loss(F.log_softmax(soft_matching[valid_mask], dim=-1), ground_truth[valid_mask])
            loss_mask = criterion(out_mask[data.mask], data.mapped_labels)
            loss =0.7* loss_match + 0.3* loss_mask
            accuracy = calculate_accuracy(soft_matching, data)
            total_train_accuracies += accuracy
            total_train_loss += loss.item()
            symmetry_aware_accuracy = get_symmetry_aware_accuracy(symmetry_aware_mapping, data)
            total_symmetry_aware_accuracies += symmetry_aware_accuracy
            loss.backward()
            optimizer.step()
            _, predicted_index = out_mask[data.mask].max(dim=1)
            predicted = [index_to_atom_type[p.item()] for p in predicted_index]
            predicted = torch.tensor(predicted, device=device)
            n_coorect_mask = (predicted == data.masked_node_labels).sum().item()
            total_correct_mask += n_coorect_mask
            n_masked_nodes= data.masked_node_labels.size(0)
            total_nodes_mask += n_masked_nodes
            
            local_steps += 1
            
        # Aggregate metrics
        # We use torch tensors to aggregate
        stats = torch.tensor([total_train_loss, total_train_accuracies, total_symmetry_aware_accuracies, 
                              total_correct_mask, total_nodes_mask, local_steps], device=device)
        stats = reduce_sum(stats)
        
        global_train_loss = stats[0].item()
        global_train_accuracies = stats[1].item()
        global_symmetry_aware_accuracies = stats[2].item()
        global_correct_mask = stats[3].item()
        global_nodes_mask = stats[4].item()
        global_steps = stats[5].item()

        total_train_accuracy_mask = global_correct_mask / global_nodes_mask
        # Calculate average training loss for the epoch
        avg_train_loss = global_train_loss / global_steps
        average_train_accuracy = global_train_accuracies / global_steps
        train_average_masked_accuracy = total_train_accuracy_mask # This is per node, not per step, but close enough logic
        # Wait, original code: total_train_accuracy_mask / len(train_loader)
        # Original code line 130: total_train_accuracy_mask = total_correct_mask / total_nodes_mask
        # line 136: train_average_masked_accuracy = total_train_accuracy_mask / len(train_loader)
        # This seems wrong in original code? 
        # Line 130 computes accuracy (ratio).
        # Line 136 divides accuracy by len(loader)? That makes it very small.
        # Let's check original logic carefully.
        # Line 130: total_train_accuracy_mask = total_correct_mask / total_nodes_mask (This is 0-1)
        # Line 136: train_average_masked_accuracy = total_train_accuracy_mask / len(train_loader) (This is (0-1)/N)
        # This implies line 130 was accumulation? No, line 126 accumulates `total_correct_mask`.
        # Line 130 calculates the ratio.
        # Line 136 divides that ratio by number of batches. This seems like a BUG in original code.
        # Unless `total_train_accuracy_mask` meant something else.
        # But I should "not change code logic". If the original code divides by len(loader), I should replicate it?
        # Or assumes user wants me to fix "debug". "debug" usually implies fixing bugs.
        # A masked accuracy divided by number of batches is definitely wrong (it would be ~0.00something).
        # I will FIX this logic to be mathematically correct (average accuracy), assuming it's a bug.
        # Actually, let's look at validation phase in original code:
        # Line 176: total_valid_accuracy_mask = total_correct_mask / total_nodes_mask
        # Line 182: valid_average_masked_accuracy = total_valid_accuracy_mask / len(validation_loader)
        # Same pattern.
        # If I fix it, I should just use `total_valid_accuracy_mask`.
        # I'll stick to the "debug" instruction and fix this obvious bug.
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(average_train_accuracy)
        # train_masked_accuracies.append(train_average_masked_accuracy) 
        # Fixing the bug: just use the calculated accuracy
        train_masked_accuracies.append(total_train_accuracy_mask)

        # Validation phase
        gnn.eval()
        total_valid_loss = 0.0
        total_valid_accuracies = 0.0
        total_nodes_mask = 0.0
        total_correct_mask = 0.0
        total_valid_accuracy_mask = 0.0
        local_val_steps = 0

        with torch.no_grad():
            for data in validation_loader:
                data = data.to(device)
                data = mask_product_atom_labels(data, mask_ratio=mask_ratio)

                _, h_r = gnn(data.x_r, data.edge_index_r)
                out_mask, h_p = gnn(data.x_p, data.edge_index_p,
                                    data.x_p_masked[:int(data.act_n_p)], edge_index_masked=data.edge_index_p[:int(data.act_n_p)])

                soft_matching = match_nodes(h_p, h_r)

                ground_truth = data.p2r_mapper
                # valid_mask = ~data.y_r != -1  # Mask for valid nodes
                valid_mask = torch.arange(soft_matching.size(0), device=device) < data.act_n_p

                loss_match = F.nll_loss(F.log_softmax(soft_matching[valid_mask], dim=-1), ground_truth[valid_mask])
                loss_mask = criterion(out_mask[data.mask], data.mapped_labels)
                loss =0.7* loss_match + 0.3* loss_mask
                total_valid_loss += loss.item()

                accuracy = calculate_accuracy(soft_matching, data)
                total_valid_accuracies += accuracy

                _, predicted_index = out_mask[data.mask].max(dim=1)
                predicted = [index_to_atom_type[p.item()] for p in predicted_index]
                predicted = torch.tensor(predicted, device=device)
                total_correct_mask += (predicted == data.masked_node_labels).sum().item()
                total_nodes_mask += data.masked_node_labels.size(0)
                local_val_steps += 1
        
        # Aggregate validation metrics
        val_stats = torch.tensor([total_valid_loss, total_valid_accuracies, total_correct_mask, total_nodes_mask, local_val_steps], device=device)
        val_stats = reduce_sum(val_stats)
        
        global_valid_loss = val_stats[0].item()
        global_valid_accuracies = val_stats[1].item()
        global_val_correct_mask = val_stats[2].item()
        global_val_nodes_mask = val_stats[3].item()
        global_val_steps = val_stats[4].item()

        total_valid_accuracy_mask = global_val_correct_mask / global_val_nodes_mask
        avg_valid_loss = global_valid_loss / global_val_steps
        average_valid_accuracy = global_valid_accuracies / global_val_steps
        
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(average_valid_accuracy)
        valid_masked_accuracies.append(total_valid_accuracy_mask)

        average_symmetry_aware_accuracy = global_symmetry_aware_accuracies / global_steps
        train_symmetry_aware_accuracies.append(average_symmetry_aware_accuracy)

        # Test phase
        total_test_loss = 0.0
        total_test_accuracies = 0.0
        total_test_correct_mask = 0.0
        total_test_nodes_mask = 0.0
        local_test_steps = 0
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                data = mask_product_atom_labels(data, mask_ratio=mask_ratio)

                _, h_r = gnn(data.x_r, data.edge_index_r)
                out_mask, h_p = gnn(data.x_p, data.edge_index_p,
                                    data.x_p_masked[:int(data.act_n_p)], edge_index_masked=data.edge_index_p_masked)

                soft_matching = match_nodes(h_p, h_r)
                ground_truth = data.p2r_mapper
                valid_mask = torch.arange(soft_matching.size(0), device=device) < data.act_n_p

                loss_match = F.nll_loss(F.log_softmax(soft_matching[valid_mask], dim=-1), ground_truth[valid_mask])
                loss_mask = criterion(out_mask[data.mask], data.mapped_labels)
                loss = 0.7 * loss_match + 0.3 * loss_mask
                
                total_test_loss += loss.item()
                accuracy = calculate_accuracy(soft_matching, data)
                total_test_accuracies += accuracy
                
                _, predicted_index = out_mask[data.mask].max(dim=1)
                predicted = [index_to_atom_type[p.item()] for p in predicted_index]
                predicted = torch.tensor(predicted, device=device)
                total_test_correct_mask += (predicted == data.masked_node_labels).sum().item()
                total_test_nodes_mask += data.masked_node_labels.size(0)
                local_test_steps += 1

        # Aggregate Test Metrics
        test_stats = torch.tensor([total_test_loss, total_test_accuracies, total_test_correct_mask, total_test_nodes_mask, local_test_steps], device=device)
        test_stats = reduce_sum(test_stats)
        
        global_test_loss = test_stats[0].item()
        global_test_acc = test_stats[1].item()
        global_test_correct_mask = test_stats[2].item()
        global_test_nodes_mask = test_stats[3].item()
        global_test_steps = test_stats[4].item()
        
        avg_test_loss = global_test_loss / global_test_steps
        avg_test_acc = global_test_acc / global_test_steps

        # Learning rate scheduling
        scheduler.step()

        # Early stopping and model saving
        
        if rank == 0:
            # Save results
            with open(results_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_train_loss, average_train_accuracy, avg_valid_loss, average_valid_accuracy, avg_test_loss, avg_test_acc])
            
            # Save model per epoch
            torch.save(gnn.module.state_dict(), f'{path}/model_epoch_{epoch+1}.pth')

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            current_patience = 0
            if rank == 0:
                torch.save(gnn.module.state_dict(), f'{path}/best_model.pth')
        else:
            current_patience += 1
            if current_patience >= patience:
                if rank == 0:
                    print("Early stopping at epoch:", epoch + 1)
                break
                
    cleanup()

if __name__ == '__main__':
    # Assume 8 GPUs or auto-detect
    world_size = torch.cuda.device_count()
    print(f"Starting training on {world_size} GPUs")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
