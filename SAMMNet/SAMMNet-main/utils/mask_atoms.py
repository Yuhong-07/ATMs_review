import torch
from torch_geometric.loader import DataLoader
try:
    from dataset_loader import *
except ImportError:
    from utils.dataset_loader import *
import numpy as np
import torch.nn as nn
get_unique = False

def get_unique_atom_types(dataset):
    unique_atom_types = set()

    for molecule in dataset:
        atom_types_r = np.unique(molecule.z_r)
        atom_types_p = np.unique(molecule.z_p)
        atom_types = np.unique(np.concatenate([atom_types_r, atom_types_p]))
        unique_atom_types.update(atom_types)

    return list(unique_atom_types)
if get_unique == True:

    train_dataset = MoleculeDataset(root="../data/", filename="train.csv")
    test_dataset = MoleculeDataset(root="../data/", filename="test.csv", test=True)
    valid_dataset = MoleculeDataset(root="../data/", filename="valid.csv", valid=True)

    train_unique = get_unique_atom_types(train_dataset)
    test_unique = get_unique_atom_types(test_dataset)
    valid_unique =get_unique_atom_types(valid_dataset)

    all_atoms = np.concatenate([train_unique, test_unique, valid_unique])
    unique_atom_types = np.unique(all_atoms)
    # save unique_atom_types
    np.save('unique_atom_types.npy', unique_atom_types)

    #read unique_atom_types
unique_atom_types = np.load('unique_atom_types.npy')


atom_type_to_index = {int(atom_type): index for index, atom_type in enumerate(unique_atom_types)}

index_to_atom_type = {index: int(atom_type) for index, atom_type in enumerate(unique_atom_types)}




def mask_reactant_atom_labels(data, mask_ratio=0.2, mask_value=-1):
    num_nodes = data.act_n_r.item()


    num_mask = max(1, int(num_nodes * mask_ratio))
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask_indices = torch.randperm(num_nodes)[:num_mask]
    mask[mask_indices] = True
    
    data.mask = mask

    data.masked_node_labels = data.z_r[mask].clone()
    data.masked_node_features = data.x_r[mask].clone()

    data.z_r[mask] = mask_value
    data.x_r[mask] = mask_value
    mapped_labels = [atom_type_to_index[z.item()] for z in data.masked_node_labels]
    mapped_labels = torch.tensor(mapped_labels, dtype=torch.long)
    data.mapped_labels = mapped_labels
    return data

def mask_product_atom_labels_origin(data, mask_ratio=0.15, mask_value=-1):
    num_nodes = data.act_n_p.item()
    num_mask = max(1, int(num_nodes * mask_ratio))

    data.x_p = data.x_p[:data.act_n_p]
    data.edge_index_p = data.edge_index_p[:data.act_n_p]
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask_indices = torch.randperm(num_nodes)[:num_mask]
    mask[mask_indices] = True
    data.mask_indices = mask_indices
    data.mask = mask
    data.masked_node_labels = data.z_p[mask].clone()
    data.masked_node_features = data.x_p[mask].clone()
    data.x_p_mask = data.x_p
    data.z_p[mask] = mask_value
    data.x_p_mask[mask] = mask_value
    data.edge_index_masked = data.edge_index_p
    mapped_labels = [atom_type_to_index[z.item()] for z in data.masked_node_labels]
    mapped_labels = torch.tensor(mapped_labels, dtype=torch.long)
    data.mapped_labels = mapped_labels
    if data.x_r.shape[0] > data.x_p.shape[0]:
        diff = data.x_r.shape[0] - data.x_p.shape[0]
        data.x_p = nn.functional.pad(data.x_p, (0, 0, 0, diff), value = 0)
    assert not torch.isnan(data.x_p).any(), "NaNs found in product node features after masking"
    assert not torch.isnan(data.z_p).any(), "NaNs found in product node labels after masking"
    return data
   
def mask_product_atom_labels_0(data, mask_ratio=0.15, mask_value=-1):
    num_nodes = data.act_n_p.item()
    num_mask = max(1, int(num_nodes * mask_ratio))

    data.x_p_masked = data.x_p[:data.act_n_p]
    data.edge_index_p_masked = data.edge_index_p[:data.act_n_p]
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask_indices = torch.randperm(num_nodes)[:num_mask]
    mask[mask_indices] = True
    data.mask_indices = mask_indices
    data.mask = mask
    data.masked_node_labels = data.z_p[mask].clone()
    data.masked_node_features = data.x_p_masked[mask].clone()
    data.z_p[mask] = mask_value
    data.x_p_masked[mask] = mask_value

    mapped_labels = [atom_type_to_index[z.item()] for z in data.masked_node_labels]
    mapped_labels = torch.tensor(mapped_labels, dtype=torch.long)
    data.mapped_labels = mapped_labels
    if data.x_r.shape[0] > data.x_p.shape[0]:
        diff = data.x_r.shape[0] - data.x_p.shape[0]
        data.x_p = nn.functional.pad(data.x_p, (0, 0, 0, diff), value = 0)
    
    return data

def mask_product_atom_labels(data, mask_ratio=0.15, mask_value=-1):
    device = data.x_p.device
    
    # Clone the tensors to ensure original data remains unchanged
    data.x_p_masked = data.x_p.clone()
    data.edge_index_p_masked = data.edge_index_p.clone()
    
    num_nodes_total = data.x_p.size(0)
    mask = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
    
    # Check if batched (using follow_batch=['x_p'])
    if hasattr(data, 'x_p_batch'):
        batch_vector = data.x_p_batch
        num_graphs = batch_vector.max().item() + 1
        
        for i in range(num_graphs):
            # Indices of nodes belonging to graph i
            indices = (batch_vector == i).nonzero(as_tuple=True)[0]
            num_nodes_i = indices.size(0)
            num_mask_i = max(1, int(num_nodes_i * mask_ratio))
            
            # Select random indices within this graph
            perm = torch.randperm(num_nodes_i, device=device)[:num_mask_i]
            mask_indices_i = indices[perm]
            mask[mask_indices_i] = True
    else:
        # Single graph or batched without batch vector (treat as one big graph)
        # But for correctness with 'act_n_p' logic, we assume standard behavior
        num_mask = max(1, int(num_nodes_total * mask_ratio))
        mask_indices = torch.randperm(num_nodes_total, device=device)[:num_mask]
        mask[mask_indices] = True

    data.mask = mask  

    # Save original labels and features for the masked nodes
    data.masked_node_labels = data.z_p[mask].clone()
    data.masked_node_features = data.x_p_masked[mask].clone()

    # Apply the mask value
    data.z_p[mask] = mask_value  
    data.x_p_masked[mask] = mask_value  

    # Map the masked labels to atom types
    mapped_labels = [atom_type_to_index[z.item()] for z in data.masked_node_labels]
    mapped_labels = torch.tensor(mapped_labels, dtype=torch.long, device=device)
    data.mapped_labels = mapped_labels  

    return data



