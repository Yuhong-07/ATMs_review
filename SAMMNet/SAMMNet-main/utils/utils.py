
import pickle
import torch 
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def write_data(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def get_model_parameters(model):
    """
    Computes the total number of parameters in the model that require gradients.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sinkhorn(similarity_matrix, n_iters=100, epsilon=1e-9):
    """
    Computes the Sinkhorn algorithm for normalizing a similarity matrix.

    Args:
        similarity_matrix (torch.Tensor): A 2D tensor representing the similarity matrix.
        n_iters (int, optional): The number of iterations for the Sinkhorn algorithm. Defaults to 100.
        epsilon (float, optional): A small value added to the diagonal of the similarity matrix for numerical stability. Defaults to 1e-9.

    Returns:
        torch.Tensor: A 2D tensor representing the normalized similarity matrix.

    The Sinkhorn algorithm is a method for normalizing a similarity matrix. It is used in various applications such as image matching, OCR, and natural language processing. The algorithm iteratively normalizes the similarity matrix by scaling the rows and columns to ensure that they sum up to 1.

    The function takes a similarity matrix as input and performs the Sinkhorn algorithm for a specified number of iterations. The similarity matrix is first initialized with a small value added to the diagonal to prevent division by zero. Then, the algorithm iteratively updates the logarithm of the scaling factors using the log-sum-exp trick. Finally, the scaling factors are exponentiated to obtain the normalized similarity matrix.

    Note:
        The function assumes that the similarity matrix is a 2D tensor and uses the PyTorch library for computations.

    Example:
        >>> similarity_matrix = torch.tensor([[0.5, 0.3], [0.2, 0.8]])
        >>> sinkhorn(similarity_matrix)
        tensor([[0.5000, 0.2500],
                [0.2500, 0.7500]])
    """
    log_alpha = similarity_matrix + epsilon
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)
    
def match_nodes(h_p, h_r):
    """
    Calculate the soft matching between two sets of nodes based on their representations.

    Args:
        h_p (torch.Tensor): The representation of reactant molecular graph.
        h_r (torch.Tensor): The representation of product molecular graph.

    Returns:
        torch.Tensor: Soft matching scores between the nodes of the two sets.
    """
    similarity_matrix = torch.matmul(h_p, h_r.T)
    soft_matching = sinkhorn(similarity_matrix)
    return soft_matching

def select_matched_nodes(soft_matching):
    """
    Selects the matched nodes based on the soft matching scores.

    Args:
        soft_matching (torch.Tensor): The soft matching scores between the nodes.

    Returns:
        torch.Tensor: The indices of the matched nodes.
    """
    predicted_matches = torch.argmax(soft_matching, dim=-1)
    return predicted_matches

def calculate_accuracy(soft_matching, data):
    """
    Calculate the accuracy of the matching between two sets of nodes based on their representations.

    Args:
        soft_matching (torch.Tensor): The soft matching scores between the nodes of the two sets.
        labels (torch.Tensor): The labels of the nodes.
        valid_mask (torch.Tensor): The mask indicating which nodes are valid (to have same size graph we paded smaller graph).

    Returns:
        float: The accuracy of the matching.
    """
    predicted_matches = select_matched_nodes(soft_matching)
    ground_truth = data.p2r_mapper

    # Use valid_mask if available (handles batching correctly)
    if hasattr(data, 'valid_mask'):
        valid_mask = data.valid_mask
        predicted_matches = predicted_matches[valid_mask]
        ground_truth = ground_truth[valid_mask]
    else:
        # Fallback for single graph or if act_n_p is scalar
        if data.act_n_p.numel() == 1:
            limit = int(data.act_n_p)
            predicted_matches = predicted_matches[:limit]
            ground_truth = ground_truth[:limit]
        else:
            # Fallback using sum if it's a batch but no valid_mask (risky if padding is interleaved)
            # This is a best-effort fallback
            limit = int(data.act_n_p.sum())
            predicted_matches = predicted_matches[:limit]
            ground_truth = ground_truth[:limit]

    correct_matches = (predicted_matches == ground_truth).sum().item()
    total_matches = ground_truth.size(0)

    accuracy = correct_matches / total_matches
    return accuracy

def flatten_list(nested_list):
    """
    Flattens a nested list into a single list.

    Args:
        nested_list (list): The list to flatten.

    Returns:
        list: The flattened list.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def convert_to_tuple(element):
    """
    Recursively convert all sets in the given element to tuples.

    Args:
        element (set, tuple, list): The element to convert.

    Returns:
        tuple: A tuple with all nested sets converted to tuples.
    """
    if isinstance(element, set):
        return tuple(convert_to_tuple(item) for item in element)
    elif isinstance(element, (list, tuple)):
        return type(element)(convert_to_tuple(item) for item in element)
    else:
        return element

def get_atom_to_set(data):
    """
    Generates a dictionary mapping atoms to their equivalent atom sets.

    Args:
        data (object): The data object containing information about equivalent atom sets.

    Returns:
        dict: A dictionary mapping atoms to their equivalent atom sets.
    """
    flat_eq_as = flatten_list(data.eq_as)
    return {atom: convert_to_tuple(atom_set) for atom_set in flat_eq_as for atom in convert_to_tuple(atom_set)}


def get_symmetry_aware_atom_mapping(soft_matching, data):
    """
    Generates a symmetry-aware atom mapping based on the soft matching scores and the data.

    Args:
        soft_matching (torch.Tensor): The soft matching scores between the nodes of the two sets.
        data (Data): The data containing information about the atoms and their equivalence classes.

    Returns:
        list: A list of predicted atom mappings after considering symmetry based on equivalent atom sets.

    e.g.
        eq_as = [{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9, 11}, {10}, {12}, {13}, {14}, {15}, {16}]
        initail_prdes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 11, 12, 13, 14, 15, 16]
        symetry_aware_preds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


    """
    pred_tensor = select_matched_nodes(soft_matching)
    
    # Use valid_mask if available (handles batching correctly)
    if hasattr(data, 'valid_mask'):
        pred_tensor = pred_tensor[data.valid_mask]
    else:
        # Fallback for single graph or if act_n_p is scalar
        if data.act_n_p.numel() == 1:
            limit = int(data.act_n_p)
            pred_tensor = pred_tensor[:limit]
        else:
            limit = int(data.act_n_p.sum())
            pred_tensor = pred_tensor[:limit]
            
    pred = pred_tensor.tolist()
    atom_to_set = get_atom_to_set(data)

    assigned_atoms = {}

    for i, pr in enumerate(pred):
        if pr in atom_to_set:
            atom_set = sorted(atom_to_set[pr])  # Sort to maintain order
            set_id = tuple(atom_set)
            if set_id not in assigned_atoms:
                assigned_atoms[set_id] = set()               
            available_atoms = [a for a in atom_set if a not in assigned_atoms[set_id]]
            if available_atoms:
                pred[i] = available_atoms[0]  # Assign unique atom
                assigned_atoms[set_id].add(available_atoms[0])

    return torch.tensor(pred, device=soft_matching.device)

def get_symmetry_aware_accuracy(symmetry_aware_mapping, data):
    ground_truth = data.p2r_mapper
    
    # Use valid_mask if available (handles batching correctly)
    if hasattr(data, 'valid_mask'):
        valid_mask = data.valid_mask
        ground_truth = ground_truth[valid_mask]
        # n = valid_mask.sum().item() # Not needed if we use tensor size
    else:
        # Fallback for single graph or if act_n_p is scalar
        if data.act_n_p.numel() == 1:
            limit = int(data.act_n_p)
            ground_truth = ground_truth[:limit]
        else:
            limit = int(data.act_n_p.sum())
            ground_truth = ground_truth[:limit]
            
    total_matches = ground_truth.size(0)
    symmetry_aware_mapping_accuracy = torch.sum((symmetry_aware_mapping == ground_truth)).item() / total_matches

    return symmetry_aware_mapping_accuracy


def sample_std(data):
    if len(data) < 2:
        raise ValueError("Sample size must be greater than 1 for sample standard deviation.")

    # Step 1: Calculate the mean
    mean = sum(data) / len(data)
    
    # Step 2: Compute the sum of squared differences from the mean
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    
    # Step 3: Calculate the sample standard deviation
    std_dev = math.sqrt(squared_diff_sum / (len(data) - 1))
    
    return std_dev

if __name__ == "__main__":
    torch.manual_seed(42)
    from torch_geometric.data import Data
    from multitask_models import *
    import numpy as np

    num_nodes = 5
    in_channels = 3  # Dimension of node features
    out_channels = 64 # Dimension of node embeddings
    gnn = MULTISAGE(in_channels, out_channels, n_layers=5)

    x1 = torch.randn(num_nodes, in_channels)  # Random node features
    x2 = torch.randn(num_nodes, in_channels)  # Random node features
    edge_index1 = torch.tensor([[0, 1, 1, 2, 1], [1, 0, 2, 1, 2]], dtype=torch.long)  # Edges
    edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # Edges
    edge_features1 = torch.randn(edge_index1.size(1), 1)  # Random edge features
    edge_features2 = torch.randn(edge_index2.size(1), 1)  # Random edge features


    data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_features1)
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_features2)
    h_p = gnn(data1.x, data1.edge_index)
    h_r = gnn(data2.x, data2.edge_index)

    soft_matching = match_nodes(h_p, h_r)
    print(soft_matching)
    matched_nodes = select_matched_nodes(soft_matching)
    print(matched_nodes)

    from dataset_loader import MoleculeDataset

    train_dataset = MoleculeDataset(root="../data/", filename="train.csv")
    data =train_dataset[1]
    atom_to_set =get_atom_to_set(data)
    print("Atom to set",atom_to_set)
    print('data.eq_as',data.eq_as)
    pred = [0,1,2,3,4,5,6,7,8,11,10,11,12,13,14,15,16]
    
    #mapping = get_symmetry_aware_atom_mapping(soft_matching, data)
    #print(mapping)

