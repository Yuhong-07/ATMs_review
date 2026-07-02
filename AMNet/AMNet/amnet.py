import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj
import torch.nn as nn
import torch.nn.functional as F  
EPS = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FMNet(torch.nn.Module):
    '''
    Graph Matching Network that identifies atom similarity based on node embeddings.
    Adapted from https://github.com/rusty1s/deep-graph-matching-consensus/blob/master/dgmc/models/dgmc.py

    Args:
        gnn (torch.nn.Module): The underlying Graph Neural Network (GNN) used for node embedding.

    Methods:
        reset_parameters(): Reset the parameters of the GNN.
        forward(x_r, edge_index_r, edge_feat_r, x_p, edge_index_p, edge_feat_p, batch_size):
            Forward pass of the FMNet model to compute similarity scores between nodes.
        get_equivalent_atom_mask(eq_as): Compute a mask for equivalent atoms.
        adjust_correspondence_matrix_for_symmetry(M, eq_as): Mask similarity matrix M based on equivalent atoms.
        loss(M, y_r, y_p, reduction='mean'): Compute negative log-likelihood loss.
        acc(M, y_r, y_p, reduction='mean'): Compute accuracy or number of correctly mapped atoms.
        hits_at_k(k, M, y_r, y_p, reduction='mean'): Compute hits@k metric.
    '''
    def __init__(self, gnn, mlp_hidden_dim=512):
        super(FMNet, self).__init__()
        self.gnn = gnn
        self.mlp = nn.Sequential(
            nn.Linear(gnn.out_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        
    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.mlp.reset_parameters()



    def forward(self, x_r, edge_index_r, edge_feat_r,
                x_p, edge_index_p,edge_feat_p, batch_r, batch_p):
        '''
        Forward pass of the FMNet model to compute similarity scores between nodes.

        Args:
            x_r (torch.Tensor): Node features of the reference graph.
            edge_index_r (torch.Tensor): Edge indices of the reference graph.
            edge_feat_r (torch.Tensor): Edge features of the reference graph.
            x_p (torch.Tensor): Node features of the pattern graph.
            edge_index_p (torch.Tensor): Edge indices of the pattern graph.
            edge_feat_p (torch.Tensor): Edge features of the pattern graph.
            batch_r (torch.Tensor): Batch vector for reference graph nodes.
            batch_p (torch.Tensor): Batch vector for pattern graph nodes.

        Returns:
            torch.Tensor: Similarity scores between nodes.
        '''
        h_r = self.gnn(x_r, edge_index_r, edge_feat_r)
        h_p = self.gnn(x_p, edge_index_p, edge_feat_p)

        # Apply non-linear activation function to node embeddings
        #h_r = F.relu(self.gnn(x_r, edge_index_r, edge_feat_r))
        #h_p = F.relu(self.gnn(x_p, edge_index_p, edge_feat_p))

        h_r, r_mask = to_dense_batch(h_r, batch_r, fill_value=0) #binary masks (r_mask and p_mask) indicate the original lengths of the embeddings.
        h_p, p_mask = to_dense_batch(h_p, batch_p, fill_value=0)

        #M_hat = self.mlp(h_r) @ self.mlp(h_p).transpose(-1, -2)
        
        M_hat = h_r @ h_p.transpose(-1, -2) # shape batch_size, N_r, N_p

        #M = torch.softmax(M_hat,dim=-1)[r_mask]

        return M_hat, r_mask
    
    
    def symmetrywise_correspondence_matrix(self, M_hat, r_mask, eq_edge_index, batch_p):
        """
        Update the predicted correspondence matrix while considering molecule symmetry to avoid penalizing indistinguishable atoms.

        Args:
            M_hat: The dense similarity scores matrix (logits) (B, Nr_max, Np_max).
            r_mask: Mask for reactant nodes (B, Nr_max).
            eq_edge_index: Edge index of equivalent atoms in product graph.
            batch_p: Batch vector for product graph nodes.

        Returns:
            Updated similarity scores matrix (flattened) with adjustments for molecule symmetry.
        """
        
        M_dense = torch.softmax(M_hat, dim=-1) # (B, Nr_max, Np_max)
        
        # Build Symmetry Adjacency Matrix S
        # eq_edge_index connects equivalent atoms.
        # to_dense_adj returns (B, Np_max, Np_max)
        # Note: to_dense_adj returns shape (B, Np_max, Np_max) if edge_attr is None, but dimension 1 might be extra if edge_attr is present? 
        # No, to_dense_adj(edge_index, batch=batch) returns (B, N, N). 
        # Wait, let's check docs or assume standard PyG behavior. 
        # Actually to_dense_adj returns (B, N, N) if edge_attr is None.
        
        if eq_edge_index.size(1) == 0:
            # No symmetry info or empty graph? Should ideally be identity if self-loops added.
            # But if empty, S should be identity?
            # Our MolGraphDataset ensures self-loops are added, so it shouldn't be empty unless graph is empty.
            pass

        S = to_dense_adj(eq_edge_index, batch_p, max_num_nodes=M_dense.size(-1))
        S = S.to(dtype=M_dense.dtype)
        
        # S might have shape (B, N, N) or (B, E, N, N) if edge attributes? No.
        # But wait, to_dense_adj returns Tensor of shape (B, N, N) usually.
        
        # Multiply: M_corr = M_dense @ S
        M_corr = torch.bmm(M_dense, S) # (B, Nr_max, Np_max)
        
        # Apply mask to get flattened M
        M_flat = M_corr[r_mask] # (total_nodes_r, Np_max)
        
        # Normalize
        row_sums = M_flat.sum(dim=1, keepdim=True)
        M_final = M_flat / (row_sums + 1e-8)
        
        return M_final
        
        

    def loss(self, M, y_r, rp_mapper,  reduction='mean'):
        """
        Computes the negative log-likelihood loss for correspondence prediction.
        Args:
            M : Predicted correspondence matrix (similarity scores matrix) between nodes.
            y_r : Ground truth  reactant atom indices.
            rp_mapper : Ground truth product to reactant atom matcher.
            reduction: Specifies the reduction method for the computed loss.
                      Options include 'none', 'mean', and 'sum'.

        Returns:
            Negative log-likelihood loss value.
        """
        index_r = range(len(y_r))
        index_r = torch.tensor(index_r, device = device)
        val = M[index_r,rp_mapper]
        nll = -torch.log(val + EPS)
        return nll if reduction == 'none' else getattr(torch, reduction)(nll)
      
           
    def acc(self, M, y_r, rp_mapper, reduction='mean'):
        """
        Calculate the accuracy or the number of correctly mapped atoms in correspondence prediction.

        Args:
            M : Predicted correspondence matrix (similarity scores matrix) between nodes.
            y_r : Ground truth  reactant node indices.
            rp_mapper : Ground truth product to reactant atom matcher.
            reduction: Specifies the reduction method for computed accuracy.
                      Options include 'mean' (default), 'sum', or 'none'.
        Returns:
            Accuracy or the number of correctly mapped atoms.
        """
        index_r = range(len(y_r))
        index_r = torch.tensor(index_r, device = device)
        pred = M[index_r].argmax(dim=-1)
        correct = (pred == rp_mapper).sum().item()
        return correct / y_r.size(0) if reduction == 'mean' else correct
    
    def hits_at_k(self,k, M, y_r,rp_mapper, reduction='mean'):
        """
        Calculate hits@k metric for correspondence prediction.
        Args:
            k: The number of top-ranked correspondences to consider.
            M  : Predicted correspondence matrix (similarity scores matrix) between nodes.
            y_r  : Ground truth  reactant node indices.
            rp_mapper : Ground truth product to reactant atom matcher.
            reduction (str, optional): Specifies the reduction method for computed hits@k metric.
                Options include 'mean' (default), 'sum', or 'none'.

        Returns:
            Hits@k metric indicating the percentage of correctly mapped atoms in the top-k predictions.
        """
        index_r = range(len(y_r))
        index_r = torch.tensor(index_r, device = device)
        pred = M[index_r].argsort(dim=-1, descending=True)[:, :k]
        correct = (pred == rp_mapper.view(-1, 1)).sum().item()
        return correct / y_r.size(0) if reduction == 'mean' else correct


    def __repr__(self):
        return ('{}(\n''gnn={},\n').format(self.__class__.__name__, self.gnn)
    

if __name__ =='__main__':
    from torch_geometric.data import Data, Batch
    from gin import GIN
    torch.manual_seed(42)
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    data = Data(x=x, edge_index=edge_index)
    x, e, b = data.x, data.edge_index, data.batch
    gnn = GIN(data.num_node_features, 16, num_layers=2)
    model = FMNet(gnn)
    M_test = model(x, e, None, x, e, None, b)
    print(M_test)
    print( M_test.size() == (data.num_nodes, data.num_nodes))

    y_r = torch.tensor([0,1, 2, 3])
    y_p = torch.tensor([0,1, 2, 3])
    M = torch.tensor([[0.5,0,0.5,0], [0,1,0,0],[0.5,0,0.5,0],[0,0,0,1]])
    rp_mapper = torch.tensor([0,1, 2, 3])
    eq_as = [{0,2}, {1},{3}]
    l = model.loss(M, y_r,  rp_mapper)
    print(l)

