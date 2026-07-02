import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, GCNConv, SAGEConv


EPS = 1e-8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
This model is used for the pretrained GIN model.
the task is to predict the labels of the nodes in the molecular  graph.

"""

class MULTIGIN(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes,n_layers = 3, cat = True):
        super(MULTIGIN, self).__init__()

        self.in_channels = in_channels
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.cat = cat
        self.apply(xavier_init) 

        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(GINConv(nn.Sequential(nn.Linear(in_channels, out_channels),
                                           nn.ReLU(),
                                           nn.Linear(out_channels, out_channels))))
            in_channels = out_channels

        if self.cat:
            out_channels = self.in_channels + out_channels * n_layers

        self.lin = nn.Linear(out_channels, self.n_classes)

    def forward(self, x, edge_index, x_masked= None, edge_index_masked = None):
        xs = [x]

        for conv in self.gnn_layers:
            xs += [conv(xs[-1], edge_index)]

        h_unmasked = torch.cat(xs, dim=-1) if self.cat else xs[-1]

        if x_masked is not None and edge_index_masked is not None:
            xs_masked = [x_masked]
            for conv in self.gnn_layers:
                xs_masked += [conv(xs_masked[-1], edge_index_masked)]
            h_masked = torch.cat(xs_masked, dim=-1) if self.cat else xs_masked[-1]
            predicted_labels = self.lin(h_masked)
            return predicted_labels , h_unmasked

        else:
            predicted_labels = self.lin(h_unmasked)
            return predicted_labels , h_unmasked

class MULTIGCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, n_layers=3, cat=True):
        super(MULTIGCN, self).__init__()

        self.n_layers = n_layers
        self.n_classes = n_classes
        self.cat = cat
        self.in_channels = in_channels
        self.gnn_layers = torch.nn.ModuleList()

        # Define GNN layers
        for _ in range(n_layers):
            self.gnn_layers.append(GCNConv(in_channels, out_channels))
            in_channels = out_channels

        # Concatenate the layers for output
        if self.cat:
            out_channels = self.in_channels + out_channels * n_layers

        # Linear layer for node classification task
        self.lin = nn.Linear(out_channels, self.n_classes)

    def forward(self, x, edge_index, x_masked=None, edge_index_masked=None):
        xs = [x]

        for conv in self.gnn_layers:
            xs += [conv(xs[-1], edge_index)]

        h_unmasked = torch.cat(xs, dim=-1) if self.cat else xs[-1]

        if x_masked is not None and edge_index_masked is not None:
            xs_masked = [x_masked]
            for conv in self.gnn_layers:
                xs_masked += [conv(xs_masked[-1], edge_index_masked)]
            h_masked = torch.cat(xs_masked, dim=-1) if self.cat else xs_masked[-1]
            predicted_labels = self.lin(h_masked)
            return predicted_labels , h_unmasked

        else:
            predicted_labels = self.lin(h_unmasked)
            return predicted_labels , h_unmasked
       
class MULTISAGE(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, n_layers = 3, cat = True):
        super(MULTISAGE, self).__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.cat = cat

        self.gnn_layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.gnn_layers.append(SAGEConv(in_channels, out_channels))
            in_channels = out_channels
        
        if self.cat:
            out_channels = self.in_channels + out_channels * n_layers

        self.lin = nn.Linear(out_channels, self.n_classes)

    def forward(self, x, edge_index, x_masked=None, edge_index_masked=None):
        xs = [x]

        for conv in self.gnn_layers:
            xs += [conv(xs[-1], edge_index)]

        h_unmasked = torch.cat(xs, dim=-1) if self.cat else xs[-1]

        if x_masked is not None and edge_index_masked is not None:
            xs_masked = [x_masked]
            for conv in self.gnn_layers:
                xs_masked += [conv(xs_masked[-1], edge_index_masked)]
            h_masked = torch.cat(xs_masked, dim=-1) if self.cat else xs_masked[-1]
            predicted_labels = self.lin(h_masked)
            return predicted_labels , h_unmasked

        else:
            predicted_labels = self.lin(h_unmasked)
            return predicted_labels , h_unmasked



def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

if __name__ =='__main__':
    from torch_geometric.data import Data, Batch
    torch.manual_seed(42)

    model = MULTIGIN(32,50, 5)
    print(model)
    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    data = Data(x=x, edge_index=edge_index)
    x, e = data.x, data.edge_index
    predicted_labels, x = model( x,e)
    print(predicted_labels)
    
    print(100*'=')
    print(x)

