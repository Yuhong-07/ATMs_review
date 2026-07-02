import argparse
import sys
import os
import torch
import multiprocessing

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(root_dir, 'AMNet'))
sys.path.append(os.path.join(root_dir, 'dataset'))
sys.path.append(os.path.join(root_dir, 'utils'))

import torch
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from gin import GIN
import pandas as pd
from amnet import FMNet
import rdkit.Chem as Chem
from molgraphdataset import *
import pickle
import os
from tqdm import tqdm

def main():
    # Set start method to spawn to avoid RDKit deadlock issues
    # Note: set_start_method should only be called once, so we wrap it in try-except
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Set random seeds for reproducibility
    import random
    import numpy as np
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU
    
    # Make PyTorch deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {SEED} for reproducibility")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    silly_smiles = "O=O"
    silly_mol = Chem.MolFromSmiles(silly_smiles)
    n_node_features = len(get_atom_features(silly_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_wl_iterations', type = int, default = 3)
    parser.add_argument('--node_features_dim', type = int, default = n_node_features)
    parser.add_argument('--edge_feature_dim', type = int, default = n_edge_features)
    parser.add_argument('--santitize', type = bool, default = False)
    parser.add_argument('--embedding_dim', type = int, default=512)
    parser.add_argument('--num_layers', type = int, default = 3)
    parser.add_argument('--lr', type=float, default = 0.0001)
    ##修改epoch为2
    parser.add_argument('--n_epochs', type = int, default = 2)
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--dataset_path', type=str, default=r'd:/DL_code_paper/AMNet/Atom-matching-network-main/train/MoleculeDataset')
    parser.add_argument('--output_path', type=str, default='experiment1/05_10_edege')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_jobs', type=int, default=None, help='Number of jobs for dataset preprocessing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    
    # Update seed if provided via command line
    if args.seed != 42:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed updated to {args.seed}")

    # Read CSVs normally (letting pandas infer headers) to handle multi-column files correctly
    train_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
    valid_data = pd.read_csv(os.path.join(args.dataset_path, 'valid.csv'))

    # Ensure we have the right columns
    if 'reactions' not in train_data.columns:
        # Fallback to assuming first column if header is missing or different
        train_data.rename(columns={train_data.columns[0]: 'reactions'}, inplace=True)
    if 'edits' not in train_data.columns and 'edits' in train_data.columns:
         pass # Assume handled or use index
         
    train_dataset = MolGraphDataset(train_data , args.num_wl_iterations, santitize=args.santitize, n_jobs=args.n_jobs)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False,follow_batch=['x_r', 'x_p'], 
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0), prefetch_factor=2 if args.num_workers > 0 else None)
    
    validation_dataset = MolGraphDataset(valid_data, args.num_wl_iterations, santitize=args.santitize, n_jobs=args.n_jobs)
    validation_loader = DataLoader(validation_dataset, args.batch_size, shuffle=False,follow_batch=['x_r', 'x_p'], 
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0), prefetch_factor=2 if args.num_workers > 0 else None)

    gnn = GIN(args.node_features_dim, args.embedding_dim, num_layers=args.num_layers, cat=True )
    model = FMNet(gnn)

    print(device)
    print(model)
    print(args)


    gnn =  gnn.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.999))
    scaler = GradScaler()

    best_valid_loss = float('inf')
    patience = 20  # Number of epochs to wait for improvement
    counter = 0  # Counter to keep track of epochs without improvement

    def train(train_loader):
        model.train()
        total_loss = total_nodes = total_correct = 0
        
        # Add tqdm progress bar
        print(f"Starting training loop with {len(train_loader)} batches...")
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False, mininterval=5.0)
        
        for i, data in pbar:   
            optimizer.zero_grad()
            data = data.to(device)
            #print(i)
            with autocast():
                M_hat, r_mask = model( data.x_r,data.edge_index_r,data.edge_feat_r,
                                    data.x_p, data.edge_index_p,data.edge_feat_p,
                                    data.x_r_batch, data.x_p_batch) 
            
                
                M = model.symmetrywise_correspondence_matrix(M_hat, r_mask, data.eq_edge_index, data.x_p_batch)
            loss = model.loss(M, data.y_r,  data.rp_mapper)

            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            
            # Calculate accuracy for progress bar (optional, can be expensive so maybe skip or do simply)
            # acc = model.acc(M , data.y_r, data.rp_mapper, reduction='sum')
            # total_correct += acc
            # total_nodes += data.y_r.size(0)
            
            # Update progress bar description
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

            # To keep original logic for return values
            total_correct += model.acc(M , data.y_r, data.rp_mapper, reduction='sum')
            total_nodes += data.y_r.size(0)
         
            
        return total_loss/len(train_loader), total_correct /total_nodes

    def validation_loss(loader):
        model.eval()
        total_loss = total_nodes = total_correct = 0
        
        # Add tqdm progress bar
        pbar = tqdm(loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for data in pbar:
                data = data.to(device)
                
                with autocast():
                    M_hat, r_mask = model( data.x_r,data.edge_index_r,data.edge_feat_r,
                                    data.x_p, data.edge_index_p,data.edge_feat_p,
                                    data.x_r_batch, data.x_p_batch) 
            
                
                M = model.symmetrywise_correspondence_matrix(M_hat, r_mask, data.eq_edge_index, data.x_p_batch)
                loss = model.loss(M, data.y_r,  data.rp_mapper)
                
                total_loss += loss.item()

                total_correct += model.acc(M, data.y_r, data.rp_mapper, reduction='sum')
                total_nodes += data.y_r.size(0)
                
                
        return total_loss/len(loader), total_correct/total_nodes

    all_train_loss = []
    all_train_acc = []

    all_valid_loss = []
    all_valid_acc = []

    for epoch in range(1, args.n_epochs+1):
        print(f'Epoch: {epoch:02d}', 5*'*')
        train_loss , train_acc = train(train_loader)
        #print(train_loss, train_acc)
        valid_loss , valid_acc  = validation_loss(validation_loader)

        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)

        all_valid_loss.append(valid_loss)
        all_valid_acc.append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0  
        else:
            counter += 1  
            
        if counter >= patience:
            print(f'Early stopping. No improvement in {epoch} epochs.')
            break
        
    path = args.output_path

    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), f'{path}/model.pth')


    with open(f'{path}/losses_train.txt', 'wb') as file:
           pickle.dump(all_train_loss, file)
           
    with open(f'{path}/acces_train.txt', 'wb') as file:
           pickle.dump(all_train_acc, file)


    with open(f'{path}/losses_valid.txt', 'wb') as file:
           pickle.dump(all_valid_loss, file)

    with open(f'{path}/acces_valid.txt', 'wb') as file:
           pickle.dump(all_valid_acc, file)

if __name__ == '__main__':
    main()
