
import torch
import numpy as np
from torch_geometric.data import Data
import collections
import rdkit.Chem as Chem
pt = Chem.GetPeriodicTable()
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')    
import pandas as pd 
from torch_geometric.loader import DataLoader
import random
import os

"""
'GraphData' is used to encapsulate information about a molecular graph.
"""
GraphData = collections.namedtuple('GraphData', [
    'n_nodes',
    'node_features',
    'edge_features',
    'edge_index',
    'atom_types'])


class PairData(Data):
    """
    Args:
        edge_index_r: Edge indices for the reactant graph.
        x_r: Node features for the reactant graph.
        edge_index_p: Edge indices for the product graph.
        x_p: Node features for the product graph.
        edge_feat_r: Edge features for the reactant graph.
        edge_feat_p: Edge features for the product graph.
        y_r: A list of atom mapping value based on graph traverse (atom indices) for the reactant graph.
        y_p: A list of atom mapping value based on graph traverse (atom indices) for the product graph.
        p2r_mapper: A mapper function to maps atoms in product to reactant.
        eq_as: Equivalent atoms to consider molecule symmetry for product graph.
        act_n_r: Actual number of reactant atoms.
        act_n_p: Actual number of product atoms.
        z_r: Atomic numbers for reactant atoms.
        z_p: Atomic numbers for product atoms.
    """
    def __init__(self, edge_index_r=None, x_r=None, edge_index_p=None, x_p=None,  \
                 edge_feat_r = None, edge_feat_p = None, y_r = None, y_p = None,  p2r_mapper = None, eq_as = None,
                 act_n_r=None, act_n_p=None, z_r=None, z_p=None):
        super().__init__()
        self.edge_index_r = edge_index_r
        self.x_r = x_r
        self.edge_index_p = edge_index_p
        self.x_p = x_p
        self.y_r = y_r
        self.y_p = y_p
        self.edge_feat_r = edge_feat_r
        self.edge_feat_p = edge_feat_p
        self.p2r_mapper = p2r_mapper
        self.eq_as = eq_as
        self.act_n_r = act_n_r
        self.act_n_p = act_n_p
        self.z_r = z_r
        self.z_p = z_p

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_r':
            return self.x_r.size(0)
        if key == 'edge_index_p':
            return self.x_p.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class MoleculeDataset(torch.utils.data.Dataset):
    """
    Initialize the MoleculeDataset.
    Args:
        root (str): Root directory of the dataset.
        filename (str): Filename of the CSV file.
        num_wl_iterations (int): Number of iterations used for finding chemically equivalent atoms.
        santitize (bool): Whether to sanitize the molecules.
    """

    def __init__(self, root, filename, num_wl_iterations=3, santitize=False, use_edits=False):
        self.root = root
        self.filename = filename
        self.filepath = os.path.join(root, filename)
        
        # Read CSV. The new format has 'reactions' as header.
        try:
            self.reactions = pd.read_csv(self.filepath)
        except Exception as e:
            print(f"Error reading {self.filepath}: {e}")
            raise e
            
        self.num_wl_iterations = num_wl_iterations
        self.santitize= santitize
        self.use_edits = use_edits

    def __len__(self):
        return len(self.reactions)

    def _get_graph_data_tensor(self,mol):
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        
        X = np.zeros((n_nodes, 111)) # Assuming 111 features based on get_atom_features
        Z = []
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            Z.append(atom.GetAtomicNum())
            
        X = torch.tensor(X, dtype = torch.float)
        Z = torch.tensor(Z, dtype = torch.long)
        
        if n_edges > 0:
            (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
            E = torch.stack([torch_rows, torch_cols], dim = 0)
            
            EF = np.zeros((n_edges, 10)) # Assuming 10 features for bonds
            for (k, (i,j)) in enumerate(zip(rows, cols)):
                EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
            EF = torch.tensor(EF, dtype = torch.float)
        else:
            E = torch.zeros((2, 0), dtype=torch.long)
            EF = torch.zeros((0, 10), dtype=torch.float)

        return GraphData(
                n_nodes,
                node_features= X,
                edge_features =EF,
                edge_index= E,
                atom_types= Z)

    def _get_reaction_mols(self,reaction_smiles):
        reactantes_smiles, products_smiles = reaction_smiles.split('>>')
        reactantes_mol = Chem.MolFromSmiles(reactantes_smiles)
        products_mol = Chem.MolFromSmiles(products_smiles)
        if self.santitize:
            Chem.SanitizeMol(reactantes_mol)
            Chem.SanitizeMol(products_mol)
        return reactantes_mol, products_mol 

    def _get_mapping_number(self, mol):
        mapping = []
        for atom in mol.GetAtoms():
                mapping.append(atom.GetAtomMapNum()) # Keep 1-based or 0-based? AMNet used -1. RDKit uses >0 for mapped. 0 for unmapped.
                # If we use 0-based index for p2r_mapper, we should probably keep map num as is and match them.
        return mapping

    def __getitem__(self, idx):
        if 'reactions' in self.reactions.columns:
            reaction_smiles = self.reactions.iloc[idx]['reactions']
        else:
            reaction_smiles = self.reactions.iloc[idx, 0]

        reactantes_mol, products_mol = self._get_reaction_mols(reaction_smiles)
        
        # If product has unmapped atoms (map num 0), we should handle it.
        # But for now assuming full mapping or handling simply.

        reactant_graph = self._get_graph_data_tensor(reactantes_mol)
        product_graph = self._get_graph_data_tensor(products_mol)

        edge_index_r = reactant_graph.edge_index
        x_r = reactant_graph.node_features
        edge_feat_r = reactant_graph.edge_features
        z_r = reactant_graph.atom_types

        edge_index_p = product_graph.edge_index
        x_p = product_graph.node_features
        edge_feat_p = product_graph.edge_features
        z_p = product_graph.atom_types

        # Calculate eq_as for product symmetry
        eq_as = get_equivalent_atoms(products_mol, self.num_wl_iterations)

        # Calculate p2r_mapper
        y_r_map = self._get_mapping_number(reactantes_mol) # List of map numbers
        y_p_map = self._get_mapping_number(products_mol)   # List of map numbers

        # p2r_mapper[i] = index of reactant atom that has same map number as product atom i
        p2r_list = []
        for p_map in y_p_map:
            if p_map == 0:
                p2r_list.append(-100) # Unmapped, use -100 for nll_loss ignore_index
            else:
                try:
                    r_idx = y_r_map.index(p_map)
                    p2r_list.append(r_idx)
                except ValueError:
                    p2r_list.append(-100) # Mapped but not found in reactant (should not happen in balanced reaction)
        
        p2r_mapper = torch.tensor(p2r_list, dtype=torch.long)
        
        y_r = torch.tensor(y_r_map) # Map numbers
        y_p = torch.tensor(y_p_map)

        act_n_r = torch.tensor(x_r.shape[0])
        act_n_p = torch.tensor(x_p.shape[0])

        data = PairData(
            edge_index_r=edge_index_r, x_r=x_r, 
            edge_index_p=edge_index_p, x_p=x_p, 
            edge_feat_r=edge_feat_r, edge_feat_p=edge_feat_p, 
            y_r=y_r, y_p=y_p,
            p2r_mapper=p2r_mapper, 
            eq_as=eq_as,
            act_n_r=act_n_r,
            act_n_p=act_n_p,
            z_r=z_r, z_p=z_p
        )
            
        return data

def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, use_chirality = True, hydrogens_implicit = True):
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', \
                                'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', \
                                'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', \
                                'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', \
                                'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', \
                                'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', \
                                'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    atom_type  = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors  = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge  = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type  = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    ex_valence = one_hot_encoding(int(atom.GetExplicitValence()), list(range(1, 7)))
    imp_valence = one_hot_encoding(int(atom.GetImplicitValence()), list(range(0, 6)))
    is_in_a_ring = [int(atom.IsInRing())]
    is_aromatic = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)] 
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    
    atom_feature_vector = atom_type  + n_heavy_neighbors + is_in_a_ring  + is_aromatic  + atomic_mass_scaled \
                         + ex_valence + imp_valence  \
                         + vdw_radius_scaled + covalent_radius_scaled  + hybridisation_type + formal_charge                                
    
    if use_chirality == True:
        chirality_type  = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type
    
    if hydrogens_implicit == True:
        n_hydrogens  = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens

    return np.array(atom_feature_vector)

def get_bond_features(bond, use_stereochemistry = True):
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type  = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj  = [int(bond.GetIsConjugated())]
    bond_is_in_ring  = [int(bond.IsInRing())]
    bond_feature_vector = bond_type  + bond_is_conj  + bond_is_in_ring

    if use_stereochemistry == True:
        stereo_type  = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type

    return np.array(bond_feature_vector)

def wl_atom_similarity(mol, num_wl_iterations):
    label_dict = dict()
    for atom in mol.GetAtoms():
        label_dict[atom.GetIdx()]= atom.GetSymbol()

    for _ in range(num_wl_iterations):
        label_dict = update_atom_labels(mol, label_dict)

    return label_dict

def update_atom_labels(mol, label_dict):
    new_label_dict = {}
    for atom in mol.GetAtoms():
        neighbors_index = [n.GetIdx() for n in atom.GetNeighbors()]
        neighbors_index.sort()
        label_string = label_dict[atom.GetIdx()]
        for neighbor in neighbors_index:
            label_string += label_dict[neighbor]
        new_label_dict[atom.GetIdx()] = label_string
    return new_label_dict

def get_equivalent_atoms(mol, num_wl_iterations):
    node_similarity = wl_atom_similarity(mol, num_wl_iterations)
    n_h_dict = dict()
    for atom in mol.GetAtoms():
        n_h_dict[atom.GetIdx()]= atom.GetTotalNumHs()
    degree_dict = dict()
    for atom in mol.GetAtoms():
        degree_dict[atom.GetIdx()] = atom.GetDegree()
    neighbor_dict = dict()
    for atom in mol.GetAtoms():
        neighbor_dict[atom.GetIdx()]= [nbr.GetSymbol() for nbr in atom.GetNeighbors()]
        
    atom_equiv_classes = []
    visited_atoms = set()
    for centralnode_indx, centralnodelabel in node_similarity.items():
        equivalence_class = set()

        if centralnode_indx not in visited_atoms:
            visited_atoms.add(centralnode_indx) 
            equivalence_class.add(centralnode_indx)

        for firstneighbor_indx, firstneighborlabel in node_similarity.items():
            if firstneighbor_indx not in visited_atoms and centralnodelabel[0] == firstneighborlabel[0] and \
                    set(centralnodelabel[1:]) == set(firstneighborlabel[1:]) and \
                    degree_dict[centralnode_indx] == degree_dict[firstneighbor_indx]  and \
                    len(centralnodelabel)== len(firstneighborlabel) and \
                    set(neighbor_dict[centralnode_indx]) == set(neighbor_dict[firstneighbor_indx]) and \
                    n_h_dict[centralnode_indx] == n_h_dict[firstneighbor_indx]:
                    equivalence_class.add(firstneighbor_indx)
                    visited_atoms.add(firstneighbor_indx)
        if equivalence_class :
            atom_equiv_classes.append(equivalence_class)
          
    return atom_equiv_classes
