from rdkit import Chem
from typing import Any, List, Tuple, Union
import torch
from torch_geometric.data import Data, Batch

ATOM_SYMBOL_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'H', 'Si', 'P', 'B', 'I', 'Li', 'Na', 'K', 'Ca',
                    'Mg', 'Al', 'Cu', 'Zn', 'Sn', 'Se', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'As', 'Bi', 'Te', 'Sb',
                    'Ba', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Pt', 'Au', 'Pb', 'Cs', 'Sm', 'Os', 'Ir', '*', 'unk']

DEGREES = list(range(10))
NUM_Hs = [0, 1, 2, 3, 4]
CHIRALTAG = [0, 1, 2, 3]
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
ATOM_FDIM = len(ATOM_SYMBOL_LIST) + len(DEGREES) +  len(NUM_Hs) + len(CHIRALTAG) + len(HYBRIDIZATION) + 1
# print(ATOM_FDIM)
# 1: [atom.GetIsAromatic()]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BONDSTEREO = list(range(6))
BOND_FDIM = len(BOND_TYPES) + len(BONDSTEREO) + 2

def one_of_k_encoding(x: Any, allowable_set) -> List:
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def get_atom_features(atom: Chem.Atom) -> List[Union[bool, int, float]]:
    atom_features = (one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL_LIST) +
                     one_of_k_encoding(atom.GetDegree(), DEGREES) +
                     one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION) +
                     one_of_k_encoding(int(atom.GetProp('NumHs')), NUM_Hs) +
                     one_of_k_encoding(int(atom.GetChiralTag()), CHIRALTAG) +
                     [atom.GetIsAromatic()]
                     )
    return atom_features


def get_bond_features(bond: Chem.Bond) -> List[Union[bool, int, float]]:
    if bond is None:
        bond_features = [0 for _ in range(BOND_FDIM)]
    else:
        bond_features = one_of_k_encoding(bond.GetBondType(), BOND_TYPES) + \
            one_of_k_encoding(int(bond.GetStereo()), BONDSTEREO) + \
            [bond.GetIsConjugated()] + [bond.IsInRing()]
    return bond_features


def label_to_tensor(edit, num_atoms, bond_vocab):
    temp_edit = torch.zeros((num_atoms, num_atoms, len(bond_vocab)))
    temp_graph = torch.zeros(1)
    if edit is not None:
        edit_idx = bond_vocab[edit[2]]
        edit_atoms = sorted(edit[:2])
        temp_edit[edit_atoms[0], edit_atoms[1], edit_idx] = 1
    else:
        temp_graph[0] = 1
    edge_index_out = torch.triu_indices(num_atoms, num_atoms, offset=1)
    label = torch.cat([temp_edit[edge_index_out[0, :], edge_index_out[1, :]].flatten(), temp_graph])
    return label



def Rxn_Graph(mols, edits: list, bond_vocab: Any):
    num_atoms = mols[0].GetNumAtoms()
    amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mols[0].GetAtoms()}
    idx_to_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mols[0].GetAtoms()}

    f_atoms_all = []; f_bonds_all = []; edge_index_all = []
    for mol_idx, mol in enumerate(mols):
        numHs_list = [0] * mol.GetNumAtoms()
        for bond in mol.GetBonds():
            atom1 = mol.GetAtoms()[bond.GetBeginAtomIdx()]
            atom2 = mol.GetAtoms()[bond.GetEndAtomIdx()]
            if atom1.GetSymbol() == 'H':
                numHs_list[idx_to_amap[atom2.GetIdx()]] += 1
            if atom2.GetSymbol() == 'H':
                numHs_list[idx_to_amap[atom1.GetIdx()]] += 1
        for atom in mol.GetAtoms():
            atom.SetProp('NumHs', str(numHs_list[atom.GetAtomMapNum()]))

        f_atoms = []
        for map_ in range(num_atoms):
            f_atom = get_atom_features(mol.GetAtoms()[amap_to_idx[map_]])
            f_atoms.append(f_atom)

        edge_index_start = []; edge_index_end = []
        f_bonds = []
        for bond in mol.GetBonds():
            atom1_map = idx_to_amap[bond.GetBeginAtomIdx()]
            atom2_map = idx_to_amap[bond.GetEndAtomIdx()]
            edge_index_start.append(atom1_map); edge_index_end.append(atom2_map)
            edge_index_start.append(atom2_map); edge_index_end.append(atom1_map)
            f_bond = get_bond_features(bond)
            # f_bonds.append(f_bond + [f_atoms[atom2_map][i] - f_atoms[atom1_map][i] for i in range(len(f_atoms[atom1_map]))])
            # f_bonds.append(f_bond + [f_atoms[atom1_map][i] - f_atoms[atom2_map][i] for i in range(len(f_atoms[atom1_map]))])
            f_bonds.append(f_bond)
            f_bonds.append(f_bond)

        f_atoms = torch.tensor(f_atoms, dtype=torch.float)
        f_bonds = torch.tensor(f_bonds, dtype=torch.float)
        edge_index = torch.tensor([edge_index_start, edge_index_end], dtype=torch.long)
        if mol_idx == 0: edge_index_out = torch.triu_indices(num_atoms, num_atoms, offset=1)

        f_atoms_all.append(f_atoms)
        f_bonds_all.append(f_bonds)
        edge_index_all.append(edge_index)

    labels = []
    for edit in edits:
        label = label_to_tensor(edit, num_atoms, bond_vocab)
        assert torch.nonzero(label).size(0) == 1
        labels.append(label)
    data = Data(x=f_atoms_all, edge_attr=f_bonds_all, edge_index=edge_index_all, edge_index_out=edge_index_out, label=labels)
    data.num_nodes = num_atoms
    return data