from rdkit import Chem
from rdkit.Chem import Mol, rdchem
from typing import Any, List, Tuple, Union
import torch

def get_bond_adjmatrix(mol):
    bondtype_to_num = {Chem.rdchem.BondType.SINGLE: 1,
                       Chem.rdchem.BondType.DOUBLE: 2,
                       Chem.rdchem.BondType.TRIPLE: 3,
                       Chem.rdchem.BondType.AROMATIC: 1.5
                       }
    num_atoms = mol.GetNumAtoms()
    idx_to_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms()}
    bond_adj = torch.zeros((num_atoms, num_atoms))
    for bond in mol.GetBonds():
        atom1_map = idx_to_amap[bond.GetBeginAtomIdx()]
        atom2_map = idx_to_amap[bond.GetEndAtomIdx()]
        # if bond_adj[atom1_map, atom2_map] not in bondtype_to_num:
        #     print(bond_adj[atom1_map, atom2_map])
        bond_adj[atom1_map, atom2_map] = bondtype_to_num[bond.GetBondType()]
        bond_adj[atom2_map, atom1_map] = bondtype_to_num[bond.GetBondType()]
    return bond_adj

class BondEditAction():
    def __init__(self, atom_map1, atom_map2, bond_type, action_vocab):
        self.atom_map1 = atom_map1
        self.atom_map2 = atom_map2
        self.bond_type = bond_type
        self.action_vocab = action_vocab

    def get_tuple(self):
        return (self.action_vocab, self.bond_type)

    def apply(self, mol):
        new_mol = Chem.RWMol(mol)
        amap_idx = {atom.GetAtomMapNum() : atom.GetIdx() for atom in new_mol.GetAtoms()}
        # print(amap_idx[torch.tensor(1.0 ,device='cuda:0')])
        atom1 = new_mol.GetAtomWithIdx(amap_idx[self.atom_map1])
        atom2 = new_mol.GetAtomWithIdx(amap_idx[self.atom_map2])
        if self.bond_type == 0: # delete bond
            new_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())
            pred_mol = new_mol.GetMol()
        else:
            bond_type = 12 if self.bond_type == 1.5 else self.bond_type
            b_type = rdchem.BondType.values[bond_type]
            bond = new_mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())
            if bond is None:
                new_mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), b_type)
            else:
                bond.SetBondType(b_type)
            pred_mol = new_mol.GetMol()
        return pred_mol

def apply_edit_to_mol(mol: Chem.Mol, edit: Tuple):
    if edit[0] == 'D':
        edit_exe = BondEditAction(
            edit[1][0], edit[1][1], edit[1][2], action_vocab='Delete Bond')
        new_mol = edit_exe.apply(mol)
    elif edit[0] == 'C':
        edit_exe = BondEditAction(
            edit[1][0], edit[1][1], edit[1][2], action_vocab='Change Bond')
        new_mol = edit_exe.apply(mol)
    elif edit[0] == 'A':
        edit_exe = BondEditAction(
            edit[1][0], edit[1][1], edit[1][2], action_vocab='Add Bond')
        new_mol = edit_exe.apply(mol)
    else:
        raise ValueError('Invalid Edit')
    return new_mol

def get_product_from_edits(r_mol, edits: List):
    int_mol = Chem.Mol(r_mol)
    for edit in edits:
        edit_exe = BondEditAction(atom_map1=edit[0], atom_map2=edit[1], bond_type=edit[2], action_vocab='test')
        int_mol = edit_exe.apply(Chem.Mol(int_mol))
    p_mol = int_mol
    return p_mol


def get_edits_from_adj(r_adj, p_adj):
    edits_rxn = {'B': [], 'C': [], 'F': []}
    indices = torch.nonzero(torch.triu(r_adj - p_adj) != 0)
    for indice in indices:
        s, e = indice.tolist()
        if p_adj[s,e].item() != 0 and r_adj[s,e].item() == 0:
            edits_rxn['B'].append((s,e,r_adj[s,e].item()))
        elif p_adj[s,e].item() != 0 and r_adj[s,e].item() != p_adj[s,e].item():
            edits_rxn['C'].append((s,e,r_adj[s,e].item()))
        elif p_adj[s,e].item() == 0 and r_adj[s,e].item() > 0:
            edits_rxn['F'].append((s,e,r_adj[s,e].item()))
    return edits_rxn

def get_bond_select_from_mol(mol, bond_vocab, device):
    num_atoms = mol.GetNumAtoms()
    bondtype_to_num = {Chem.rdchem.BondType.SINGLE: 1,
                       Chem.rdchem.BondType.DOUBLE: 2,
                       # Chem.rdchem.BondType.TRIPLE: 3,
                       Chem.rdchem.BondType.AROMATIC: 1.5
                       }
    idx_to_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms()}
    non_zero_indices = torch.triu_indices(num_atoms, num_atoms, offset=1).T
    bond_select = torch.ones(non_zero_indices.size()[0] * len(bond_vocab) + 1).to(device)
    for bond in mol.GetBonds():
        atom1_map = idx_to_amap[bond.GetBeginAtomIdx()]
        atom2_map = idx_to_amap[bond.GetEndAtomIdx()]
        if atom1_map > atom2_map: atom2_map, atom1_map = atom1_map, atom2_map
        if bond.GetBondType() in bondtype_to_num:
            bond_type = bond_vocab[bondtype_to_num[bond.GetBondType()]]
            edit_num = int((non_zero_indices == torch.tensor((atom1_map, atom2_map))).all(dim=1).nonzero(as_tuple=True)[0].item() * len(bond_vocab) + bond_type)
            assert edit_num >= 0
            bond_select[edit_num] = 0
    return bond_select

