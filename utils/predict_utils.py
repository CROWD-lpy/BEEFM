import torch
from collections import deque
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data, Batch
import argparse, os, glob, json, joblib
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')


os.chdir('/raid/home/liangpengyu/romote_project/Elementary_reaction/graph_20240926/utils')

import sys
sys.path.append('/raid/home/liangpengyu/romote_project/Elementary_reaction/graph_20240926')

from process_utils import ATOM_FDIM, BOND_FDIM, get_atom_features, get_bond_features
from mol_operation import BondEditAction
from model.model import RXN_Sequence


def get_bond_adjmatrix(mol):
    bondtype_to_num = {Chem.rdchem.BondType.SINGLE: 1,
                       Chem.rdchem.BondType.DOUBLE: 2,
                       Chem.rdchem.BondType.TRIPLE: 3,
                       Chem.rdchem.BondType.AROMATIC: 1.5}
    num_atoms = mol.GetNumAtoms()
    idx_to_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms()}
    bond_adj = torch.zeros((num_atoms, num_atoms))
    for bond in mol.GetBonds():
        atom1_map = idx_to_amap[bond.GetBeginAtomIdx()]
        atom2_map = idx_to_amap[bond.GetEndAtomIdx()]
        bond_adj[atom1_map, atom2_map] = bondtype_to_num[bond.GetBondType()]
        bond_adj[atom2_map, atom1_map] = bondtype_to_num[bond.GetBondType()]
    return bond_adj

def Rxn_Graph_WithoutLabel(react_mol: Chem.Mol):
    num_atoms = react_mol.GetNumAtoms()
    num_bonds = react_mol.GetNumBonds()
    amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in react_mol.GetAtoms()}
    idx_to_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in react_mol.GetAtoms()}

    f_atoms = []
    for map_ in range(num_atoms):

        numHs_list = [0] * react_mol.GetNumAtoms()
        for bond in react_mol.GetBonds():
            atom1 = react_mol.GetAtoms()[bond.GetBeginAtomIdx()]
            atom2 = react_mol.GetAtoms()[bond.GetEndAtomIdx()]
            if atom1.GetSymbol() == 'H':
                numHs_list[idx_to_amap[atom2.GetIdx()]] += 1
            if atom2.GetSymbol() == 'H':
                numHs_list[idx_to_amap[atom1.GetIdx()]] += 1
        for atom in react_mol.GetAtoms():
            atom.SetProp('NumHs', str(numHs_list[atom.GetAtomMapNum()]))

        f_atom = get_atom_features(react_mol.GetAtoms()[amap_to_idx[map_]])
        f_atoms.append(f_atom)

    edge_index_start = []; edge_index_end = []
    f_bonds = []
    for bond in react_mol.GetBonds():
        atom1_map = idx_to_amap[bond.GetBeginAtomIdx()]
        atom2_map = idx_to_amap[bond.GetEndAtomIdx()]
        edge_index_start.append(atom1_map); edge_index_end.append(atom2_map)
        edge_index_start.append(atom2_map); edge_index_end.append(atom1_map)
        f_bond = get_bond_features(bond)
        f_bonds.append(f_bond)
        f_bonds.append(f_bond)
    f_atoms = torch.tensor(f_atoms, dtype=torch.float)
    f_bonds = torch.tensor(f_bonds, dtype=torch.float)
    edge_index = torch.tensor([edge_index_start, edge_index_end], dtype=torch.long)
    edge_index_out = torch.triu_indices(num_atoms, num_atoms, offset=1)
    data = Data(x=f_atoms, edge_attr=f_bonds, edge_index=edge_index, edge_index_out=edge_index_out)
    return data

def check_equal(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return torch.equal(x, y)
    else:
        for x_, y_ in zip(x,y):
            if not torch.equal(x_, y_):
                return False
        return True

def get_smiles(x):
    heavy_atom = {}; heavy_atom_num = 0
    for atom in x.GetAtoms():
        if atom.GetSymbol() != 'H':
            heavy_atom[heavy_atom_num] = (atom.GetAtomMapNum(), atom.GetAtomicNum())
            heavy_atom_num += 1

    new_mol = Chem.RWMol(); hydrogen_num = {}
    for key, value in heavy_atom.items():
        atom_idx = new_mol.AddAtom(Chem.Atom(value[1]))
        new_mol.GetAtoms()[atom_idx].SetAtomMapNum(value[0])
        hydrogen_num[value[0]] = 0
    amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()}

    for bond in x.GetBonds():
        atom1, atom2 = x.GetAtoms()[bond.GetBeginAtomIdx()], x.GetAtoms()[bond.GetEndAtomIdx()]
        if atom1.GetSymbol() == 'H' and atom2.GetSymbol() != 'H':
            hydrogen_num[atom2.GetAtomMapNum()] += 1
        if atom2.GetSymbol() == 'H' and atom1.GetSymbol() != 'H':
            hydrogen_num[atom1.GetAtomMapNum()] += 1
        if atom2.GetSymbol() != 'H' and atom1.GetSymbol() != 'H':
            bond_type = bond.GetBondType()
            new_mol.AddBond(amap_to_idx[atom1.GetAtomMapNum()], amap_to_idx[atom2.GetAtomMapNum()], bond_type)

    for atom_map, num_Hs in hydrogen_num.items():
        new_mol.GetAtoms()[amap_to_idx[atom_map]].SetNumExplicitHs(num_Hs)
    for atom in new_mol.GetAtoms(): atom.SetAtomMapNum(0)
    try:
        return Chem.MolToSmiles(new_mol, canonical=True)
    except Exception:
        return None

    # x_copy = Chem.Mol(x)
    # for atom in x_copy.GetAtoms(): atom.SetAtomMapNum(0)
    # hydrogen_nums = {}
    # for bond in x_copy.GetBonds():
    #     atom1, atom2 = x_copy.GetAtoms()[bond.GetBeginAtomIdx()], x_copy.GetAtoms()[bond.GetEndAtomIdx()]
    #     if atom1.GetSymbol() == 'H' and atom2.GetSymbol() != 'H':
    #         if atom1.GetIdx() not in hydrogen_nums:
    #             hydrogen_nums[atom2.GetIdx()] = 1
    #         else:
    #             hydrogen_nums[atom2.GetIdx()] += 1
    #     if atom2.GetSymbol() == 'H' and atom1.GetSymbol() != 'H':
    #         if atom1.GetIdx() not in hydrogen_nums:
    #             hydrogen_nums[atom1.GetIdx()] = 1
    #         else:
    #             hydrogen_nums[atom1.GetIdx()] += 1
    # for atom_idx, num_Hs in hydrogen_nums.items():
    #     x_copy.GetAtoms()[atom_idx].SetNumExplicitHs(num_Hs)
    #
    # mol_x = Chem.RemoveHs(x_copy, sanitize=False)
    # try:
    #     smiles_x = Chem.MolToSmiles(mol_x, canonical=True)
    #     return smiles_x
    # except Exception:
    #     return None

def model_test(args):
    '''
    return (prodcut_mols: List(Tuple), correctness_idx: int)
    prodcuts_mols[x]: (mol, prop, bond_adj, smiles, edits)
    correctness_idx: -1 represents wrong, x>0 represents prodcut_mols[x] is the same as true product
    '''
    model, idx, rxn_data, bond_vocab, device, step_cut, topk_split = args

    model.to(device)
    model.eval()
    with torch.no_grad():
        r_mol = rxn_data[0]
        edits_real = rxn_data[1]
        int_mol = Chem.Mol(r_mol)
        # print(edits_real)
        for edit in edits_real:
            edit_exe = BondEditAction(atom_map1=edit[0], atom_map2=edit[1], bond_type=edit[2], action_vocab='test')
            int_mol = edit_exe.apply(Chem.Mol(int_mol))
        p_adj = get_bond_adjmatrix(int_mol)
        p_smiles = get_smiles(int_mol)

        data_graph = Rxn_Graph_WithoutLabel(react_mol=Chem.Mol(r_mol)).to(device)
        data_batch = Batch.from_data_list([data_graph])  # Data(x, edge_attr, edge_index, edge_index_out)

        edits_pred = model.predict(data_batch, Chem.Mol(r_mol), step_cut, topk_split)
        pred_tuple = []

        for edits in edits_pred:
            int_mol = Chem.Mol(r_mol)
            for edit in edits:
                if edit[2] != -1:
                    edit_exe = BondEditAction(atom_map1=edit[0], atom_map2=edit[1], bond_type=edit[2], action_vocab='test')
                    int_mol = edit_exe.apply(Chem.Mol(int_mol))
            total_value = edits[-1][3]
            pred_adj = get_bond_adjmatrix(int_mol)
            pred_smiles = get_smiles(int_mol)
            if pred_smiles is not None:
                pred_tuple.append((int_mol, total_value, pred_adj, pred_smiles, edits))

        pred_tuple_deduplication = []; smiles_set = []
        for i in range(len(pred_tuple)):
            check_same = -1
            for j, smiles in enumerate(smiles_set):
                if smiles == pred_tuple[i][3]:
                    check_same = j
                    break
            if check_same >= 0:
                pred_tuple_deduplication[check_same]= (pred_tuple_deduplication[check_same][0],
                                                       pred_tuple_deduplication[check_same][1] + pred_tuple[i][1],
                                                       pred_tuple_deduplication[check_same][2],
                                                       pred_tuple_deduplication[check_same][3],
                                                       pred_tuple_deduplication[check_same][4])
            else:
                smiles_set.append(pred_tuple[i][3])
                pred_tuple_deduplication.append(pred_tuple[i])
        pred_tuple = pred_tuple_deduplication

        prop_sum = sum([pred_tuple_[1] for pred_tuple_ in pred_tuple])
        prop_list = [pred_tuple_[1]/prop_sum for pred_tuple_ in pred_tuple]
        pred_tuple = [(pred_tuple_[0], prop_list[tuple_idx], pred_tuple_[2], pred_tuple_[3], pred_tuple_[4]) for tuple_idx, pred_tuple_ in enumerate(pred_tuple)]
        pred_tuple = sorted(pred_tuple, key=lambda x: x[1], reverse=True)

        correctness_idx = -1
        for pred_idx, pred_tuple_ in enumerate(pred_tuple):
            # print(pred_tuple_[-1], pred_tuple_[-2], pred_tuple_[1])
            if pred_tuple_[3] == p_smiles:
                correctness_idx = pred_idx

        # print(idx, correctness_idx)
        return pred_tuple, correctness_idx


def main(args):
    out_dir = os.path.join('../experiments', args['experiment'])
    pt_files = glob.glob(os.path.join(out_dir, 'checkpoints', '*.pt'))
    best_epoch = max([int(os.path.basename(pt_file).split('.')[0].split('_')[1]) for pt_file in pt_files])
    best_pt_file = os.path.join(out_dir, 'checkpoints', f'epoch_{best_epoch}.pt')
    print(best_pt_file)

    json_path = os.path.join(out_dir, 'config.json')
    config = json.load(open(json_path, 'r'))
    config['n_atom_feat'] = ATOM_FDIM
    config['n_bond_feat'] = BOND_FDIM

    dataset = config['dataset']
    data_dir = os.path.join('../data', dataset)
    bond_vocab_dict = joblib.load(os.path.join(data_dir, 'bond_vocab.txt'))
    bond_vocab_forward = {key: i for i, key in enumerate(bond_vocab_dict.keys())}
    bond_vocab_reverse = {i: key for i, key in enumerate(bond_vocab_dict.keys())}
    bond_vocab = (bond_vocab_forward, bond_vocab_reverse)

    model = RXN_Sequence(config=config, bond_vocab=bond_vocab[0], device=args['device'])
    checkpoint = torch.load(best_pt_file)
    model.load_state_dict(checkpoint['state'])
    model.to(args['device'])

    if args['data_file'] is None:
        test_data = joblib.load(os.path.join(data_dir, 'test.file'))
    else:
        test_data = joblib.load(args['data_file'])

    pred_rxns = []
    for idx, test_data_ in enumerate(test_data):
        args_list = (model, idx, test_data_, bond_vocab, args['device'], args['step_cut'], args['tree_child_num'])
        pred_rxns.append(model_test(args_list))

    extension = os.path.splitext(os.path.basename(args['data_file']))[1]
    pred_path = args['data_file'][:-len(extension)] + '_pred' + extension
    joblib.dump(pred_rxns, pred_path)
    print(f"{args['data_file']} Finished Successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='26-09-2024--17-21-37', help='12-10-2024--11-51-14')
    parser.add_argument('--data_file', type=str, default=None, help='../data/grambow/processing_BCF_False/valid.file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--step_cut', type=int, default=8, help='max edit step for a elementary reaction')
    parser.add_argument('--tree_child_num', type=int, default=3, help="only work when test_method = 'tree', best option = 2/3")

    args = parser.parse_args().__dict__
    main(args)