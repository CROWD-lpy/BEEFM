import argparse, os, glob, json, joblib
from tqdm import tqdm
import sys
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data, Batch

os.chdir('/raid/home/liangpengyu/romote_project/Elementary_reaction/graph_20240926/utils')
sys.path.append('/raid/home/liangpengyu/romote_project/Elementary_reaction/graph_20240926')

from model.model import RXN_Sequence
from process_utils import ATOM_FDIM, BOND_FDIM
from mol_operation import BondEditAction
from predict_utils import get_smiles, Rxn_Graph_WithoutLabel, get_bond_adjmatrix

def model_test(args):
    model, idx, mol, bond_vocab, device, step_cut, topk_split = args

    model.to(device)
    model.eval()
    with torch.no_grad():
        data_graph = Rxn_Graph_WithoutLabel(react_mol=Chem.Mol(mol)).to(device)
        data_batch = Batch.from_data_list([data_graph])  # Data(x, edge_attr, edge_index, edge_index_out)
        edits_pred = model.predict(data_batch, Chem.Mol(mol), step_cut, topk_split)
        pred_tuple = []

        for edits in edits_pred:
            int_mol = Chem.Mol(mol)
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

        # print([pred[-1] for pred in pred_tuple])

        return pred_tuple


def model_test_seq(args):
    model, idx, rxn_data, bond_vocab, device, step_cut, elementary_cutoff, topk_split = args
    p_smiles = get_smiles(rxn_data[1])
    step = rxn_data[2]; r_mol = rxn_data[0]; mols = [(r_mol, 1, [])]
    for step_idx in range(step - 1):
        prods = []; prod_smiles = {}
        for mol, value, _ in mols:
            args_step = (model, idx, mol, bond_vocab, device, step_cut, topk_split)
            pred_prods = model_test(args_step)
            prod_num = 0
            for prod in pred_prods:
                if prod[-1][0][0] == -1: continue
                if prod_num == 2: break
                prod_num += 1
                if prod[1] > elementary_cutoff:
                    if prod[3] not in prod_smiles:
                        prods.append((prod[0], prod[1] * value, prod[-1]))
                        prod_smiles[prod[3]] = len(prods) - 1
                    else:
                        value_before = prods[prod_smiles[prod[3]]][1]
                        prods[prod_smiles[prod[3]]] = (prod[0], prod[1] * value + value_before, prod[-1])
        mols = prods

    prods = sorted(mols, key=lambda x:x[1], reverse=True)
    correctness_idx = -1

    for prod_idx, prod in enumerate(prods):
        pred_smile = get_smiles(prod[0])
        if pred_smile == p_smiles:
            correctness_idx = prod_idx
            break
    print(idx, correctness_idx)
    return prods, correctness_idx



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
        args_list = (model, idx, test_data_, bond_vocab, args['device'], args['step_cut'],
                     args['elementary_cutoff'], args['tree_child_num'])
        pred_rxns.append(model_test_seq(args_list))

    extension = os.path.splitext(os.path.basename(args['data_file']))[1]
    pred_path = args['data_file'][:-len(extension)] + '_pred' + extension
    joblib.dump(pred_rxns, pred_path)
    print(f"{args['data_file']} Finished Successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='03-10-2024--13-12-12')
    parser.add_argument('--data_file', type=str, default='../data/coley/processing_BCF_False/seq_data2.file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--step_cut', type=int, default=8, help='max edit step for a elementary reaction')
    parser.add_argument('--elementary_cutoff', type=float, default=0)
    parser.add_argument('--tree_child_num', type=int, default=3, help="only work when test_method = 'tree', best option = 2/3")

    args = parser.parse_args().__dict__
    main(args)