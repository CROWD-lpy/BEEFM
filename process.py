import argparse, os, joblib
from tqdm import tqdm
from rdkit import Chem, RDLogger
import random
import torch
from torch_geometric.data import Data, Batch

RDLogger.DisableLog('rdApp.*')

from utils.edit_operation import (atommapnum_minus_1, check_edit_bond, check_edit_hydrogen_charge,
                                  get_edits, shuffle_edits, check_aro, update_statistic, update_Hs_mapnum)
from utils.mol_operation import BondEditAction, get_bond_adjmatrix, get_product_from_edits, get_edits_from_adj
from utils.process_utils import Rxn_Graph

def preprocessing(process_dir, args):
    data_path = f"./data/{args.dataset}/origin_wb97xd_selected"
    os.makedirs(process_dir, exist_ok=True)

    for data_type in ['train', 'valid', 'test']:
        data_path_ = os.path.join(data_path, f"{data_type}.proc")
        with open(data_path_, 'r') as file: rxns = file.readlines()

        smiles = [rxn.split()[0] for rxn in rxns]
        edits  = [rxn.split()[1] for rxn in rxns]

        edits_type = {}; edits_length = {}
        rxns_process = []; rxn_remove = []
        rxn_none = 0; rxn_nochange = 0; rxn_aro = 0

        for smile_idx, (smile, edits_rxn) in enumerate(tqdm(zip(smiles, edits), total=len(smiles))):
            r_smi, p_smi = smile.split('>>')
            if args.dataset == 'coley':
                r_mol = Chem.MolFromSmiles(r_smi)
                if r_mol is None:
                    rxn_none += 1;
                    rxn_remove.append(smile_idx)
                    continue
                r_mol = atommapnum_minus_1(r_mol)
                r_mol_dh = Chem.AddHs(r_mol)
                update_Hs_mapnum(r_mol_dh)
            else:
                r_mol = Chem.MolFromSmarts(r_smi)
                if r_mol is None:
                    rxn_none += 1;
                    rxn_remove.append(smile_idx)
                    continue
                r_mol = atommapnum_minus_1(r_mol)
                r_mol_dh = r_mol

            edits_bond = check_edit_bond(edits_rxn.split('/')[0]) if edits_rxn.split('/')[0] != '' else []
            edits_hydrogen = check_edit_hydrogen_charge(edits_rxn.split('/')[1]) if edits_rxn.split('/')[1] != '' else []
            edits_charge = check_edit_hydrogen_charge(edits_rxn.split('/')[2]) if edits_rxn.split('/')[2] != '' else []

            if args.remove_nochange and len(edits_bond) + len(edits_hydrogen) == 0:
                rxn_nochange += 1; rxn_remove.append(smile_idx)
                continue

            r_mol_dh, edits_rxn = get_edits(r_mol_dh, edits_bond, edits_hydrogen)
            if args.remove_aromatic and check_aro(edits_rxn):
                rxn_aro += 1; rxn_remove.append(smile_idx)
                continue

            r_mol_dh, edits_list = shuffle_edits(r_mol_dh, edits_rxn, args.edit_order)
            edits_type, edits_length = update_statistic(edits_list, edits_type, edits_length)
            rxns_process.append((r_mol_dh, edits_list))

            if args.add_reverse:
                p_mol = get_product_from_edits(r_mol_dh, edits_list)
                r_adj, p_adj = get_bond_adjmatrix(r_mol_dh), get_bond_adjmatrix(p_mol)
                edits_rxn = get_edits_from_adj(r_adj, p_adj)
                p_mol, edits_list = shuffle_edits(p_mol, edits_rxn, args.edit_order)
                edits_type, edits_length = update_statistic(edits_list, edits_type, edits_length)
                rxns_process.append((p_mol, edits_list))

        print(f"{data_type} reactions with {len(rxns)} reactions complete getting edits...")
        print(f"Removed {rxn_none} reactions containing empty structures...")
        print(f"Removed {rxn_nochange} reactions have no changes...")
        print(f"Removed {rxn_aro} reactions contain aromatic bonds...")
        print('edits_length: ', edits_length)
        print(f"{len(rxns_process)} reaction saved...")
        if data_type == 'train':
            print(f'edit_type in {data_type} data: {edits_type}')
            joblib.dump(rxn_remove, os.path.join(process_dir, f"{data_type}_remove.index"))
            joblib.dump(rxns_process, os.path.join(process_dir, f'{data_type}.file'))
            joblib.dump(edits_type, os.path.join(process_dir, 'bond_vocab.txt'))
        else:
            bond_vocab = joblib.load(os.path.join(process_dir, 'bond_vocab.txt'))
            cover_num = 0
            for edit_type in edits_type.keys():
                if edit_type in bond_vocab:
                    cover_num += 1
            joblib.dump(rxn_remove, os.path.join(process_dir, f"{data_type}_remove.index"))
            joblib.dump(rxns_process, os.path.join(process_dir, f'{data_type}.file'))
            print(f'edit_type in {data_type} data: {edits_type}')
            print(f"The cover rate of edit type in {data_type} data is {cover_num}/{len(bond_vocab)}.")


def processing(process_dir, args):
    bond_vocab_dict = joblib.load(os.path.join(process_dir, 'bond_vocab.txt'))
    bond_vocab = {key: i for i, key in enumerate(bond_vocab_dict.keys())}
    for data_type in ['valid', 'test', 'train']:
        save_dir = os.path.join(process_dir, f'{data_type}')
        os.makedirs(save_dir, exist_ok=True)

        rxns = joblib.load(os.path.join(process_dir, f'{data_type}.file'))
        random.seed(args.random)
        random.shuffle(rxns)

        data_graphs = []; batch_num = 0
        for rxn in tqdm(rxns):
            r_mol = rxn[0]; edits = rxn[1]
            if len(edits) > 7: continue
            edits += [None] * (8 - len(edits))

            mols = [r_mol]; int_mol = r_mol
            for edit in edits[:-1]:
                if edit is not None:
                    edit_exe = BondEditAction(atom_map1=edit[0], atom_map2=edit[1], bond_type=edit[2], action_vocab='train')
                    int_mol = edit_exe.apply(Chem.Mol(int_mol))
                mols.append(int_mol)

            data_graph = Rxn_Graph(mols, edits = edits, bond_vocab=bond_vocab)
            data_graphs.append(data_graph)

            if len(data_graphs) == args.batch_size:
                batch = Batch.from_data_list(data_graphs)
                torch.save(batch, os.path.join(save_dir, f"{batch_num:05d}.pt"))
                data_graphs = []; batch_num += 1

        if len(data_graphs) != 0:
            batch = Batch.from_data_list(data_graphs)
            torch.save(batch, os.path.join(save_dir, f"{batch_num:05d}.pt"))
            batch_num += 1
        print(f"{data_type} dataset have saved {batch_num} batches.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='grambow',
                        help='Dataset: coley, combinatorial_all, combinatorial-seperated_elec_nuc, manually_curated')
    parser.add_argument('--edit_order', type=str, default='BCF',
                        help='Order with edits generated: BCF(Break-Change-Form), FCB')
    parser.add_argument('--add_reverse', type=bool, default=False,
                        help='Whether add reverse reaction into dataset')
    parser.add_argument('--remove_nochange', type=bool, default=True,
                        help='Whether add reactions with no change')
    parser.add_argument('--remove_aromatic', type=bool, default=False,
                        help='Whether add reactions with aromatic bonds')
    parser.add_argument('--random', type=int, default=42,
                        help='Seed of Random Shuffle')
    parser.add_argument("--batch_size", default=48,
                        type=int, help="Number of batch size")
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    process_dir = f'./data/{args.dataset}/processing_{args.edit_order}_{args.add_reverse}'
    preprocessing(process_dir=process_dir, args=args)
    processing(process_dir=process_dir, args=args)


'''
train: Counter({2: 270483, 1: 174804, 4: 28192, 3: 70388, 7: 663, 6: 3, 5: 543})
valid: Counter({2: 33408, 1: 21499, 4: 3534, 3: 8902, 7: 84, 5: 67})
test : Counter({2: 33669, 1: 22125, 4: 3551, 3: 8674, 7: 78, 5: 73})

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 719050/719050 [12:08<00:00, 987.39it/s]
train reactions with 719050 reactions complete getting edits...
Removed 2359 reactions containing empty structures...
Removed 0 reactions have no changes...
Removed 0 reactions contain aromatic bonds...
edits_length:  {2: 270483, 1: 174804, 0: 171615, 4: 28192, 3: 70388, 7: 663, 6: 3, 5: 543}
716691 reaction saved...
edit_type in train data: {0.0: 441316, 1.0: 490426, 2.0: 115277, 1.5: 57}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 89081/89081 [01:29<00:00, 995.39it/s]
valid reactions with 89081 reactions complete getting edits...
Removed 297 reactions containing empty structures...
Removed 0 reactions have no changes...
Removed 0 reactions contain aromatic bonds...
edits_length:  {2: 33408, 1: 21499, 0: 21290, 4: 3534, 3: 8902, 7: 84, 5: 67}
88784 reaction saved...
edit_type in valid data: {0.0: 55018, 1.0: 60765, 2.0: 14297}
The cover rate of edit type in valid data is 3/4.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90024/90024 [01:30<00:00, 999.68it/s]
test reactions with 90024 reactions complete getting edits...
Removed 374 reactions containing empty structures...
Removed 0 reactions have no changes...
Removed 0 reactions contain aromatic bonds...
edits_length:  {2: 33669, 1: 22125, 0: 21480, 4: 3551, 3: 8674, 7: 78, 5: 73}
89650 reaction saved...
edit_type in test data: {0.0: 55270, 1.0: 61149, 2.0: 14176, 1.5: 5}
The cover rate of edit type in test data is 4/4.



add_reverse without_nochange
train reactions with 719050 reactions complete getting edits...
Removed 2359 reactions containing empty structures...
Removed 171615 reactions have no changes...
Removed 0 reactions contain aromatic bonds...
edits_length:  {2: 540977, 1: 349608, 4: 56384, 3: 140776, 7: 1317, 6: 4, 5: 1086}
1090152 reaction saved...
edit_type in train data: {0.0: 807490, 1.0: 1046470, 2.0: 239530, 1.5: 57, 3.0: 552}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 89081/89081 [03:46<00:00, 394.09it/s]
valid reactions with 89081 reactions complete getting edits...
Removed 297 reactions containing empty structures...
Removed 21290 reactions have no changes...
Removed 0 reactions contain aromatic bonds...
edits_length:  {2: 66816, 1: 42998, 4: 7068, 3: 17804, 7: 168, 5: 134}
134988 reaction saved...
edit_type in valid data: {0.0: 100470, 1.0: 130003, 2.0: 29610, 3.0: 77}
The cover rate of edit type in valid data is 4/5.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90024/90024 [03:46<00:00, 397.72it/s]
test reactions with 90024 reactions complete getting edits...
Removed 374 reactions containing empty structures...
Removed 21480 reactions have no changes...
Removed 0 reactions contain aromatic bonds...
edits_length:  {2: 67339, 1: 44250, 4: 7102, 3: 17348, 7: 155, 5: 146}
136340 reaction saved...
edit_type in test data: {0.0: 101063, 1.0: 130536, 2.0: 29532, 3.0: 59, 1.5: 5}
The cover rate of edit type in test data is 5/5.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134988/134988 [44:44<00:00, 50.28it/s]
valid dataset have saved 2813 batches.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 136340/136340 [46:04<00:00, 49.33it/s]
test dataset have saved 2841 batches.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1090152/1090152 [6:10:06<00:00, 49.09it/s]:
train dataset have saved 22712 batches.


grambow add_reverse
b97d3_selected
100%|█████████████████████████████████████| 9060/9060 [00:05<00:00, 1638.24it/s]
train reactions with 9060 reactions complete getting edits...
Removed 0 reactions containing empty structures...
Removed 0 reactions have no changes...
Removed 0 reactions contain aromatic bonds...
edits_length:  {6: 3862, 4: 8810, 3: 3436, 5: 834, 2: 954, 7: 82, 1: 46, 8: 90, 9: 6}
18120 reaction saved...
edit_type in train data: {0.0: 26285, 1.0: 35226, 2.0: 11465, 3.0: 1789, 1.5: 1427}
100%|█████████████████████████████████████| 1132/1132 [00:00<00:00, 1699.43it/s]
valid reactions with 1132 reactions complete getting edits...
Removed 0 reactions containing empty structures...
Removed 0 reactions have no changes...
Removed 0 reactions contain aromatic bonds...
edits_length:  {4: 1148, 6: 462, 3: 412, 2: 96, 5: 120, 7: 14, 1: 6, 8: 6}
2264 reaction saved...
edit_type in valid data: {0.0: 3279, 2.0: 1456, 1.0: 4416, 3.0: 233, 1.5: 160}
The cover rate of edit type in valid data is 5/5.
100%|█████████████████████████████████████| 1132/1132 [00:00<00:00, 1702.55it/s]
test reactions with 1132 reactions complete getting edits...
Removed 0 reactions containing empty structures...
Removed 0 reactions have no changes...
Removed 0 reactions contain aromatic bonds...
edits_length:  {4: 1142, 3: 374, 6: 494, 2: 110, 5: 120, 7: 12, 8: 6, 1: 6}
2264 reaction saved...
edit_type in test data: {0.0: 3285, 1.0: 4466, 2.0: 1476, 3.0: 211, 1.5: 174}
The cover rate of edit type in test data is 5/5.
100%|██████████████████████████████████████| 2264/2264 [00:07<00:00, 316.63it/s]
valid dataset have saved 48 batches.
100%|██████████████████████████████████████| 2264/2264 [00:07<00:00, 313.39it/s]
test dataset have saved 48 batches.
100%|████████████████████████████████████| 18120/18120 [00:58<00:00, 312.17it/s]
train dataset have saved 376 batches.
'''