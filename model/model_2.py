import torch
import torch.nn as nn
from rdkit import Chem
import sys
sys.path.append('..')

from model.model_layer import Encoder_MPNN, Encoder_Graphormer, Global_Attention, creat_edits_feats, unbatch_feats
from utils.process_utils import get_bond_features, get_atom_features
from utils.mol_operation import BondEditAction, get_bond_select_from_mol
def get_accuracy(scores, batch_split):
    batch_size = len(batch_split); edit_length = len(scores)

    bond_edit = 0; bond_match_edit = 0; graph_edit = 0; graph_match_edit = 0

    match_edit = 0.0; match_rxn = 0.0
    for data_idx, data_split in enumerate(batch_split):
        data_scores = [(score[0][data_idx], score[1][data_idx]) for score in scores]
        data_label = data_split.label
        for data_label_ in data_label: assert torch.sum(data_label_ == 1) == 1
        data_match_edit = 0
        for edit_idx in range(edit_length):
            if data_label[edit_idx][-1].item() == 0:
                bond_edit += 1
                if ((torch.argmax(data_label[edit_idx][:-1]).item() == torch.argmax(data_scores[edit_idx][0]).item())
                        and (torch.argmax(data_scores[edit_idx][1]).item() == 0)
                ):
                    match_edit += 1; data_match_edit += 1; bond_match_edit += 1
            elif data_label[edit_idx][-1].item() == 1:
                graph_edit += 1
                if torch.argmax(data_scores[edit_idx][1]).item() == 1:
                    match_edit += 1; data_match_edit += 1; graph_match_edit += 1
        if data_match_edit == edit_length: match_rxn += 1

    return match_rxn / batch_size, match_edit / (edit_length * batch_size), bond_match_edit / bond_edit, graph_match_edit / graph_edit

def get_loss(scores, batch_split, loss_fn, loss_ratio_step, ratio=0.6):
    edit_length = len(scores); assert edit_length == len(loss_ratio_step)
    device = batch_split[0].x[0].device
    loss_bond_stack = []; loss_graph_stack = []
    for data_idx, data_split in enumerate(batch_split):
        data_label = data_split.label
        data_scores = [(score[0][data_idx], score[1][data_idx])for score in scores]
        for data_label_ in data_label: assert torch.sum(data_label_ == 1) == 1
        for edit_idx in range(edit_length):
            data_label_edit = data_label[edit_idx]
            if data_label_edit[-1].item() == 0:
                loss_bond = loss_fn(data_scores[edit_idx][0].unsqueeze(0), torch.argmax(data_label_edit[:-1]).unsqueeze(0).long()).sum() * loss_ratio_step[edit_idx]
                loss_bond_stack.append(loss_bond)
            loss_graph = loss_fn(data_scores[edit_idx][1].unsqueeze(0), data_label_edit[-1].unsqueeze(0).long()).sum() * loss_ratio_step[edit_idx]
            loss_graph_stack.append(loss_graph)

    if len(loss_bond_stack) == 0:
        loss = (1 - ratio) * torch.stack(loss_graph_stack, dim=0).mean()
    else:
        loss = (ratio * torch.stack(loss_bond_stack, dim=0).mean() + (1 - ratio) * torch.stack(loss_graph_stack, dim=0).mean())
    return loss

def get_edit(num_atoms, indice, bond_vocab):
    edge_index_out = torch.triu_indices(num_atoms, num_atoms, offset=1)
    label_num = edge_index_out.size()[1] * len(bond_vocab) + 1
    if indice == label_num - 1:
        return None
    edge = indice // len(bond_vocab)
    return edge_index_out[:,edge].tolist()


def update_prelabel_from_label(edit_idx, prelabel, batch, bond_label):
    atom_prev = 0
    for idx, batch_split in enumerate(batch.to_data_list()):
        atom_num = batch_split.x[0].size()[0]
        label = batch_split.label[edit_idx]
        indice = torch.argmax(label, 0)
        ans = get_edit(atom_num, indice, bond_label)
        if ans is not None:
            prelabel[ans[0] + atom_prev] += 1
            prelabel[ans[1] + atom_prev] += 1
        atom_prev += batch_split.x[0].size()[0]
    return prelabel

def update_mol(mol, edit, device):

    edit_exe = BondEditAction(atom_map1=edit[0], atom_map2=edit[1], bond_type=edit[2], action_vocab='predict')
    mol_update = edit_exe.apply(Chem.Mol(mol))

    idx_to_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol_update.GetAtoms()}
    edge_index_start = []; edge_index_end = []
    f_bonds = []
    for bond in mol.GetBonds():
        atom1_map = idx_to_amap[bond.GetBeginAtomIdx()]
        atom2_map = idx_to_amap[bond.GetEndAtomIdx()]
        edge_index_start.append(atom1_map); edge_index_end.append(atom2_map)
        edge_index_start.append(atom2_map); edge_index_end.append(atom1_map)
        f_bond = get_bond_features(bond)
        f_bonds.append(f_bond)
        f_bonds.append(f_bond)
    f_bonds = torch.tensor(f_bonds, dtype=torch.float, device=device)
    edge_index = torch.tensor([edge_index_start, edge_index_end], dtype=torch.long, device=device)
    return mol_update, edge_index, f_bonds

def get_mol_features(mol, device):
    f_atoms = []; num_atoms = mol.GetNumAtoms()
    amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    idx_to_amap = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms()}

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

    for map_ in range(num_atoms):
        f_atom = get_atom_features(mol.GetAtoms()[amap_to_idx[map_]])
        f_atoms.append(f_atom)
    return torch.tensor(f_atoms, dtype=torch.float, device=device)


class RXN_Sequence(nn.Module):
    def __init__(self, config, bond_vocab, device):
        super(RXN_Sequence, self).__init__()
        self.config = config
        self.bond_vocab = bond_vocab
        self.device = device

        self.bond_outdim = len(bond_vocab)
        self.edit_length = 8
        self._build_layers()

    def _build_layers(self):
        config = self.config

        self.w_a = nn.Sequential(
            nn.Linear(config['n_atom_feat'], config['mpn_size'] * 2),
            nn.ReLU(),
            nn.Linear(config['mpn_size'] * 2, config['mpn_size'] - 1)
        )
        self.w_b = nn.Sequential(
            nn.Linear(config['n_bond_feat'], config['mpn_size'] * 2),
            nn.ReLU(),
            nn.Linear(config['mpn_size'] * 2, config['mpn_size'])
        )

        self.gru = nn.GRU(config['mpn_size'], config['mpn_size'], batch_first=True)
        self.encoder_mpnn  = Encoder_MPNN(config['mpn_size'], config['mpn_depth'], config['dropout_mpn'])
        self.encoder_graphormer = Encoder_Graphormer(config['mpn_size'], config['graphormer_n_heads'],
                                                     config['graphormer_depth'], config['dropout_mpn'])
        self.Attention = Global_Attention(d_model=config['mpn_size'], heads=config['attn_n_heads'],
                                          n_layers=config['attn_depth'], dropout=config['dropout_mpn'])

        self.w_a2 = nn.Sequential(
            nn.Linear(config['mpn_size'] * 2, config['mpn_size'] * 4),
            nn.ReLU(),
            nn.Linear(config['mpn_size'] * 4, config['mpn_size'])
        )
        self.w_a3 = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mpn_size'] * 2),
            nn.ReLU(),
            nn.Linear(config['mpn_size'] * 2, config['mpn_size'])
        )

        self.w_b_out = nn.Sequential(
            nn.Linear(config['mpn_size'] * 2, config['mlp_size'] * 2),
            nn.ReLU(),
            nn.Linear(config['mlp_size'] * 2, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], self.bond_outdim),
        )
        self.w_g_out = nn.Sequential(
            nn.Linear(config['mpn_size'], config['mlp_size'] * 2),
            nn.ReLU(),
            nn.Linear(config['mlp_size'] * 2, config['mlp_size']),
            nn.ReLU(),
            nn.Dropout(p=config['dropout_mlp']),
            nn.Linear(config['mlp_size'], 2)
        )

    def calc_model(self, batch):
        device = batch.x[0].device
        batch_split = batch.to_data_list()
        batch_size = len(batch_split)
        atom_out_list = [batch_split_.num_nodes for batch_split_ in batch_split]
        label_out_list = [batch_split_.edge_index_out.size()[1] * len(self.bond_vocab) + 1 for batch_split_ in batch_split]

        scores = []; prelabel = torch.zeros(batch.x[0].size()[0], 1).to(device)
        atom_feats_2 = torch.zeros(batch.x[0].size()[0], self.config['mpn_size']).to(device)

        for edit_idx in range(self.edit_length):
            atom_feats_1 = self.w_a(batch.x[edit_idx])  # (n_atoms, config['mpn_size'] - 1)
            bond_feats = self.w_b(batch.edge_attr[edit_idx])  # (n_bonds. config['mpn_size'])
            atom_feats_1 = torch.cat([atom_feats_1, prelabel], dim=1)

            atom_feats = torch.stack((atom_feats_1, atom_feats_2), dim=1)
            atom_feats, _ = self.gru(atom_feats)
            atom_feats = atom_feats[:, -1, :]

            edge_ptr = [0]
            for batch_split_ in batch_split:
                edge_ptr.append(batch_split_.edge_attr[edit_idx].size()[0] + edge_ptr[-1])

            atom_feats_gnn = self.encoder_mpnn(atom_feats, batch.edge_index[edit_idx], bond_feats, batch.ptr)   # (n_atoms, config['mpn_size'])
            atom_attn_pre, mask = creat_edits_feats(atom_feats, batch.ptr)
            _, atom_attn_aft = self.Attention(atom_attn_pre, mask)
            atom_feats_attn = unbatch_feats(atom_attn_aft, batch.ptr)

            atom_feats = self.w_a2(torch.cat([atom_feats_gnn, atom_feats_attn], dim=1)) # (n_atoms, config['mpn_size'])
            atom_feats, mask = creat_edits_feats(atom_feats, batch.ptr)
            atom_feats = self.encoder_graphormer(atom_feats, mask,
                                                 [batch_split_.edge_index[edit_idx] for batch_split_ in batch_split],
                                                 bond_feats, edge_ptr)
            atom_feats = unbatch_feats(atom_feats, batch.ptr)   # (n_atoms, config['mpn_size'])
            atom_feats_2 = self.w_a3(atom_feats) # (n_atoms, config['mpn_size'])

            bond_start = atom_feats[batch.edge_index_out[0,:]]
            bond_end   = atom_feats[batch.edge_index_out[1,:]]
            bond_feats_outs = torch.cat([bond_start, bond_end], dim=1)
            bond_outs  = self.w_b_out(bond_feats_outs).flatten()

            bond_scores = []; label_num = 0
            for label_out_list_ in label_out_list:
                bond_scores.append(bond_outs[label_num: label_num + label_out_list_ - 1])
                label_num += label_out_list_ - 1
            graph_outs = []; atom_num = 0
            for atom_out_list_ in atom_out_list:
                graph_outs.append(atom_feats[atom_num: atom_num + atom_out_list_].sum(dim=0))
                atom_num += atom_out_list_
            graph_outs = torch.stack(graph_outs).to(atom_feats.device)
            graph_outs = self.w_g_out(graph_outs)
            graph_scores = [graph_outs[i] for i in range(batch_size)]

            prelabel = update_prelabel_from_label(edit_idx, prelabel, batch, self.bond_vocab)
            scores.append((bond_scores, graph_scores))
        return scores

    def forward(self, batch, loss_fn):
        batch = batch.to(self.device)
        batch_split = batch.to_data_list()
        scores = self.calc_model(batch)
        assert len(scores) == self.edit_length

        acc = get_accuracy(scores, batch_split)
        loss = get_loss(scores, batch_split, loss_fn, ratio=self.config['loss_ratio'], loss_ratio_step=self.config['loss_ratio_step'])
        return loss, acc

    def predict(self, batch, mol, step_pred=8, edit_save=3):
        batch = batch.to(self.device)  # only one reaction to be predicted
        bond_vocab_reverse = {value: key for key, value in self.bond_vocab.items()}

        softmax = nn.Softmax(dim=0)
        num_atoms = batch.x.size()[0]


        bond_feats = batch.edge_attr
        bond_index = batch.edge_index

        results = []
        pre_model_test = [(mol, batch.x, bond_index, bond_feats,
                           torch.zeros(batch.x.size()[0], self.config['mpn_size']).to(self.device),
                           torch.zeros(batch.x.size()[0], 1).to(self.device), [])]

        for edit_idx in range(step_pred):

            pre_model_test_update = []
            for pred_idx, pre_model_test_ in enumerate(pre_model_test):
                int_mol, atom_feats_1, bond_index, bond_feats, atom_feats_2, prelabel, pre_edits = pre_model_test_

                #-------------------------------------------------------------
                atom_feats_1 = self.w_a(atom_feats_1)  # (n_atoms, config['mpn_size'] - 1)
                bond_feats = self.w_b(bond_feats)      # (n_bonds. config['mpn_size'])
                atom_feats_1 = torch.cat([atom_feats_1, prelabel], dim=1)

                atom_feats = torch.stack((atom_feats_1, atom_feats_2), dim=1)
                atom_feats, _ = self.gru(atom_feats)
                atom_feats = atom_feats[:, -1, :]

                edge_ptr = [0, bond_index.size()[1]]
                atom_feats_gnn = self.encoder_mpnn(atom_feats, bond_index, bond_feats, batch.ptr)  # (n_atoms, config['mpn_size'])
                atom_attn_pre, mask = creat_edits_feats(atom_feats, batch.ptr)
                _, atom_attn_aft = self.Attention(atom_attn_pre, mask)
                atom_feats_attn = unbatch_feats(atom_attn_aft, batch.ptr)

                atom_feats = self.w_a2(torch.cat([atom_feats_gnn, atom_feats_attn], dim=1))  # (n_atoms, config['mpn_size'])
                atom_feats, mask = creat_edits_feats(atom_feats, batch.ptr)
                atom_feats = self.encoder_graphormer(atom_feats, mask, [bond_index], bond_feats, edge_ptr)
                atom_feats = unbatch_feats(atom_feats, batch.ptr)  # (n_atoms, config['mpn_size'])
                atom_feats_2 = self.w_a3(atom_feats)  # (n_atoms, config['mpn_size'])

                bond_start = atom_feats[batch.edge_index_out[0, :]]
                bond_end = atom_feats[batch.edge_index_out[1, :]]
                bond_feats_outs = torch.cat([bond_start, bond_end], dim=1)
                bond_outs = self.w_b_out(bond_feats_outs).flatten()
                graph_outs = self.w_g_out(atom_feats.sum(dim=0))

                graph_outs_softmax = softmax(graph_outs)

                bond_select = get_bond_select_from_mol(int_mol, self.bond_vocab, self.device)
                edit_outs = torch.cat([softmax(bond_outs) * graph_outs_softmax[0].item(), graph_outs_softmax[1].unsqueeze(0)], dim=0) * bond_select
                edit_outs = edit_outs / torch.sum(edit_outs)

                # edit_outs = torch.cat([softmax(bond_outs) * graph_outs_softmax[0].item(), graph_outs_softmax[1].unsqueeze(0)], dim=0)
                #-------------------------------------------------------------

                indices = []; values = []
                value, indice = torch.topk(edit_outs, 1, dim=0)
                indices.append(indice[0].item()); values.append(value[0].item())
                values_, indices_ = torch.topk(edit_outs, edit_save, dim=0)
                for value, indice in zip(values_.tolist(), indices_.tolist()):
                    if indice not in indices:
                        indices.append(indice); values.append(value)

                values_sum = sum(values)
                for indice_idx, (value, indice) in enumerate(zip(values, indices)):
                    ans = get_edit(num_atoms, indice, self.bond_vocab)
                    prelabel_update = prelabel.clone()
                    value_update = value / values_sum if len(pre_edits) == 0 else value / values_sum * pre_edits[-1][3]
                    if ans is not None:
                        s, e = ans
                        prelabel_update[ans[0]] += 1
                        prelabel_update[ans[1]] += 1
                        action = bond_vocab_reverse[indice % len(self.bond_vocab)]
                        edit = (s, e, action, value_update)
                        mol_update, bond_index, bond_feats = update_mol(mol, edit, batch.x.device)
                        atom_feats_new = get_mol_features(mol_update, batch.x.device)
                        pre_model_test_update.append(
                            (mol_update, atom_feats_new, bond_index.clone(), bond_feats.clone(), atom_feats_2.clone(), prelabel_update, pre_edits + [edit])
                        )
                    else:
                        edit = (-1, -1, -1, value_update)
                        results.append(pre_edits + [edit])

            pre_model_test = sorted(pre_model_test_update, key=lambda x: x[-1][-1][3], reverse=True)[:min(10, len(pre_model_test_update))]
        results_noterminate = [data[-1] for data in pre_model_test]
        results_total = sorted(results + results_noterminate, key=lambda  x: x[-1][3], reverse=True)[:10]

        # results_total = sorted(results, key=lambda  x: x[-1][3], reverse=True)[:min(10, len(results))]

        return results_total