import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import add_self_loops, degree


def creat_edits_feats(atom_feats, atom_scope):
    a_feats = []
    masks = []
    for idx in range(atom_scope.size()[0]):
        if idx == 0: continue
        st_a = atom_scope[idx - 1].item()
        le_a = atom_scope[idx].item()
        feats = atom_feats[st_a: le_a]
        mask = torch.ones(feats.size(0), dtype=torch.uint8).to(atom_feats.device)
        a_feats.append(feats)
        masks.append(mask)
    a_feats = pad_sequence(a_feats, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    return a_feats, masks

def unbatch_feats(feats, atom_scope):
    atom_feats = []
    for idx in range(atom_scope.size()[0] - 1):
        le_a = atom_scope[idx + 1].item() - atom_scope[idx].item()
        atom_feats.append(feats[idx][:le_a])
    a_feats = torch.cat(atom_feats, dim=0).to(feats.device)
    return a_feats

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1)
            mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1, 1)
            scores[~mask.bool()] = float(-9e15)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return scores, output

    def forward(self, x, mask=None):
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores, output = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return scores, output.squeeze(-1)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        output = self.net(x)
        return self.layer_norm(x + output)


class Global_Attention(nn.Module):
    def __init__(self, d_model, heads, n_layers=1, dropout=0.1):
        super(Global_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)

    def forward(self, x, mask):
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x


# class Atom_Update_Layer_Attention(MessagePassing):
#     def __init__(self, node_in_channels, edge_in_channels, out_channels, attn_depth: int, dropout=0.15, heads=8):
#         super().__init__(aggr='add')
#         self.mlp = nn.Linear(node_in_channels + edge_in_channels, out_channels)
#         self.attn = Global_Attention(d_model=out_channels, heads=heads, n_layers=attn_depth, dropout=dropout)
#
#     def forward(self, x, edge_index, edge_attr, ptr):
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
#         out = self.mlp(out)
#
#         feats, mask = creat_edits_feats(out, ptr)
#         attention_score, feats = self.attn(feats, mask)
#         atom_feats = unbatch_feats(feats, ptr)
#         return atom_feats
#
#     def message(self, x_j ,edge_attr):
#         return torch.cat([x_j, edge_attr], dim=1)
#
#     def update(self, aggr_out):
#         return aggr_out


class Atom_Update_Layer(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, out_channels, dropout=0.15):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(node_in_channels * 2 + edge_in_channels, out_channels * 2),
            nn.GELU(),
            nn.Linear(out_channels * 2, out_channels),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr, ptr):
        row, col = edge_index
        deg = degree(col, x.size()[0], dtype=x.dtype)
        deg_inv_sqrt = deg.pow(0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)
        out = self.dropout(self.mlp(torch.cat([x,out], dim=1)))
        return out

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * torch.cat([x_j, edge_attr], dim=1)

    def update(self, aggr_out):
        return aggr_out


class Encoder_MPNN(nn.Module):
    def __init__(self, out_channels, depth, dropout):
        super(Encoder_MPNN, self).__init__()
        self.ModuleList = nn.ModuleList()
        for idx in range(depth):
            self.ModuleList.append(Atom_Update_Layer(out_channels, out_channels, out_channels, dropout))

    def forward(self, x, edge_index, edge_attr, ptr):
        for idx, module in enumerate(self.ModuleList):
            x = module(x, edge_index, edge_attr, ptr)
        return x


class Graphormer_Layer(nn.Module):
    def __init__(self, out_channels, dropout, n_heads):
        super(Graphormer_Layer, self).__init__()
        self.d_model = out_channels
        self.d_k = out_channels // n_heads
        self.h = n_heads
        self.q_linear = nn.Linear(out_channels + 2, out_channels, bias=False)
        self.k_linear = nn.Linear(out_channels + 2, out_channels, bias=False)
        self.v_linear = nn.Linear(out_channels + 2, out_channels, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.edge_feats_linear = nn.Linear(out_channels, out_channels)
        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, edge_index_list, edge_attr, edge_ptr):
        deg_list = []; mol_num = x.size(0); atom_num = x.size(1)
        edge_feats = self.edge_feats_linear(edge_attr).mean(dim=-1) # (n_bond, 1)
        edge_prev_num = 0; edge_bias = torch.zeros((mol_num, atom_num, atom_num), device=x.device)
        for mol_idx, edge_index in enumerate(edge_index_list):
            row, col = edge_index
            degree_row = degree(row, x.size()[1], dtype=x.dtype)
            degree_col = degree(col, x.size()[1], dtype=x.dtype)
            deg_list.append(torch.stack([degree_row, degree_col], dim=1)) # (n_atoms, )

            assert edge_index.size()[1] == edge_ptr[mol_idx + 1] - edge_ptr[mol_idx]
            edge_bias[mol_idx, row, col] = edge_feats[edge_ptr[mol_idx]: edge_ptr[mol_idx + 1]]


        deg_total = torch.stack(deg_list, dim=0)
        x_new = torch.cat([x, deg_total], dim=2)
        k = self.k_linear(x_new).view(mol_num, atom_num, self.h, self.d_k)
        q = self.q_linear(x_new).view(mol_num, atom_num, self.h, self.d_k)
        v = self.v_linear(x_new).view(mol_num, atom_num, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1)
            mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1, 1)
            scores[~mask.bool()] = float(-9e15)

        edge_bias = edge_bias.unsqueeze(1)
        # 需要给scores增加空间编码来构成graphomer
        scores = scores + edge_bias.repeat(1, self.h, 1, 1) # edge
        scores = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(mol_num, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return output


class Encoder_Graphormer(nn.Module):
    def __init__(self, out_channels, n_heads, depth, dropout):
        super(Encoder_Graphormer, self).__init__()
        self.depth = depth
        self.n_heads = n_heads
        self.graphormers = nn.ModuleList()
        self.feedforwards = nn.ModuleList()
        for idx in range(depth):
            self.graphormers.append(Graphormer_Layer(out_channels, dropout, self.n_heads))
            self.feedforwards.append(FeedForward(out_channels, dropout))

    def forward(self, x, mask, edge_index_list, edge_attr_list, ptr):
        for idx  in range(self.depth):
            x = self.graphormers[idx](x, mask, edge_index_list, edge_attr_list, ptr)
            x = self.feedforwards[idx](x)
        return x