import argparse
import os
import pickle
import joblib
import json, glob
from datetime import datetime as dt
import multiprocessing
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, lr_scheduler

from utils.process_utils import ATOM_FDIM, BOND_FDIM
from model.model import RXN_Sequence

DATE_TIME = dt.now().strftime('%d-%m-%Y--%H-%M-%S')

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def build_model_config(args):
    model_config = {}

    model_config['batch_size'] = args['batch_size']
    model_config['n_atom_feat'] = ATOM_FDIM
    model_config['n_bond_feat'] = BOND_FDIM

    model_config['mpn_size'] = args['mpn_size']
    model_config['mpn_depth'] = args['mpn_depth']
    model_config['dropout_mpn'] = args['dropout_mpn']
    model_config['attn_depth'] = args['attn_depth']
    model_config['dropout_attn'] = args['dropout_attn']
    model_config['attn_n_heads'] = args['attn_n_heads']
    model_config['graphormer_depth'] = args['graphormer_depth']
    model_config['dropout_graphormer'] = args['dropout_graphormer']
    model_config['graphormer_n_heads'] = args['graphormer_n_heads']
    model_config['mlp_size'] = args['mlp_size']
    model_config['dropout_mlp'] = args['dropout_mlp']
    model_config['loss_ratio'] = args['loss_ratio']
    model_config['loss_ratio_step'] = args['loss_ratio_step']

    return model_config

def save_checkpoint(model, path, epoch):
    save_dict = {'state': model.state_dict()}
    if hasattr(model, 'get_saveables'):
        save_dict['saveables'] = model.get_saveables()
    name = f'epoch_{epoch + 1}.pt'
    save_file = os.path.join(path, name)
    torch.save(save_dict, save_file)

def print_and_save(out_dir, str):
    print(str)
    with open(out_dir, 'a') as f:
        f.write(str + '\n')


def train_epoch(args, epoch, model, train_dataset, loss_fn, optimizer, out_path):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0.0; reaction_accuracy = 0.0; edit_accuracy = 0.0; bond_accuracy = 0.0; graph_accuracy = 0.0
    epoch_loss = 0.0; reaction_accuracy_epoch = 0.0; edit_accuracy_epoch = 0.0

    for batch_id, batch_path in enumerate(train_dataset):
        batch_data = torch.load(batch_path)
        loss, acc = model(batch_data, loss_fn)
        # acc[0] = reaction_accuracy; acc[1] = edit_accuracy; acc[2] = bond_accuracy; acc[3] = graph_accuracy
        total_loss += loss.item(); reaction_accuracy += acc[0]; edit_accuracy += acc[1]; bond_accuracy += acc[2]; graph_accuracy += acc[3]
        epoch_loss += loss.item(); reaction_accuracy_epoch += acc[0]; edit_accuracy_epoch += acc[1];

        if (batch_id + 1) % args['save_step'] == 0:
            average_loss = total_loss / args['save_step']
            average_reaction_acc  = reaction_accuracy / args['save_step']
            average_edit_acc = edit_accuracy / args['save_step']
            average_bond_accuracy = bond_accuracy / args['save_step']
            average_graph_accuracy = graph_accuracy / args['save_step']
            print_and_save(out_path,
                           f"epoch {epoch + 1} batch {batch_id + 1}/{len(train_dataset)} reaction_accuracy: {average_reaction_acc:.4f} train_accuracy: {average_edit_acc:.4f} train_loss: {average_loss:.4f} bond_accuracy: {average_bond_accuracy:.4f} graph_accuracy: {average_graph_accuracy:.4f}")
            total_loss = 0.0; reaction_accuracy = 0.0; edit_accuracy = 0.0; bond_accuracy = 0.0; graph_accuracy = 0.0

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()

    epoch_average_loss = float('%.4f' % (epoch_loss / len(train_dataset)))
    epoch_average_acc = float('%.4f' % (reaction_accuracy_epoch / len(train_dataset)))
    epoch_average_acc_edit = float('%.4f' % (edit_accuracy_epoch / len(train_dataset)))
    print_and_save(out_path,
                   f"epoch {epoch + 1} reaction_accuracy: {epoch_average_acc:.4f} train_accuracy: {epoch_average_acc_edit:.4f} train_loss: {epoch_average_loss:.4f}")
    print_and_save(out_path, f"epoch {epoch+1} Finished...")
    return epoch_average_loss, epoch_average_acc

def test_epoch(args, epoch, model, test_dataset, loss_fn, out_path):
    model.eval()
    total_loss = 0.0; reaction_accuracy = 0.0; edit_accuracy = 0.0
    with torch.no_grad():
        for batch_id, batch_path in enumerate(test_dataset):
            batch_data = torch.load(batch_path)
            loss, acc = model(batch_data, loss_fn)
            total_loss += loss.item()
            reaction_accuracy += acc[0]
            edit_accuracy += acc[1]

    average_loss = total_loss / len(test_dataset)
    average_acc  = reaction_accuracy / len(test_dataset)
    average_acc_edit = edit_accuracy / len(test_dataset)

    if epoch != 'test':
        print_and_save(out_path,
                       f"epoch {epoch + 1} reaction_accuracy: {average_acc:.4f} valid_accuracy: {average_acc_edit:.4f} valid_loss: {average_loss:.4f}")
    else:
        print_and_save(out_path, f'Final Test: reaction_accuracy: {average_acc:.4f} test_accuracy: {average_acc_edit:.4f} test_loss: {average_loss:.4f}')
    return average_loss, average_acc


def main(args):
    if args['resume']:
        out_dir = os.path.join('./experiments', args['experiment'])
        out_path = os.path.join(out_dir, 'output_resume')
        checkpoints_dir = os.path.join(out_dir, 'checkpoints_resume')
        os.makedirs(checkpoints_dir, exist_ok=True)
        json_path = os.path.join(out_dir, 'config.json')
        model_config = json.load(open(json_path, 'r'))
        model_config['n_atom_feat'] = ATOM_FDIM
        model_config['n_bond_feat'] = BOND_FDIM
    else:
        model_config = build_model_config(args)
        out_dir = os.path.join('./experiments', f"{DATE_TIME}")
        out_path = os.path.join(out_dir, 'output')
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, 'config.json')
        checkpoints_dir = os.path.join(out_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        with open(json_path, 'w') as file: json.dump(args, file)
        print_and_save(out_path, str(model_config))

    dataset = args['dataset']
    data_dir = os.path.join('./data', dataset)
    bond_vocab_dict = joblib.load(os.path.join(data_dir, 'bond_vocab.txt'))
    bond_vocab = {key: i for i, key in enumerate(bond_vocab_dict.keys())}

    train_data_path = os.path.join(data_dir, 'train')
    train_dataset = sorted([os.path.join(train_data_path, f) for f in os.listdir(train_data_path) if f.endswith('.pt')])
    valid_data_path = os.path.join(data_dir, 'valid')
    valid_dataset = sorted([os.path.join(valid_data_path, f) for f in os.listdir(valid_data_path) if f.endswith('.pt')])
    test_data_path = os.path.join(data_dir, 'test')
    test_dataset = sorted([os.path.join(test_data_path, f) for f in os.listdir(test_data_path) if f.endswith('.pt')])

    model = RXN_Sequence(config=model_config, bond_vocab=bond_vocab, device=args['device'])
    if args['resume']:
        pt_files = glob.glob(os.path.join(out_dir, 'checkpoints', '*.pt'))
        best_epoch = max([int(os.path.basename(pt_file).split('.')[0].split('_')[1]) for pt_file in pt_files])
        best_pt_file = os.path.join(out_dir, 'checkpoints', f'epoch_{best_epoch}.pt')
        checkpoint = torch.load(best_pt_file)
        model.load_state_dict(checkpoint['state'])
    print_and_save(out_path, f"Converting model to device: {args['device']}")
    model.to(args['device'])
    print_and_save(out_path, f"Param Count: {sum([x.nelement() for x in model.parameters()]) / 10 ** 6} M")
    print_and_save(out_path, '')

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    if args['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=args['lr'])
    elif args['optimizer'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args['lr'])
    else:
        raise ValueError('Optimizer can only be Adam or AdamW !!!')
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=args['patience'], factor=args['factor'], threshold=args['thresh'],
        threshold_mode='abs')

    counter_valid = 0
    best_acc = 0
    for epoch in range(args['epoches']):
        _, _ = train_epoch(args, epoch, model, train_dataset, loss_fn, optimizer, out_path)
        _, valid_acc = test_epoch(args, epoch, model, valid_dataset, loss_fn, out_path)
        scheduler.step(valid_acc)
        if valid_acc > best_acc:
            print_and_save(out_path, f'Saving Current Best Model from epoch {epoch+1} (acc = {valid_acc:.4f})')
            save_checkpoint(model, checkpoints_dir, epoch)
            best_acc = valid_acc
        else:
            counter_valid += 1
        if counter_valid >= args['valid_patience']:
            print_and_save(out_path, "Early stopping triggered. Training stopped.")
            break

    print_and_save(out_path, 'Experiment Finished!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coley/processing_FCB_False')
    # parser.add_argument('--edit_order', type=str, default='BCF', help='Order with edits generated: BCF(Break-Change-Form)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam/AdamW')
    parser.add_argument('--device', type=str, default='cuda:2', help='cuda:0/1/2')

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--experiment', type=str, default='20-10-2024--22-27-55')

    parser.add_argument('--batch_size', type=int, default=48, help='Number of Batch Size')
    parser.add_argument('--epoches', type=int, default=100, help='Maximum number of batches for training')
    parser.add_argument('--save_step', type=int, default=1, help='Step for saving model pt file')
    parser.add_argument('--valid_patience', type=int, default=10, help='Patience for stopping training if valid performs not good')

    parser.add_argument('--mpn_size', type=int, default=256,  help='MPN hidden_dim')
    parser.add_argument('--mpn_depth', type=int, default=3, help='Number of GNN iterations')
    parser.add_argument('--dropout_mpn', type=float, default=0.15, help='MPN dropout rate')
    parser.add_argument('--attn_depth', type=int, default=2, help='Number of Attention Layer')
    parser.add_argument('--dropout_attn', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--attn_n_heads', type=int, default=4, help='Number of heads in Multihead attention')
    parser.add_argument('--graphormer_depth', type=int, default=1, help='Number of Graphormer Layer')
    parser.add_argument('--dropout_graphormer', type=float, default=0.1, help='Graphormer dropout rate')
    parser.add_argument('--graphormer_n_heads', type=int, default=4, help='Number of heads in Graphormer')
    parser.add_argument('--mlp_size', type=int, default=512, help='MLP hidden_dim')
    parser.add_argument('--dropout_mlp', type=float, default=0.1, help='MLP dropout rate')
    parser.add_argument('--loss_ratio', type=float, default=0.6, help='ratio of loss bond and loss graph')
    parser.add_argument('--loss_ratio_step', type=list, default=[1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7],
                        help='[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] or [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no improvement after which lr will be reduced')
    parser.add_argument('--factor', type=float, default=0.1, help='Factor by which the lr will be reduced')
    parser.add_argument('--thresh', type=float, default=1e-4, help='Threshold for measuring the new optimum')
    parser.add_argument('--max_clip', type=float, default=1, help='Maximum number of gradient clip')

    parser.add_argument('--num_processes', type=int, default=24)
    parser.add_argument('--step_cut', type=int, default=8, help='max edit step for a elementary reaction')
    parser.add_argument('--tree_child_num', type=int, default=3, help="only work when test_method = 'tree', best option = 2/3")
    args = parser.parse_args().__dict__
    main(args)


