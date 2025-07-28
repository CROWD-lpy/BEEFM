import os, argparse, json, joblib
import time
import multiprocessing as mp
from subprocess import Popen, PIPE
import shutil
from tqdm import tqdm

os.chdir('/raid/home/liangpengyu/romote_project/Elementary_reaction/graph_20240926')
python = '/raid/home/liangpengyu/miniconda3/envs/learnts-copy/bin/python'

def func(data, gpu_id, experiment, step_cut, tree_child_num):
    gpu = f'cuda:{gpu_id}'
    my_env = os.environ.copy()
    process = Popen([python, 'utils/predict_utils.py',
                     '--experiment', experiment,
                     '--data_file', data,
                     '--device', gpu,
                     '--step_cut', str(step_cut),
                     '--tree_child_num', str(tree_child_num)],
                    env=my_env,
                    stdout=PIPE,
                    stderr=PIPE)

    stdout, stderr = process.communicate()
    print(str(stdout), str(stderr))
    return stdout


def worker(data_queue, gpu_id, experiment, step_cut, tree_child_num):
    while not data_queue.empty():
        data = data_queue.get()
        func(data, gpu_id, experiment, step_cut, tree_child_num)

def print_and_save(out_dir, str):
    print(str)
    with open(out_dir, 'a') as f:
        f.write(str + '\n')


def main(args):
    out_dir = os.path.join('./experiments', args['experiment'])
    json_path = os.path.join(out_dir, 'config.json')
    config = json.load(open(json_path, 'r'))
    dataset = config['dataset'] if args['data_file'] is None else args['data_file']
    data_path = f"data/{dataset}/test.file"
    test_data = joblib.load(data_path)
    print(f'Testset data has {len(test_data)} reactions...')
    out_file = os.path.join(out_dir, 'test.result')

    os.makedirs('temp', exist_ok=True)
    avg_size = len(test_data) // args['num_processes']
    remainder = len(test_data) % args['num_processes']
    start = 0; file_list = []
    for i in range(args['num_processes']):
        end = start + avg_size + (0 if i != args['num_processes'] - 1 else remainder)
        joblib.dump(test_data[start: end], f'temp/test_{i:03d}.file')
        file_list.append(f'../temp/test_{i:03d}.file')
        start = end
    print('Successfully split test.file')

    data_queue = mp.Queue()
    for file in file_list:
        data_queue.put(file)
    processes = []; gpu_available = [int(id) for id in args['device_available'].split('/')]
    gpu_process = []
    for gpu_id in gpu_available:
        gpu_process += [gpu_id] * (args['num_processes'] // len(gpu_available))

    for gpu_id in gpu_process:
        p = mp.Process(target=worker, args=(data_queue, gpu_id, args['experiment'], args['step_cut'], args['tree_child_num']))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    pred_files = sorted([os.path.join('temp', f) for f in os.listdir('temp') if f.endswith('_pred.file')])
    pred_rxns = []
    for pred_file in tqdm(pred_files):
        pred_rxns += joblib.load(pred_file)

    accuracy_topk = {}
    topk_candidate = [1, 2, 3, 5, 10]
    for topk_candidate_ in topk_candidate:
        right_num = sum([1 for pred_rxns_idx in pred_rxns if 0 <= pred_rxns_idx[1] <= (topk_candidate_ - 1)])
        accuracy = right_num / len(test_data)
        accuracy_topk[topk_candidate_] = accuracy
    print_and_save(out_file, 'Test by tree structure:')
    for k in accuracy_topk.keys():
        print_and_save(out_file, f'Top {k} Prediction Accuracy: {accuracy_topk[k]:.4f}')

    shutil.rmtree('temp')
    with open(os.path.join(out_dir, 'statistic.result'), 'a') as file:
        for data_idx, data in enumerate(tqdm(test_data)):
            file.write(f'{data_idx} {len(data[1])} {pred_rxns[data_idx][1]}\n')
    joblib.dump(pred_rxns, os.path.join(out_dir, "pred.file"))

    print('Temp Files Have Removed.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='15-07-2025--10-40-22', help='26-09-2024--17-14-04')
    parser.add_argument('--data_file', type=str, default=None, help='coley/processing_FCB_False')

    parser.add_argument('--device_available', type=str, default='0/1/2')
    parser.add_argument('--num_processes', type=int, default=36)

    parser.add_argument('--step_cut', type=int, default=8, help='max edit step for a elementary reaction')
    parser.add_argument('--tree_child_num', type=int, default=3, help="only work when test_method = 'tree', best option = 2/3")
    args = parser.parse_args().__dict__
    main(args)

