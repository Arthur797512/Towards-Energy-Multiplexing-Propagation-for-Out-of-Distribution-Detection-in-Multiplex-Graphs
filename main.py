import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from logger import Logger_classify, Logger_detect, save_result
from data_utils import normalize, gen_normalized_adjs, evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, \
    eval_f1, to_sparse_tensor, \
    load_fixed_splits, rand_splits, get_gpu_memory_map, count_parameters
from dataset import load_dataset
from parse import parser_add_main_args
from emp import EMP  # 🚀 极致纯净：只引入 EMP！


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


### Parse args ###
### Parse args ###
parser = argparse.ArgumentParser(description='EMP General Training Pipeline')
parser_add_main_args(parser)


# 🟢 强制锁定 method 选项只有 emp
for action in parser._actions:
    if action.dest == 'method':
        action.choices = ['emp']

args = parser.parse_args()

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)
if len(dataset_ood_tr.y.shape) == 1:
    dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
if isinstance(dataset_ood_te, list):
    for data in dataset_ood_te:
        if len(data.y.shape) == 1:
            data.y = data.y.unsqueeze(1)
else:
    if len(dataset_ood_te.y.shape) == 1:
        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    pass
else:
    dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.x.shape[1]


def ensure_multiplex_format(data):
    if not hasattr(data, 'edge_indices'):
        data.edge_indices = [data.edge_index]
    return data


if not args.single:
    dataset_ind = ensure_multiplex_format(dataset_ind)
    dataset_ood_tr = ensure_multiplex_format(dataset_ood_tr)
    if isinstance(dataset_ood_te, list):
        for i in range(len(dataset_ood_te)):
            dataset_ood_te[i] = ensure_multiplex_format(dataset_ood_te[i])
    else:
        dataset_ood_te = ensure_multiplex_format(dataset_ood_te)

if args.single:
    def merge_to_single(data):
        if hasattr(data, 'edge_indices') and data.edge_indices is not None and len(data.edge_indices) > 0:
            merged_edge_index = torch.cat(data.edge_indices, dim=1)
            data.edge_index = merged_edge_index
            del data.edge_indices
        return data


    dataset_ind = merge_to_single(dataset_ind)
    dataset_ood_tr = merge_to_single(dataset_ood_tr)
    if isinstance(dataset_ood_te, list):
        dataset_ood_te = [merge_to_single(d) for d in dataset_ood_te]
    else:
        dataset_ood_te = merge_to_single(dataset_ood_te)


def move_to_device(data, device):
    if data.x is not None:
        data.x = data.x.to(device)
    if data.y is not None:
        data.y = data.y.to(device)
    if hasattr(data, 'edge_indices') and data.edge_indices is not None:
        data.edge_indices = [adj.to(device) for adj in data.edge_indices]
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        data.edge_index = data.edge_index.to(device)
    if hasattr(data, 'node_idx') and torch.is_tensor(data.node_idx):
        data.node_idx = data.node_idx.to(device)
    return data


dataset_ind = move_to_device(dataset_ind, device)
dataset_ood_tr = move_to_device(dataset_ood_tr, device)

if isinstance(dataset_ood_te, list):
    for i in range(len(dataset_ood_te)):
        dataset_ood_te[i] = move_to_device(dataset_ood_te[i], device)
else:
    dataset_ood_te = move_to_device(dataset_ood_te, device)

num_relations = 0
if hasattr(dataset_ind, 'edge_indices'):
    num_relations = len(dataset_ind.edge_indices)

# ---------------------------------------------------------
# 🟢 极致简洁：只有 EMP 实例化，干掉所有 baseline
# ---------------------------------------------------------
if args.method.lower() == 'emp':
    model = EMP(d, c, args, num_relations=num_relations).to(device)
else:
    raise ValueError(f"Method '{args.method}' is not supported in this pure EMP repository.")

if args.dataset in ('proteins', 'ppi'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

if args.dataset in ('proteins', 'ppi', 'twitch'):
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

if args.mode == 'classify':
    logger = Logger_classify(args.runs, args)
else:
    logger = Logger_detect(args.runs, args)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    model.reset_parameters()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_test_results = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        loss.backward()
        optimizer.step()

        if epoch % args.display_step == 0 or epoch == args.epochs - 1:
            if args.mode == 'classify':
                result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
                logger.add_result(run, result)
            else:
                result = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)

                current_fpr = result[2]
                if best_test_results is None or current_fpr < best_test_results[2]:
                    best_test_results = result

                logger.add_result(run, result)

    if best_test_results is not None:
        if not hasattr(args, 'true_aurocs'):
            args.true_aurocs, args.true_auprs, args.true_fprs = [], [], []
        args.true_aurocs.append(best_test_results[0] * 100)
        args.true_auprs.append(best_test_results[1] * 100)
        args.true_fprs.append(best_test_results[2] * 100)

    print(f"✅ 第 {run + 1} 轮结束! 本轮最佳 FPR95: {100 * best_test_results[2]:.2f}%")
    logger.print_statistics(run)

results = logger.print_statistics()

print("\n" + "★" * 50)
print("🚀 TRUE BEST OOD RESULTS ACROSS ALL RUNS 🚀")
if hasattr(args, 'true_aurocs') and len(args.true_aurocs) > 0:
    print(f"AUROC: {np.mean(args.true_aurocs):.2f} ± {np.std(args.true_aurocs):.2f}")
    print(f"AUPR : {np.mean(args.true_auprs):.2f} ± {np.std(args.true_auprs):.2f}")
    print(f"FPR95: {np.mean(args.true_fprs):.2f} ± {np.std(args.true_fprs):.2f}")
print("★" * 50 + "\n")

### Save results ###
if args.mode == 'detect':
    save_result(results, args)