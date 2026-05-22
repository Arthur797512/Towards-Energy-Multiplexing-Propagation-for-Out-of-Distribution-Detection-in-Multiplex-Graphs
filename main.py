import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from logger import Logger_classify, Logger_detect, save_result
from data_utils import evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, rand_splits
from dataset import load_dataset
from parse import parser_add_main_args
from emp import EMP  # 🚀 极致纯净：只引入 EMP！


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
parser = argparse.ArgumentParser(description='EMP General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

device = torch.device("cpu") if args.cpu else torch.device(
    "cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

### Load and preprocess data ###
dataset_ind, dataset_ood_tr, dataset_ood_te = load_dataset(args)

# 标签维度修正
if len(dataset_ind.y.shape) == 1: dataset_ind.y = dataset_ind.y.unsqueeze(1)
if len(dataset_ood_tr.y.shape) == 1: dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
if isinstance(dataset_ood_te, list):
    for data in dataset_ood_te:
        if len(data.y.shape) == 1: data.y = data.y.unsqueeze(1)
else:
    if len(dataset_ood_te.y.shape) == 1: dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

if args.dataset not in ['cora', 'citeseer', 'pubmed']:
    dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.x.shape[1]


# -----------------------------------------------------------------------
# [标准化] 确保所有数据为 Multiplex 格式
# -----------------------------------------------------------------------
def ensure_multiplex_format(data):
    if not hasattr(data, 'edge_indices'):
        data.edge_indices = [data.edge_index]
    return data


if not args.single:
    dataset_ind = ensure_multiplex_format(dataset_ind)
    dataset_ood_tr = ensure_multiplex_format(dataset_ood_tr)
    if isinstance(dataset_ood_te, list):
        dataset_ood_te = [ensure_multiplex_format(d) for d in dataset_ood_te]
    else:
        dataset_ood_te = ensure_multiplex_format(dataset_ood_te)

# -----------------------------------------------------------------------
# [核心修改] 单图模式处理逻辑 (Merge Multiplex -> Single)
# -----------------------------------------------------------------------
if args.single:
    print("\n" + "=" * 40)
    print("⚠️  [SINGLE GRAPH MODE] 合并多路图...")
    print("=" * 40 + "\n")


    def merge_to_single(data):
        if hasattr(data, 'edge_indices') and data.edge_indices is not None and len(data.edge_indices) > 0:
            print(f"   Merging {len(data.edge_indices)} relations into one edge_index...")
            data.edge_index = torch.cat(data.edge_indices, dim=1)
            del data.edge_indices
        return data


    dataset_ind = merge_to_single(dataset_ind)
    dataset_ood_tr = merge_to_single(dataset_ood_tr)
    if isinstance(dataset_ood_te, list):
        dataset_ood_te = [merge_to_single(d) for d in dataset_ood_te]
    else:
        dataset_ood_te = merge_to_single(dataset_ood_te)

# -----------------------------------------------------------------------
# 数据迁移到 GPU
# -----------------------------------------------------------------------
print("Moving datasets to GPU for speed optimization...")


def move_to_device(data, device):
    if data.x is not None: data.x = data.x.to(device)
    if data.y is not None: data.y = data.y.to(device)
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
    dataset_ood_te = [move_to_device(d, device) for d in dataset_ood_te]
else:
    dataset_ood_te = move_to_device(dataset_ood_te, device)

num_relations = len(dataset_ind.edge_indices) if hasattr(dataset_ind, 'edge_indices') else 1
print(f"Detected Graph structure with {num_relations} layer(s).")

### Load method ###
if args.method == 'emp':
    print(f"🚀 Initializing Official EMP Model for {num_relations} relation(s)...")
    model = EMP(d, c, args, num_relations).to(device)
else:
    raise ValueError(f"Method '{args.method}' is not supported in this pure EMP repository.")

criterion = nn.BCEWithLogitsLoss() if args.dataset in ('proteins', 'ppi') else nn.NLLLoss()
eval_func = eval_rocauc if args.dataset in ('proteins', 'ppi', 'twitch') else eval_acc
logger = Logger_classify(args.runs, args) if args.mode == 'classify' else Logger_detect(args.runs, args)

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
                print(
                    f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}%')
            else:
                result = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device)
                current_fpr = result[2]

                if best_test_results is None or current_fpr < best_test_results[2]:
                    best_test_results = result
                    if epoch > 0:
                        print(f"   [New Best] Epoch {epoch}: FPR95 updated to {100 * current_fpr:.2f}%")

                logger.add_result(run, result)
                print(
                    f'Epoch: {epoch:02d}, Loss: {loss:.4f}, AUROC: {100 * result[0]:.2f}%, AUPR: {100 * result[1]:.2f}%, FPR95: {100 * result[2]:.2f}%, Test Score: {100 * result[-2]:.2f}%')

    if best_test_results is not None:
        print(f"Run {run:02d} Best FPR95: {100 * best_test_results[2]:.2f}%")

    logger.print_statistics(run)

results = logger.print_statistics()

if args.mode == 'detect':
    save_result(results, args)