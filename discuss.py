import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import time
import os

from logger import Logger_classify, Logger_detect
from data_utils import normalize, gen_normalized_adjs, evaluate_classify, evaluate_detect, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, rand_splits, get_gpu_memory_map, count_parameters
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
parser = argparse.ArgumentParser(description='Discussion Pipeline (Hyper-parameter, Efficiency, Visualization)')
parser_add_main_args(parser)
parser.add_argument('--dis_type', type=str, default='margin', choices=['margin', 'lamda', 'prop', 'backbone', 'time', 'vis_energy', 'alpha_inter'])

# 🟢 强制锁定 method 选项只有 emp
for action in parser._actions:
    if action.dest == 'method':
        action.choices = ['emp']

args = parser.parse_args()
print(args)

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

num_relations = 1
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

val_loss_min = 100.
train_time = 0

### Training loop ###
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        train_start = time.time()
        loss = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        loss.backward()
        optimizer.step()
        train_time += time.time() - train_start

        if args.mode == 'classify':
            result = evaluate_classify(model, dataset_ind, eval_func, criterion, args, device)
            logger.add_result(run, result)
        else:
            result, test_in_score, test_ood_score = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device, return_score=True)
            logger.add_result(run, result)

            if result[-1] < val_loss_min:
                val_loss_min = result[-1]
                in_score, ood_score = test_in_score, test_ood_score

    logger.print_statistics(run)

results = logger.print_statistics()
if args.dis_type == 'time':
    infer_start = time.time()
    test_ind_score = model.detect(dataset_ind, dataset_ind.splits['test'], device, args)
    infer_time = time.time() - infer_start

### Save results ###
if not os.path.exists(f'results/discuss'):
    os.makedirs(f'results/discuss')

if args.dis_type == 'vis_energy':
    filename = 'results/vis_scores/emp.csv'
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"{in_score.shape[0]} {ood_score.shape[0]}\n")
        for i in range(in_score.shape[0]):
            write_obj.write(f"{in_score[i]}\n")
        for i in range(ood_score.shape[0]):
            write_obj.write(f"{ood_score[i]}\n")
else:
    filename = f'results/discuss/{args.dis_type}.csv'
    with open(f"{filename}", 'a+') as write_obj:
        if args.dis_type == 'time':
            write_obj.write(f"{args.method} {args.dataset} {args.ood_type} {train_time} {infer_time}\n")
        else:
            if args.dis_type == 'margin':
                write_obj.write(f"{args.dataset} {args.ood_type} {args.m_in} {args.m_out}\n")
            elif args.dis_type == 'prop':
                write_obj.write(f"{args.dataset} {args.ood_type} {args.K} {args.alpha}\n")
            elif args.dis_type == 'alpha_inter':
                write_obj.write(f"{args.dataset} {args.ood_type} {args.K} {args.alpha_inter}\n")
            elif args.dis_type == 'lamda':
                write_obj.write(f"{args.dataset} {args.ood_type} {args.lamda}\n")
            elif args.dis_type == 'backbone':
                write_obj.write(f"{args.dataset} {args.ood_type} {args.backbone}\n")
            for k in range(results.shape[1] // 3):
                r = results[:, k * 3]
                write_obj.write(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f} ± {r.std():.2f} ')
                r = results[:, k * 3 + 1]
                write_obj.write(f'OOD Test {k + 1} Final AUPR: {r.mean():.2f} ± {r.std():.2f} ')
                r = results[:, k * 3 + 2]
                write_obj.write(f'OOD Test {k + 1} Final FPR: {r.mean():.2f} ± {r.std():.2f}\n')
            r = results[:, -1]
            write_obj.write(f'IND Test Score: {r.mean():.2f} ± {r.std():.2f}\n')