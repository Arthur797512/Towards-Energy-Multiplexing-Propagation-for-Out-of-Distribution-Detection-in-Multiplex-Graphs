import os
from os import path
import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
import torch_geometric.transforms as T
import torch_geometric.utils as utils
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, Entities, MovieLens
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from data_utils import rand_splits


# ===================================================================
# 1. AIFB Multiplex Loader
# ===================================================================
def load_aifb_multiplex(data_dir, ood_type):
    save_dir = os.path.join(data_dir, 'AIFB')
    print("[INFO] Downloading and preprocessing AIFB Heterogeneous Graph...")
    dataset = Entities(root=save_dir, name='AIFB')
    data = dataset[0]
    num_nodes = data.num_nodes

    dataset_x = torch.eye(num_nodes, dtype=torch.float)
    dataset_y = torch.full((num_nodes, 1), -1, dtype=torch.long)
    dataset_y[data.train_idx, 0] = data.train_y
    dataset_y[data.test_idx, 0] = data.test_y

    mask_view1 = data.edge_type < 45
    mask_view2 = data.edge_type >= 45
    view1_edge_index = data.edge_index[:, mask_view1]
    view2_edge_index = data.edge_index[:, mask_view2]

    dataset_edge_indices = [view1_edge_index, view2_edge_index]
    print(f"   -> View 1 (Internal) edges: {view1_edge_index.size(1)}")
    print(f"   -> View 2 (External) edges: {view2_edge_index.size(1)}")

    labeled_mask = (dataset_y.squeeze() != -1)
    valid_node_idx = torch.arange(num_nodes)[labeled_mask]

    dataset_ind = Data(x=dataset_x, y=dataset_y)
    dataset_ind.edge_indices = dataset_edge_indices
    dataset_ind.edge_index = view1_edge_index

    if ood_type == 'label':
        class_t = 3
        mask_ind = (dataset_y.squeeze() < class_t) & labeled_mask
        dataset_ind.node_idx = torch.arange(num_nodes)[mask_ind]
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=.6, valid_prop=.2)

        mask_ood = (dataset_y.squeeze() == class_t) & labeled_mask
        ood_nodes = torch.arange(num_nodes)[mask_ood]

        perm = torch.randperm(ood_nodes.size(0))
        ood_nodes = ood_nodes[perm]
        split_point = ood_nodes.size(0) // 2

        dataset_ood_tr = Data(x=dataset_x, y=dataset_y)
        dataset_ood_tr.edge_indices = dataset_edge_indices
        dataset_ood_tr.node_idx = ood_nodes[:split_point]

        dataset_ood_te = Data(x=dataset_x, y=dataset_y)
        dataset_ood_te.edge_indices = dataset_edge_indices
        dataset_ood_te.node_idx = ood_nodes[split_point:]
        print(f"[INFO] AIFB Label Shift: ID Nodes={dataset_ind.node_idx.size(0)}, OOD={ood_nodes.size(0)}")
    else:
        dataset_ind.node_idx = valid_node_idx
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=.6, valid_prop=.2)
        if ood_type == 'structure':
            dataset_ood_tr = create_multiplex_structure_noise_dataset(dataset_ind)
            dataset_ood_te = create_multiplex_structure_noise_dataset(dataset_ind)
        else:
            dataset_ood_tr = create_multiplex_feat_noise_dataset(dataset_ind)
            dataset_ood_te = create_multiplex_feat_noise_dataset(dataset_ind)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


# ===================================================================
# 2. MovieLens Multiplex Loader
# ===================================================================
def load_movielens_dataset(data_dir, ood_type):
    save_dir = os.path.join(data_dir, 'MovieLens')
    print("[INFO] Downloading PyG MovieLens (ml-latest-small)...")
    torch_dataset = MovieLens(root=save_dir)
    data = torch_dataset[0]

    num_movies = data['movie'].num_nodes
    dataset = Data()

    dataset.x = data['movie'].x[:, :384]
    genre_matrix = data['movie'].x[:, 384:]
    dataset.y = genre_matrix.argmax(dim=1)

    # View 1: User overlap
    rates_edge = data['user', 'rates', 'movie'].edge_index
    adj_mu = SparseTensor(row=rates_edge[1], col=rates_edge[0], value=torch.ones(rates_edge.size(1)),
                          sparse_sizes=(num_movies, data['user'].num_nodes))
    adj_view1 = adj_mu.matmul(adj_mu.t())
    row1, col1, val1 = adj_view1.coo()
    mask1 = (row1 != col1) & (val1 >= 4)
    view1_edge_index = torch.stack([row1[mask1], col1[mask1]], dim=0)
    print(f"   -> View 1 (User overlap) edges: {view1_edge_index.size(1)}")

    # View 2: Genre overlap
    genre_edge = genre_matrix.nonzero(as_tuple=False).t()
    adj_mg = SparseTensor(row=genre_edge[0], col=genre_edge[1], value=torch.ones(genre_edge.size(1)),
                          sparse_sizes=(num_movies, genre_matrix.size(1)))
    adj_view2 = adj_mg.matmul(adj_mg.t())
    row2, col2, val2 = adj_view2.coo()
    mask2 = (row2 != col2) & (val2 >= 2)
    view2_edge_index = torch.stack([row2[mask2], col2[mask2]], dim=0)
    print(f"   -> View 2 (Genre overlap) edges: {view2_edge_index.size(1)}")

    dataset.edge_index = view1_edge_index
    dataset.edge_indices = [view1_edge_index, view2_edge_index]
    full_node_idx = torch.arange(num_movies)
    dataset.node_idx = full_node_idx

    dataset_ind = Data(x=dataset.x, y=dataset.y)
    dataset_ind.edge_indices = dataset.edge_indices
    dataset_ind.edge_index = dataset.edge_index

    if ood_type == 'label':
        num_classes = genre_matrix.size(1)
        num_ood_genres = 3
        ood_genres = list(range(num_classes - num_ood_genres, num_classes))

        ood_scores = genre_matrix[:, ood_genres].sum(dim=1)
        mask_ood = (ood_scores > 0)
        mask_ind = (ood_scores == 0)

        id_genre_matrix = genre_matrix.clone()
        id_genre_matrix[:, ood_genres] = 0
        dataset.y = id_genre_matrix.argmax(dim=1)

        dataset_ind.node_idx = full_node_idx[mask_ind]
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=.6, valid_prop=.2)

        ood_node_idx = full_node_idx[mask_ood]
        dataset_ood_tr = Data(x=dataset.x, y=dataset.y)
        dataset_ood_tr.edge_indices = dataset.edge_indices
        dataset_ood_te = Data(x=dataset.x, y=dataset.y)
        dataset_ood_te.edge_indices = dataset.edge_indices

        perm = torch.randperm(ood_node_idx.size(0))
        ood_node_idx = ood_node_idx[perm]
        split_point = ood_node_idx.size(0) // 2

        dataset_ood_tr.node_idx = ood_node_idx[:split_point]
        dataset_ood_te.node_idx = ood_node_idx[split_point:]
        print(f"[INFO] MovieLens Label Shift: ID Nodes={dataset_ind.node_idx.size(0)}, OOD={ood_node_idx.size(0)}")
    else:
        dataset_ind.node_idx = full_node_idx
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=.6, valid_prop=.2)
        if ood_type == 'structure':
            dataset_ood_tr = create_multiplex_structure_noise_dataset(dataset)
            dataset_ood_te = create_multiplex_structure_noise_dataset(dataset)
        else:
            dataset_ood_tr = create_multiplex_feat_noise_dataset(dataset)
            dataset_ood_te = create_multiplex_feat_noise_dataset(dataset)

    return dataset_ind, dataset_ood_tr, dataset_ood_te


# ===================================================================
# 3. Real Multiplex (.mat) Loader (ACM, DBLP, IMDB, AMAZON)
# ===================================================================
def load_real_multiplex_dataset(data_dir, dataname, ood_type):
    dataname = dataname.lower()
    paths_to_try = [
        f"{data_dir}/{dataname.upper()}/{dataname}.mat",
        f"{data_dir}/{dataname}.mat"
    ]

    raw_data = None
    for p in paths_to_try:
        if path.exists(p):
            try:
                raw_data = sio.loadmat(p)
                print(f"[INFO] Loaded multiplex dataset from {p}")
                if 'imdb' in raw_data:
                    struct = raw_data['imdb'][0, 0]
                    for key in struct.dtype.names: raw_data[key] = struct[key]
                break
            except Exception:
                continue
    if raw_data is None: raise FileNotFoundError(f"Could not find {dataname}.mat")

    def get_key(data, keys):
        for k in keys:
            if k in data: return data[k]
        return None

    x = get_key(raw_data, ['feature', 'features', 'X', 'TvsP', 'PvsT'])
    if x is None: x = sp.eye(1)
    if sp.issparse(x): x = x.todense()
    x = torch.FloatTensor(np.asarray(x))

    y = get_key(raw_data, ['label', 'labels', 'gnd', 'PvsL', 'Y'])
    if sp.issparse(y): y = y.todense()
    y = torch.LongTensor(np.asarray(y))

    if dataname == 'imdb':
        if y.dim() > 1 and y.shape[0] < y.shape[1]: y = y.t()
        if x.shape[0] != y.shape[0] and x.shape[1] == y.shape[0]: x = x.t()
        if x.shape[0] != y.shape[0]: x = torch.eye(y.shape[0])
        row_sum = x.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        x = x / row_sum

    if y.dim() > 1 and y.shape[1] > 1: y = torch.argmax(y, dim=1)
    if y.dim() == 1: y = y.unsqueeze(1)

    edge_indices = []
    # 正确的一行：
    possible_adjs = ['PLP', 'PAP', 'PSP', 'MAM', 'MDM', 'MGM', 'MKM', 'APA', 'APCPA', 'APTPA', 'net_APA', 'net_APCPA',
                     'net_APTPA']
    for key in possible_adjs:
        if key in raw_data: edge_indices.append(raw_data[key])

    if len(edge_indices) == 0 and dataname == 'acm':
        if 'PvsA' in raw_data:
            p_vs_a = sp.csr_matrix(raw_data['PvsA']) if not sp.issparse(raw_data['PvsA']) else raw_data['PvsA']
            edge_indices.append(p_vs_a @ p_vs_a.T)
        if 'PvsC' in raw_data:
            p_vs_c = sp.csr_matrix(raw_data['PvsC']) if not sp.issparse(raw_data['PvsC']) else raw_data['PvsC']
            edge_indices.append(p_vs_c @ p_vs_c.T)

    tensor_edge_indices = []
    for adj in edge_indices:
        if not sp.issparse(adj): adj = sp.csr_matrix(adj)
        adj.setdiag(0)
        adj.eliminate_zeros()
        adj_coo = adj.tocoo()
        row, col = torch.from_numpy(adj_coo.row), torch.from_numpy(adj_coo.col)
        tensor_edge_indices.append(torch.stack([row, col], dim=0).long())

    dataset = Data(x=x, y=y)
    dataset.edge_indices = tensor_edge_indices
    full_node_idx = torch.arange(x.shape[0])
    dataset.node_idx = full_node_idx

    train_idx = get_key(raw_data, ['train_idx'])
    if train_idx is not None:
        def to_idx(idx):
            if sp.issparse(idx): idx = idx.todense()
            return torch.LongTensor(np.asarray(idx).squeeze())

        dataset.splits = {'train': to_idx(train_idx), 'valid': to_idx(get_key(raw_data, ['val_idx'])),
                          'test': to_idx(get_key(raw_data, ['test_idx']))}
    else:
        dataset.splits = rand_splits(dataset.node_idx, train_prop=.6, valid_prop=.2)

    dataset_ind = dataset

    if ood_type == 'structure':
        dataset_ood_tr = create_multiplex_structure_noise_dataset(dataset)
        dataset_ood_te = create_multiplex_structure_noise_dataset(dataset)
    elif ood_type == 'feature':
        dataset_ood_tr = create_multiplex_feat_noise_dataset(dataset)
        dataset_ood_te = create_multiplex_feat_noise_dataset(dataset)
    elif ood_type == 'label':
        num_classes = y.max().item() + 1
        class_t = num_classes - 1
        label = y.squeeze()

        mask_ind = (label < class_t)
        mask_ood = (label == class_t)

        dataset_ind.node_idx = full_node_idx[mask_ind]
        dataset_ind.splits = rand_splits(dataset_ind.node_idx, train_prop=.6, valid_prop=.2)

        dataset_ood_tr = Data(x=x, y=y)
        dataset_ood_tr.edge_indices = tensor_edge_indices
        dataset_ood_te = Data(x=x, y=y)
        dataset_ood_te.edge_indices = tensor_edge_indices

        ood_nodes = full_node_idx[mask_ood]
        perm = torch.randperm(ood_nodes.size(0))
        ood_nodes = ood_nodes[perm]
        split_point = ood_nodes.size(0) // 2

        dataset_ood_tr.node_idx = ood_nodes[:split_point]
        dataset_ood_te.node_idx = ood_nodes[split_point:]
    else:
        dataset_ood_tr, dataset_ood_te = dataset, dataset

    return dataset_ind, dataset_ood_tr, dataset_ood_te


# ===================================================================
# 4. MAIN ROUTER (核心路由函数，千万不能丢)
# ===================================================================
def load_dataset(args):
    dataset_name = args.dataset.lower()

    # 🔴 核心修改：Amazon 也被归为 Real Multiplex 数据集处理
    if dataset_name in ('acm', 'dblp', 'imdb', 'amazon'):
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_real_multiplex_dataset(args.data_dir, dataset_name,
                                                                                  args.ood_type)
    elif dataset_name == 'movielens':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_movielens_dataset(args.data_dir, args.ood_type)
    elif dataset_name == 'aifb':
        dataset_ind, dataset_ood_tr, dataset_ood_te = load_aifb_multiplex(args.data_dir, args.ood_type)
    else:
        raise ValueError(f"[ERROR] Invalid or unsupported dataset name: {args.dataset}")

    print("\n" + "=" * 50)
    print(f"[STATISTICS] DATASET: {args.dataset.upper()}")
    num_nodes = dataset_ind.x.size(0)
    print(f"Nodes: {num_nodes:,}")

    if hasattr(dataset_ind, 'edge_indices') and isinstance(dataset_ind.edge_indices, list):
        m_structures = len(dataset_ind.edge_indices)
        total_edges = sum([adj.size(1) for adj in dataset_ind.edge_indices])
        print(f"Structures (M): {m_structures}")
        print(f"Total Edges: {total_edges:,}")
        for i, adj in enumerate(dataset_ind.edge_indices):
            print(f"  -> Layer {i + 1} Edges: {adj.size(1):,}")
    else:
        print("Structures (M): 1 (Single-layer graph)")
        print(f"Total Edges: {dataset_ind.edge_index.size(1):,}")

    num_classes = dataset_ind.y.size(1) if dataset_ind.y.dim() > 1 and dataset_ind.y.size(1) > 1 else int(
        dataset_ind.y.max().item()) + 1
    print(f"Classes: {num_classes}")
    print("=" * 50 + "\n")

    return dataset_ind, dataset_ood_tr, dataset_ood_te


# ===================================================================
# 5. OOD Generators
# ===================================================================
def create_multiplex_structure_noise_dataset(data, noise_ratio=0.3):
    new_edge_indices = []
    n = data.num_nodes
    for edge_index in data.edge_indices:
        edge_index_dropped, _ = utils.dropout_adj(edge_index, p=noise_ratio)
        num_add = int(edge_index.size(1) * noise_ratio)
        edge_index_added = torch.randint(0, n, (2, num_add), device=edge_index.device)
        new_adj = torch.cat([edge_index_dropped, edge_index_added], dim=1)
        new_edge_indices.append(new_adj)
    dataset = Data(x=data.x, y=data.y)
    dataset.edge_indices = new_edge_indices
    dataset.node_idx = torch.arange(n)
    return dataset


def create_multiplex_feat_noise_dataset(data):
    n = data.num_nodes
    perm_idx = torch.randperm(n)
    x_new = data.x[perm_idx]
    dataset = Data(x=x_new, y=data.y)
    dataset.edge_indices = getattr(data, 'edge_indices', [getattr(data, 'edge_index', None)])
    dataset.node_idx = torch.arange(n)
    return dataset