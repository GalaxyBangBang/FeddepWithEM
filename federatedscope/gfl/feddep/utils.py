import torch
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.loader import NeighborSampler

import networkx as nx
import numpy as np

from federatedscope.core.configs.config import global_cfg
from federatedscope.gfl.model.sage import Encoder

from federatedscope.gfl.feddep.dec_cluster.clustering import train_clustering
import random


def get_prototypes(emb,
                   K,
                   batch_size=None,
                   CUDA=None,
                   ae_pretrained_epochs=None,
                   ae_finetune_epochs=None,
                   dec_epochs=None):

    emb_shape = len(emb[0])
    K = int(K)
    # 把每个节点指定一个类别（聚类）
    proto_idx = train_clustering(node_embs=emb,
                                 num_prototypes=K,
                                 batch_size=batch_size,
                                 CUDA=CUDA,
                                 ae_pretrained_epochs=ae_pretrained_epochs,
                                 ae_finetune_epochs=ae_finetune_epochs,
                                 dec_epochs=dec_epochs).reshape(-1)

    prototypes = np.zeros(shape=(K, emb_shape))
    proto_idx = np.asarray(proto_idx, dtype=np.int32).reshape(-1)
    if emb.device != 'cpu':
        emb = emb.cpu()
    emb = emb.numpy()
    for cluster in range(K):
        row_ix = np.where(proto_idx == cluster)
        prototypes[cluster] = emb[row_ix].mean(axis=0)
    return prototypes, proto_idx


def get_emb(all_data, _cfg):
    if _cfg.device == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(_cfg.device)
    subgraph_sampler = NeighborSampler(
        all_data.edge_index,
        num_nodes=all_data.num_nodes,
        node_idx=torch.tensor([i for i in range(all_data.num_nodes)]),
        sizes=[5] * _cfg.feddep.encoder.L,
        batch_size=4096,
        shuffle=False)
    # 训练集中所有节点的ID
    train_idx = torch.where(all_data.train_mask == True)[0]
    dataloader = {
        'data': all_data,
        'train': NeighborSampler(all_data.edge_index,
                                 num_nodes=all_data.num_nodes,
                                 node_idx=train_idx,
                                 sizes=[5] * _cfg.feddep.encoder.L,
                                 batch_size=_cfg.feddep.encoder.batch_size,
                                 shuffle=_cfg.data.shuffle),
        'val': subgraph_sampler,
        'test': subgraph_sampler
    }

    encoder = Encoder(in_channels=len(all_data.x[0]),
                      out_channels=_cfg.feddep.encoder.out_channels,
                      hidden=_cfg.feddep.encoder.hidden,
                      max_depth=_cfg.feddep.encoder.L,
                      dropout=_cfg.feddep.encoder.dropout)

    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(),
                                 lr=0.01,
                                 weight_decay=5e-4)
    for epoch in range(_cfg.feddep.encoder.epochs):
        pbar = tqdm(total=int(all_data.train_mask.sum()))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_correct = 0
        for batch_size, n_id, adjs in dataloader['train']:
            optimizer.zero_grad()
            out = encoder(x=all_data.x[n_id], adjs=adjs, device=device)
            loss = torch.nn.functional.cross_entropy(
                out.to(device), all_data.y[n_id[:batch_size]].to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            total_correct += int(
                out.argmax(dim=-1).cpu().eq(
                    all_data.y[n_id[:batch_size]]).sum().cpu())
            pbar.update(batch_size)

        pbar.close()
        loss = total_loss / len(dataloader['train'])
        approx_acc = total_correct / int(all_data.train_mask.sum())
        print(
            f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}'
        )

    encoder.eval()
    all_emb = encoder.get_enc(dataloader['data'].x, dataloader['test'],
                              device).detach()

    return all_emb


class HideGraph(BaseTransform):
    r"""
    Generate impaired graph with labels and features to train NeighGen,
    hide Node from validation set from raw graph.

    Arguments:
        hidden_portion (int): hidden_portion of validation set.
        num_pred (int): hyperparameters which limit
            the maximum value of the prediction

    :returns:
        filled_data : impaired graph with attribute "num_missing"
    :rtype:
        nx.Graph
    """
    def __init__(self, hidden_portion=0.5, num_pred=5, num_proto=5):
        self.hidden_portion = hidden_portion
        self.num_pred = num_pred
        self.num_proto = num_proto

    def get_prototypes(self, X, config):
        emb = get_emb(X, config)

        self.prototypes, self.proto_idx = get_prototypes(
            emb=emb,
            K=self.num_proto,
            batch_size=config.feddep.cluster_batch_size,
            CUDA=config.use_gpu,
            ae_pretrained_epochs=config.feddep.ae_pretrained_epochs,
            ae_finetune_epochs=config.feddep.ae_finetune_epochs,
            dec_epochs=config.feddep.dec_epochs)
        self.emb = np.zeros((len(self.proto_idx), len(self.prototypes[0])))
        for i in range(len(self.emb)):
            self.emb[i] = self.prototypes[self.proto_idx[i]]

        return

    def __call__(self, data, config=None):
        # get prototypes
        self.get_prototypes(data, config)

        # 表示每个节点可能缺失邻居的嵌入向量
        self.x_missing = torch.zeros(
            (len(data.x), self.num_pred, len(self.emb[0])))

        val_ids = torch.where(data.val_mask == True)[0]
        hide_ids = np.random.choice(val_ids,
                                    int(len(val_ids) * self.hidden_portion),
                                    replace=False)
        remaining_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        remaining_mask[hide_ids] = False
        remaining_nodes = torch.where(remaining_mask == True)[0].numpy()

        # 为每个节点创建一个列表来存储缺失的邻居节点索引，并更新数据集中的节点嵌入向量
        data.ids_missing = [[] for _ in range(data.num_nodes)]

        data.emb = torch.tensor(self.emb)
        data.index_orig = torch.tensor([i for i in range(data.num_nodes)],
                                       dtype=torch.long)

        G = to_networkx(data,
                        node_attrs=[
                            'x', 'y', 'train_mask', 'val_mask', 'test_mask',
                            'index_orig', 'ids_missing', 'emb'
                        ],
                        to_undirected=True)
        # 对于每个被隐藏的节点，更新其邻居节点的ids_missing列表，并将缺失的邻居嵌入向量存储在x_missing中。
        # 这应该是ground truth
        for missing_node in hide_ids:
            neighbors = G.neighbors(missing_node)
            for i in neighbors:
                G.nodes[i]['ids_missing'].append(missing_node)
        for i in G.nodes:
            ids_missing = G.nodes[i]['ids_missing']
            del G.nodes[i]['ids_missing']
            G.nodes[i]['num_missing'] = np.array([len(ids_missing)],
                                                 dtype=np.float32)
            if len(ids_missing) > 0:
                if len(ids_missing) <= self.num_pred:
                    G.nodes[i]['x_missing'] = torch.tensor(
                        np.vstack((self.emb[ids_missing],
                                   np.zeros((self.num_pred - len(ids_missing),
                                             self.emb.shape[1])))))

                else:
                    G.nodes[i]['x_missing'] = torch.tensor(
                        self.emb[ids_missing[:self.num_pred]])
            else:
                G.nodes[i]['x_missing'] = torch.zeros(
                    (self.num_pred, self.emb.shape[1]))
            self.x_missing[i] = G.nodes[i]['x_missing']
        # 从完整的图中提取出仅包含剩余节点的子图，这个子图即为受损图
        impaired_graph = from_networkx(nx.subgraph(G, remaining_nodes))
        # print(impaired_graph)

        return impaired_graph, self.emb, self.x_missing

    def __repr__(self):
        return f'{self.__class__.__name__}({self.hidden_portion})'


def FillGraph(impaired_data, original_data, pred_missing, pred_feats, num_pred,
              mask_real):
    # Mend the original data
    original_data = original_data.detach().cpu()
    G = to_networkx(original_data,
                    node_attrs=[
                        'x', 'y', 'train_mask', 'val_mask', 'test_mask',
                        'index_orig', 'emb'
                    ],
                    to_undirected=True)

    # TODO 增加新的edge_index
    for i in range(mask_real.size(0)):
        # 随机生成一个 0 到 9 之间的整数
        skip = random.randint(0, 9)

        # 根据随机生成的 skip 值决定是否跳过当前迭代
        if skip % 2 == 0:  # 举例：如果生成的随机数是偶数，就跳过
            continue  # 跳过当前迭代，继续下一个迭代
        neighbors = G.neighbors(i)
        mask_edges = torch.nonzero(mask_real[i]).squeeze()
        if mask_edges.dim() != 0:
            for edge in mask_edges:
                if edge not in neighbors:
                    G.add_edge(i, edge.item())
                    break
    original_data = from_networkx(G)
    new_edge_index = original_data.edge_index.T
    pred_missing = pred_missing.detach().cpu().numpy()

    pred_feats = pred_feats.detach().cpu().reshape(
        (len(pred_missing), num_pred, -1))

    emb_len = pred_feats.shape[-1]
    start_id = original_data.num_nodes
    mend_emb = torch.zeros(size=(start_id, num_pred, emb_len))
    for node in range(len(pred_missing)):
        num_fill_node = np.around(pred_missing[node]).astype(np.int32).item()
        # print(f"num_fill_node: {num_fill_node}")
        if num_fill_node > 0:
            org_id = impaired_data.index_orig[node]
            mend_emb[org_id][:num_fill_node] += pred_feats[
                node][:num_fill_node]

    filled_data = Data(
        x=original_data.x,
        edge_index=new_edge_index.T,
        train_idx=torch.where(original_data.train_mask == True)[0],
        valid_idx=torch.where(original_data.val_mask == True)[0],
        test_idx=torch.where(original_data.test_mask == True)[0],
        y=original_data.y,
        mend_emb=mend_emb)

    return filled_data


@torch.no_grad()
def GraphMender(model, impaired_data, original_data, mask_real):
    r"""Mend the graph with generation model
    Arguments:
        model (torch.nn.module): trained generation model
        impaired_data (PyG.Data): impaired graph
        original_data (PyG.Data): raw graph
    :returns:
        filled_data : Graph after Data Enhancement
    :rtype:
        PyG.data
    """
    device = impaired_data.x.device
    model = model.to(device)
    pred_missing, pred_feats, _ = model(impaired_data)

    return FillGraph(impaired_data, original_data, pred_missing, pred_feats,
                     global_cfg.feddep.num_pred, mask_real)
