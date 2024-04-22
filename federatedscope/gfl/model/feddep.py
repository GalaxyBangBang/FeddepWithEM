from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from federatedscope.gfl.model import SAGE_Net
from federatedscope.gfl.model.sage import Classifier_F
"""
https://proceedings.neurips.cc//paper/2021/file/ \
34adeb8e3242824038aa65460a47c29e-Paper.pdf
Fedsageplus models from the "Subgraph Federated Learning with Missing
Neighbor Generation" (FedSage+) paper, in NeurIPS'21
Source: https://github.com/zkhku/fedsage
"""


class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, inputs):
        rand = torch.normal(0, 1, size=inputs.shape)

        return inputs + rand.to(inputs.device)


class EmbGenerator(nn.Module):
    def __init__(self, latent_dim, dropout, num_pred, feat_shape):
        super(EmbGenerator, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        self.dropout = dropout
        # 这个sampling像一个对正态分布进行采样的东西
        self.sample = Sampling()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 2048)
        self.fc_flat = nn.Linear(2048, self.num_pred * self.feat_shape)

    def forward(self, x):
        x = self.sample(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))

        return x


class NumPredictor(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super(NumPredictor, self).__init__()
        self.reg_1 = nn.Linear(self.latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.reg_1(x))
        return x


# Mend the graph via NeighGen
class MendGraph(nn.Module):
    def __init__(self, num_pred):
        super(MendGraph, self).__init__()
        self.num_pred = num_pred
        for param in self.parameters():
            param.requires_grad = False

    def mend_graph_old(self, x, edge_index, pred_degree, gen_feats):
        device = gen_feats.device
        num_node, num_feature = x.shape
        new_edges = []
        gen_feats = gen_feats.view(-1, self.num_pred, num_feature)

        if pred_degree.device.type != 'cpu':
            pred_degree = pred_degree.cpu()
        pred_degree = torch._cast_Int(torch.round(pred_degree)).detach()
        x = x.detach()
        fill_feats = torch.vstack((x, gen_feats.view(-1, num_feature)))

        for i in range(num_node):
            for j in range(min(self.num_pred, max(0, pred_degree[i]))):
                new_edges.append(
                    np.asarray([i, num_node + i * self.num_pred + j]))

        new_edges = torch.tensor(np.asarray(new_edges).reshape((-1, 2)),
                                 dtype=torch.int64).T
        new_edges = new_edges.to(device)
        if len(new_edges) > 0:
            fill_edges = torch.hstack((edge_index, new_edges))
        else:
            fill_edges = torch.clone(edge_index)
        return fill_feats, fill_edges

    def mend_graph(self, pred_degree, gen_embs):
        device = gen_embs.device
        if pred_degree.device.type != 'cpu':
            pred_degree = pred_degree.cpu()
        num_node = len(pred_degree)

        mend_emb = gen_embs.view(num_node, self.num_pred, -1)
        # mend_emb = torch.zeros(gen_embs.shape).to(device)

        pred_degree = torch._cast_Int(torch.round(pred_degree)).detach()

        mend_num = torch.zeros(mend_emb.shape).to(device)
        for i in range(num_node):
            mend_num[i][:min(self.num_pred, max(0, pred_degree[i]))] = 1

        mend_emb = mend_emb * mend_num

        return mend_emb

    def forward(self, pred_missing, gen_feats):
        mend_emb = self.mend_graph(pred_missing, gen_feats)

        return mend_emb


class LocalDGen(nn.Module):
    def __init__(
            self,
            in_channels,
            emb_len,  #emb_len
            out_channels,
            hidden,
            gen_hidden,
            dropout=0.5,
            num_pred=5):
        super(LocalDGen, self).__init__()

        self.encoder_model = SAGE_Net(in_channels=in_channels,
                                      out_channels=gen_hidden,
                                      hidden=hidden,
                                      max_depth=2,
                                      dropout=dropout)
        self.reg_model = NumPredictor(latent_dim=gen_hidden)
        self.gen = EmbGenerator(latent_dim=gen_hidden,
                                dropout=dropout,
                                num_pred=num_pred,
                                feat_shape=emb_len)
        self.mend_graph = MendGraph(num_pred)

        self.classifier = Classifier_F(in_channels=(in_channels, emb_len),
                                       out_channels=out_channels,
                                       hidden=hidden,
                                       max_depth=2,
                                       dropout=dropout)

    def forward(self, data):
        x = self.encoder_model(data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats = self.mend_graph(degree, gen_feat)
        nc_pred = self.classifier(
            Data(x=data.x, mend_emb=mend_feats, edge_index=data.edge_index))
        return degree, gen_feat, nc_pred[:data.num_nodes]

    def inference(self, impared_data):
        x = self.encoder_model(impared_data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats = self.mend_graph(degree, gen_feat)
        nc_pred = self.classifier(
            Data(x=impared_data.x,
                 mend_emb=mend_feats,
                 edge_index=impared_data.edge_index))
        return degree, gen_feat, nc_pred[:impared_data.num_nodes]


class FedDEP(nn.Module):
    def __init__(self, local_graph: LocalDGen):
        super(FedDEP, self).__init__()
        self.encoder_model = local_graph.encoder_model
        self.reg_model = local_graph.reg_model
        self.gen = local_graph.gen
        self.mend_graph = local_graph.mend_graph
        self.classifier = local_graph.classifier
        self.encoder_model.requires_grad_(False)
        self.reg_model.requires_grad_(False)
        self.mend_graph.requires_grad_(False)
        self.classifier.requires_grad_(False)

    def forward(self, data):
        x = self.encoder_model(data)
        degree = self.reg_model(x)
        gen_feat = self.gen(x)
        mend_feats = self.mend_graph(degree, gen_feat)
        nc_pred = self.classifier(
            Data(x=data.x, mend_emb=mend_feats, edge_index=data.edge_index))
        return degree, gen_feat, nc_pred[:data.num_nodes]
