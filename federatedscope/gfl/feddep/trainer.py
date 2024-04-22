import torch
import copy
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, from_networkx

from federatedscope.gfl.loss.reconstruction_loss import LocalRecLoss, FedRecLoss
from federatedscope.gfl.trainer.nodetrainer import NodeFullBatchTrainer


class LocalGenTrainer(NodeFullBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(LocalGenTrainer, self).__init__(model, data, device, config,
                                              only_for_eval, monitor)
        self.criterion_num = F.smooth_l1_loss
        self.criterion_rec = LocalRecLoss

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        # mask为缺失邻居的节点
        mask = batch['{}_mask'.format(ctx.cur_mode)]
        pred_missing, pred_emb, nc_pred = ctx.model(batch)
        pred_missing, pred_emb, nc_pred = pred_missing[mask], pred_emb[
            mask], nc_pred[mask]
        mask_int = np.where(mask.cpu().numpy() == True)[0]
        loss_num = self.criterion_num(pred_missing, batch.num_missing[mask])
        loss_rec = self.criterion_rec(
            pred_embs=pred_emb,
            true_embs=[batch.x_missing[node] for node in mask_int],
            pred_missing=pred_missing,
            true_missing=batch.num_missing[mask],
            num_pred=self.cfg.feddep.num_pred)
        loss_clf = ctx.criterion(nc_pred, batch.y[mask])
        ctx.loss_batch = self.cfg.feddep.beta_d * loss_num + self.cfg.feddep.beta_c * loss_clf
        ctx.loss_batch += self.cfg.feddep.beta_n * loss_rec

        ctx.batch_size = torch.sum(mask).item()
        ctx.loss_batch = ctx.loss_batch.float()

        ctx.y_true = batch.num_missing[mask]
        ctx.y_prob = pred_missing


class FedDEPTrainer(LocalGenTrainer):
    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        mask = batch['{}_mask'.format(ctx.cur_mode)]
        pred_missing, pred_emb, nc_pred = ctx.model(batch)
        pred_missing, pred_emb, nc_pred = pred_missing[mask], pred_emb[
            mask], nc_pred[mask]
        mask_int = np.where(mask.cpu().numpy() == True)[0]
        loss_num = self.criterion_num(pred_missing, batch.num_missing[mask])
        loss_rec = self.criterion_rec(
            pred_embs=pred_emb,
            true_embs=[batch.x_missing[node] for node in mask_int],
            pred_missing=pred_missing,
            true_missing=batch.num_missing[mask],
            num_pred=self.cfg.feddep.num_pred)
        loss_clf = ctx.criterion(nc_pred, batch.y[mask])
        ctx.batch_size = torch.sum(mask).item()
        ctx.loss_batch = self.cfg.feddep.beta_d * loss_num + self.cfg.feddep.beta_c * loss_clf

        ctx.loss_batch += self.cfg.feddep.beta_n * loss_rec
        ctx.loss_batch = ctx.loss_batch.float() / self.cfg.federate.client_num

        ctx.y_true = batch.num_missing[mask]
        ctx.y_prob = pred_missing

    def update_by_grad(self, grads):
        """
        Arguments:
            grads: grads of other clients to optimize the local model
        :returns:
            state_dict of generation model
        """
        for key in grads.keys():
            if isinstance(grads[key], list):
                grads[key] = torch.FloatTensor(grads[key]).to(self.ctx.device)

        for key, value in self.ctx.model.named_parameters():
            value.grad += grads[key]
        self.ctx.optimizer.step()
        return self.ctx.model.cpu().state_dict()

    def cal_grad(self, x_missing, G_emb, model_para, embedding):
        """
        Arguments:
            x_missing:这个是本地的ground_truth
            G_emb：这个是本地的embedding
            model_para: 这个是别人的模型参数
            embedding: 这个是别人的embedding
        :returns:
            grads: grads to optimize the model of other clients
        """
        para_backup = copy.deepcopy(self.ctx.model.cpu().state_dict())

        for key in model_para.keys():
            if isinstance(model_para[key], list):
                model_para[key] = torch.FloatTensor(model_para[key])
        self.ctx.model.load_state_dict(model_para)
        self.ctx.model = self.ctx.model.to(self.ctx.device)
        self.ctx.model.train()

        embedding = torch.FloatTensor(embedding).to(self.ctx.device)
        pred_missing = self.ctx.model.reg_model(embedding)
        pred_emb = self.ctx.model.gen(embedding)
        emb_len = pred_emb.shape[-1] // self.cfg.feddep.num_pred

        x_missing = x_missing.to(self.ctx.device)

        choice = np.random.choice(len(x_missing), embedding.shape[0])
        global_target_emb = []
        for c_i in choice:
            choice_i = np.random.choice(len(x_missing[c_i]),
                                        self.cfg.feddep.num_pred)
            for ch_i in choice_i:
                if torch.sum(x_missing[c_i][ch_i]) < 1e-15:
                    global_target_emb.append(G_emb[c_i])
                else:
                    global_target_emb.append(
                        x_missing[c_i][ch_i].detach().cpu().numpy())
        global_target_emb = np.asarray(global_target_emb).reshape(
            (embedding.shape[0], self.cfg.feddep.num_pred, emb_len))

        loss_emb = FedRecLoss(pred_embs=pred_emb,
                              true_embs=global_target_emb,
                              pred_missing=pred_missing,
                              num_pred=self.cfg.feddep.num_pred)

        loss = (1.0 / self.cfg.federate.client_num * self.cfg.feddep.beta_n *
                loss_emb).requires_grad_()

        loss.backward()
        grads = {
            key: value.grad
            for key, value in self.ctx.model.named_parameters()
        }
        # Rollback
        self.ctx.model.load_state_dict(para_backup)
        return grads

    @torch.no_grad()
    def embedding(self):
        model = self.ctx.model.to(self.ctx.device)
        data = self.ctx.data['data'].to(self.ctx.device)
        return model.encoder_model(data).to('cpu')

    def fill_edge(self, mask, begin_ind, num):
        if not hasattr(self, 'already_filled'):
            self.already_filled = True
        else:
            return
        mask_real = mask[begin_ind:begin_ind + num, begin_ind:begin_ind + num]
        data = self.ctx.data['data'].to(self.ctx.device)
        G = to_networkx(data,
                        node_attrs=[
                            'x', 'y', 'train_mask', 'val_mask', 'test_mask',
                            'index_orig', 'emb'
                        ],
                        to_undirected=True)
        for i in range(mask_real.size(0)):
            neighbors = G.neighbors(i)
            # 取非0元素的下标，检查是不是在neighbors里面，如果不在，就添加一个neighbors
            mask_edges = torch.nonzero(mask_real[i]).squeeze()
            if mask_edges.dim() != 0:
                for edge in mask_edges:
                    if edge not in neighbors:
                        G.add_edge(i, edge.item())
        data = from_networkx(G)
        self.ctx.data['data'] = data
