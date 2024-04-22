import torch
import logging
import copy

from torch_geometric.loader import NeighborSampler

from federatedscope.core.data import ClientData
from federatedscope.core.message import Message
from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.register import register_worker

from federatedscope.gfl.trainer.nodetrainer import NodeMiniBatchTrainer
from federatedscope.gfl.model.feddep import LocalDGen, FedDEP
from federatedscope.gfl.feddep.utils import GraphMender, HideGraph
from federatedscope.gfl.feddep.trainer import LocalGenTrainer, FedDEPTrainer

logger = logging.getLogger(__name__)


class FedDEPServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        r"""
        FedSage+ consists of three of training stages.
        Stage1: 0, local pre-train for generator.
        Stage2: -> 2 * feddep_epoch, federated training for generator.
        Stage3: -> 2 * feddep_epoch + total_round_num: federated training
        for GraphSAGE Classifier
        """
        super(FedDEPServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        assert self.model_num == 1, "Not supported multi-model for " \
                                    "FedDEPServer"

        # If state < feddep_epoch and state % 2 == 0:
        #     Server receive [model, embedding, label]
        # If state < feddep_epoch and state % 2 == 1:
        #     Server receive [gradient]
        self.feddep_epoch = 2 * self._cfg.feddep.feddep_epoch
        self.total_round_num = total_round_num + self.feddep_epoch
        self.grad_cnt = 0

    def _register_default_handlers(self):
        self.register_handlers('join_in', self.callback_funcs_for_join_in)
        self.register_handlers('join_in_info', self.callback_funcs_for_join_in)
        self.register_handlers('clf_para', self.callback_funcs_model_para)
        self.register_handlers('dep_para', self.callback_funcs_model_para)
        self.register_handlers('gradient', self.callback_funcs_gradient)
        self.register_handlers('metrics', self.callback_funcs_for_metrics)

    def callback_funcs_for_join_in(self, message: Message):
        if 'info' in message.msg_type:
            sender, info = message.sender, message.content
            for key in self._cfg.federate.join_in_info:
                assert key in info
            self.join_in_info[sender] = info
            logger.info('Server: Client #{:d} has joined in !'.format(sender))
        else:
            self.join_in_client_num += 1
            sender, address = message.sender, message.content
            if int(sender) == -1:  # assign number to client
                sender = self.join_in_client_num
                self.comm_manager.add_neighbors(neighbor_id=sender,
                                                address=address)
                self.comm_manager.send(
                    Message(msg_type='assign_client_id',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            content=str(sender)))
            else:
                self.comm_manager.add_neighbors(neighbor_id=sender,
                                                address=address)

            if len(self._cfg.federate.join_in_info) != 0:
                self.comm_manager.send(
                    Message(msg_type='ask_for_join_in_info',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            content=self._cfg.federate.join_in_info.copy()))

        if self.check_client_join_in():
            if self._cfg.federate.use_ss:
                self.broadcast_client_address()

            self.comm_manager.send(
                Message(msg_type='local_pretrain',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state))

    def callback_funcs_gradient(self, message: Message):
        round, _, content = message.state, message.sender, message.content
        gen_grad, ID = content
        # For a new round
        if round not in self.msg_buffer['train'].keys():
            self.msg_buffer['train'][round] = dict()
        self.grad_cnt += 1
        # Sum up all grad from other client
        if ID not in self.msg_buffer['train'][round]:
            self.msg_buffer['train'][round][ID] = dict()
            for key in gen_grad.keys():
                self.msg_buffer['train'][round][ID][key] = torch.FloatTensor(
                    gen_grad[key].cpu())
        else:
            for key in gen_grad.keys():
                self.msg_buffer['train'][round][ID][key] += torch.FloatTensor(
                    gen_grad[key].cpu())
        self.check_and_move_on()

    def check_and_move_on(self, check_eval_result=False):
        client_IDs = [i for i in range(1, self.client_num + 1)]

        if check_eval_result:
            # all clients are participating in evaluation
            minimal_number = self.client_num
        else:
            # sampled clients are participating in training
            minimal_number = self.sample_client_num

        # Transmit model and embedding to get gradient back
        if self.check_buffer(
                self.state, self.client_num
        ) and self.state < self._cfg.feddep.feddep_epoch and self.state\
                % 2 == 0:
            # FedGen: we should wait for all messages
            # 初始化一个空列表来存储每次叠放的结果
            concatenated_tensor = []
            num_of_clients = {}
            for sender in self.msg_buffer['train'][self.state]:
                content = self.msg_buffer['train'][self.state][sender]
                dep_para, embedding, label = content
                if sender == 1:
                    concatenated_tensor = embedding
                else:
                    concatenated_tensor = torch.cat(
                        (concatenated_tensor, embedding), dim=0)
                num_of_clients[sender] = embedding.size(0)
            adjacency_matrix = torch.matmul(concatenated_tensor,
                                            concatenated_tensor.t())
            adjacency_matrix = torch.max(adjacency_matrix,
                                         torch.zeros_like(adjacency_matrix))
            mask = torch.zeros_like(adjacency_matrix, dtype=torch.bool)
            # TODO 后来可以把这个5给换成参数
            for i, item in enumerate(adjacency_matrix):
                _, indices = torch.topk(item, 2)
                mask[i, indices] = True
            # 根据这个tensor算出一个矩阵,保存这个mask
            self.mask = mask
            self.num_of_clients = num_of_clients
            for sender in self.msg_buffer['train'][self.state]:
                content = self.msg_buffer['train'][self.state][sender]
                dep_para, embedding, label = content
                receiver_IDs = client_IDs[:sender - 1] + client_IDs[sender:]
                self.comm_manager.send(
                    Message(msg_type='dep_para',
                            sender=self.ID,
                            receiver=receiver_IDs,
                            state=self.state + 1,
                            content=[dep_para, embedding, label, sender]))
                logger.info(f'\tServer: Transmit dep_para to'
                            f' {receiver_IDs} @{self.state//2}.')
            self.state += 1

        # Sum up gradient client-wisely and send back
        # 如果当前状态self.state小于预定的训练轮次self._cfg.feddep.feddep_epoch，
        # 并且是偶数（表示训练阶段），服务器会检查是否收到了足够数量的客户端反馈。
        # 如果是，它会将依赖参数（可能是模型参数）发送给所有客户端，以便它们可以计算梯度并反馈给服务器。
        if self.check_buffer(
                self.state, self.client_num
        ) and self.state < self._cfg.feddep.feddep_epoch and self.state\
                % 2 == 1 and self.grad_cnt == self.client_num * (
                self.client_num - 1):
            for ID in self.msg_buffer['train'][self.state]:
                grad = self.msg_buffer['train'][self.state][ID]
                self.comm_manager.send(
                    Message(msg_type='gradient',
                            sender=self.ID,
                            receiver=[ID],
                            state=self.state + 1,
                            content=[grad, self.mask, self.num_of_clients]))
                logger.info(f'\tServer: Transmit gradient and mask to'
                            f' {ID} @{self.state // 2}.')
            # reset num of grad counter
            self.grad_cnt = 0
            self.state += 1

        if self.check_buffer(self.state, self.client_num
                             ) and self.state == self._cfg.feddep.feddep_epoch:
            self.state += 1
            # Setup Clf_trainer for each client
            self.comm_manager.send(
                Message(msg_type='setup',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state))

        if self.check_buffer(self.state, minimal_number, check_eval_result
                             ) and self.state >= self._cfg.feddep.feddep_epoch:

            if not check_eval_result:  # in the training process
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                msg_list = list()
                for client_id in train_msg_buffer:
                    msg_list.append(train_msg_buffer[client_id])

                # Trigger the monitor here (for training)
                if 'dissim' in self._cfg.eval.monitoring:
                    B_val = self._monitor.calc_blocal_dissim(
                        self.model.load_state_dict(), msg_list)
                    formatted_logs = self._monitor.format_eval_res(
                        B_val, rnd=self.state, role='Server #')
                    logger.info(formatted_logs)

                # Aggregate
                agg_info = {
                    'client_feedback': msg_list,
                    'recover_fun': self.recover_fun
                }
                result = self.aggregator.aggregate(agg_info)
                self.model.load_state_dict(result)
                self.aggregator.update(result)

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server : Starting evaluation at round {:d}.'.format(
                            self.state))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round(Round '
                        f'#{self.state}) -------------')
                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res)
                self.check_and_save()


class FedDEPClient(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):
        super(FedDEPClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)
        self.data = data

        # 此处hide_data是受损图，emb是中心向量，x_missing 是每个节点丢失的邻居
        self.hide_data, self.emb, self.x_missing = HideGraph(
            hidden_portion=self._cfg.feddep.hide_portion,
            num_pred=self._cfg.feddep.num_pred,
            num_proto=self._cfg.feddep.num_proto)(data=data['data'],
                                                  config=self._cfg)

        # Convert to `ClientData`
        self.hide_data = ClientData(self._cfg,
                                    train=[self.hide_data],
                                    val=[self.hide_data],
                                    test=[self.hide_data],
                                    data=self.hide_data)

        self.data.emb = self.emb
        self.device = device
        self.sage_batch_size = 64
        self.gen = LocalDGen(in_channels=data['data'].x.shape[-1],
                             out_channels=self._cfg.model.out_channels,
                             emb_len=self._cfg.model.in_channels,
                             hidden=self._cfg.model.hidden,
                             gen_hidden=self._cfg.feddep.gen_hidden,
                             dropout=self._cfg.model.dropout,
                             num_pred=self._cfg.feddep.num_pred)

        self.clf = model

        self.trainer_loc = LocalGenTrainer(self.gen,
                                           self.hide_data,
                                           self.device,
                                           self._cfg,
                                           monitor=self._monitor)

        self.register_handlers('clf_para', self.callback_funcs_for_model_para)
        self.register_handlers('local_pretrain',
                               self.callback_funcs_for_local_pre_train)
        self.register_handlers('gradient', self.callback_funcs_for_gradient)
        self.register_handlers('dep_para', self.callback_funcs_for_dep_para)
        self.register_handlers('setup', self.callback_funcs_for_setup_feddep)

    def callback_funcs_for_local_pre_train(self, message: Message):
        round, sender, _ = message.state, message.sender, message.content
        # Local pre-train
        logger.info(f'\tClient #{self.ID} pre-train start...')
        for i in range(self._cfg.feddep.loc_epoch):
            # 主要看这个pre-train的过程，client做了什么，做了一件事情就是更新本地的模型（邻居生成器）参数
            num_samples_train, _, _ = self.trainer_loc.train()
            logger.info(f'\tClient #{self.ID} local pre-train @Epoch {i}.')

        self.feddep = FedDEP(self.gen)

        self.trainer_feddep = FedDEPTrainer(self.feddep,
                                            self.hide_data,
                                            self.device,
                                            self._cfg,
                                            monitor=self._monitor)

        dep_para = self.feddep.cpu().state_dict()
        embedding = self.trainer_feddep.embedding()

        self.state = round
        logger.info(f'\tClient #{self.ID} pre-train finish!')
        self.num_missing = self.hide_data['data'].num_missing
        self.comm_manager.send(
            Message(msg_type='dep_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=[
                        dep_para, embedding, self.hide_data['data'].num_missing
                    ]))
        logger.info(f'\tClient #{self.ID} send dep_para to Server #{sender}.')

    def callback_funcs_for_dep_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        dep_para, embedding, label, ID = content
        dep_grad = self.trainer_feddep.cal_grad(self.x_missing, self.emb,
                                                dep_para, embedding)
        self.state = round
        self.comm_manager.send(
            Message(msg_type='gradient',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=[dep_grad, ID]))
        logger.info(f'\tClient #{self.ID}: send gradient to Server #{sender}.')

    def callback_funcs_for_gradient(self, message):
        # Aggregate gen_grad on server
        round, sender, content = message.state, message.sender, message.content
        dep_grad, mask, num_of_clients = content
        self.trainer_feddep.train()
        dep_para = self.trainer_feddep.update_by_grad(dep_grad)
        # TODO 修补本地子图,就是改这个ctx_data
        begin_ind = 0
        for i in range(1, self.ID + 1):
            begin_ind += num_of_clients[i]
        self.trainer_feddep.fill_edge(mask, begin_ind, num_of_clients[self.ID])
        self.mask_real = mask[begin_ind:begin_ind + num_of_clients[self.ID],
                              begin_ind:begin_ind + num_of_clients[self.ID]]
        embedding = self.trainer_feddep.embedding()
        self.state = round
        self.comm_manager.send(
            Message(msg_type='dep_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=[dep_para, embedding, self.num_missing]))
        logger.info(f'\tClient #{self.ID}: send dep_para to Server #{sender}.')

    def callback_funcs_for_setup_feddep(self, message: Message):
        round, sender, _ = message.state, message.sender, message.content
        self.filled_data = GraphMender(
            model=self.feddep,
            impaired_data=self.hide_data['data'].cpu(),
            original_data=self.data['data'],
            mask_real=self.mask_real)
        subgraph_sampler = NeighborSampler(
            self.data['data'].edge_index,
            num_nodes=self.data['data'].num_nodes,
            sizes=[-1],
            batch_size=4096,
            shuffle=False,
            num_workers=self._cfg.dataloader.num_workers)
        fill_dataloader = {
            'data': self.filled_data,
            'train': NeighborSampler(
                self.filled_data.edge_index,
                num_nodes=self.filled_data.num_nodes,
                node_idx=self.filled_data.train_idx,
                sizes=self._cfg.dataloader.sizes,
                batch_size=self.sage_batch_size,
                shuffle=self._cfg.dataloader.shuffle,
                num_workers=self._cfg.dataloader.num_workers),
            'val': subgraph_sampler,
            'test': subgraph_sampler
        }
        self._cfg.merge_from_list(
            ['dataloader.batch_size', self.sage_batch_size])
        self.trainer_clf = NodeMiniBatchTrainer(self.clf,
                                                fill_dataloader,
                                                self.device,
                                                self._cfg,
                                                monitor=self._monitor)
        sample_size, clf_para, results = self.trainer_clf.train()
        self.state = round
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID)))
        self.comm_manager.send(
            Message(msg_type='clf_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, clf_para)))

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.trainer_clf.update(content)
        self.state = round
        sample_size, clf_para, results = self.trainer_clf.train()
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            clf_para = copy.deepcopy(clf_para)
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID)))
        self.comm_manager.send(
            Message(msg_type='clf_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, clf_para)))


def call_my_worker(method):
    if method == 'feddep':
        worker_builder = {'client': FedDEPClient, 'server': FedDEPServer}
        return worker_builder


register_worker('feddep', call_my_worker)
