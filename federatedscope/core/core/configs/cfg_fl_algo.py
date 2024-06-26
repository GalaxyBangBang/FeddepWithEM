from federatedscope.core.configs.config import CN
from federatedscope.core.configs.yacs_config import Argument
from federatedscope.register import register_config


def extend_fl_algo_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # fedopt related options, a general fl algorithm
    # ---------------------------------------------------------------------- #
    cfg.fedopt = CN()

    cfg.fedopt.use = False

    cfg.fedopt.optimizer = CN(new_allowed=True)
    cfg.fedopt.optimizer.type = Argument(
        'SGD', description="optimizer type for FedOPT")
    cfg.fedopt.optimizer.lr = Argument(
        0.01, description="learning rate for FedOPT optimizer")

    # ---------------------------------------------------------------------- #
    # fedprox related options, a general fl algorithm
    # ---------------------------------------------------------------------- #
    cfg.fedprox = CN()

    cfg.fedprox.use = False
    cfg.fedprox.mu = 0.

    # ---------------------------------------------------------------------- #
    # Personalization related options, pFL
    # ---------------------------------------------------------------------- #
    cfg.personalization = CN()

    # client-distinct param names, e.g., ['pre', 'post']
    cfg.personalization.local_param = []
    cfg.personalization.share_non_trainable_para = False
    cfg.personalization.local_update_steps = -1
    # @regular_weight:
    # The smaller the regular_weight is, the stronger emphasising on
    # personalized model
    # For Ditto, the default value=0.1, the search space is [0.05, 0.1, 0.2,
    # 1, 2]
    # For pFedMe, the default value=15
    cfg.personalization.regular_weight = 0.1

    # @lr:
    # 1) For pFedME, the personalized learning rate to calculate theta
    # approximately using K steps
    # 2) 0.0 indicates use the value according to optimizer.lr in case of
    # users have not specify a valid lr
    cfg.personalization.lr = 0.0

    cfg.personalization.K = 5  # the local approximation steps for pFedMe
    cfg.personalization.beta = 1.0  # the average moving parameter for pFedMe

    # ---------------------------------------------------------------------- #
    # FedSage+ related options, gfl
    # ---------------------------------------------------------------------- #
    cfg.fedsageplus = CN()

    # Number of nodes generated by the generator
    cfg.fedsageplus.num_pred = 5
    # Hidden layer dimension of generator
    cfg.fedsageplus.gen_hidden = 128
    # Hide graph portion
    cfg.fedsageplus.hide_portion = 0.5
    # Federated training round for generator
    cfg.fedsageplus.fedgen_epoch = 200
    # Local pre-train round for generator
    cfg.fedsageplus.loc_epoch = 1
    # Coefficient for criterion number of missing node
    cfg.fedsageplus.a = 1.0
    # Coefficient for criterion feature
    cfg.fedsageplus.b = 1.0
    # Coefficient for criterion classification
    cfg.fedsageplus.c = 1.0



    # ---------------------------------------------------------------------- #
    # FedDEP related options, gfl
    # ---------------------------------------------------------------------- #
    cfg.feddep = CN()
    cfg.feddep.num_proto= 5
    cfg.feddep.cluster_batch_size=64
    cfg.feddep.ae_pretrained_epochs= 20
    cfg.feddep.ae_finetune_epochs=30
    cfg.feddep.dec_epochs= 50

    cfg.feddep.num_pred = 5
    cfg.feddep.gen_hidden = 128
    cfg.feddep.hide_portion = 0.5
    cfg.feddep.feddep_epoch = 200
    cfg.feddep.loc_epoch = 1
    cfg.feddep.beta_d = 1.0
    cfg.feddep.beta_n = 1.0
    cfg.feddep.beta_c = 1.0

    cfg.feddep.encoder=CN()
    cfg.feddep.encoder.type= 'feddep_encoder'
    cfg.feddep.encoder.hidden=64
    cfg.feddep.encoder.L=2
    cfg.feddep.encoder.batch_size= 64
    cfg.feddep.encoder.dropout=0.5
    cfg.feddep.encoder.epochs=30
    cfg.feddep.encoder.out_channels=7

    # ---------------------------------------------------------------------- #
    # GCFL+ related options, gfl
    # ---------------------------------------------------------------------- #
    cfg.gcflplus = CN()

    # Bound for mean_norm
    cfg.gcflplus.EPS_1 = 0.05
    # Bound for max_norm
    cfg.gcflplus.EPS_2 = 0.1
    # Length of the gradient sequence
    cfg.gcflplus.seq_length = 5
    # Whether standardized dtw_distances
    cfg.gcflplus.standardize = False

    # ---------------------------------------------------------------------- #
    # FLIT+ related options, gfl
    # ---------------------------------------------------------------------- #
    cfg.flitplus = CN()

    cfg.flitplus.tmpFed = 0.5  # gamma in focal loss (Eq.4)
    cfg.flitplus.lambdavat = 0.5  # lambda in phi (Eq.10)
    cfg.flitplus.factor_ema = 0.8  # beta in omega (Eq.12)
    cfg.flitplus.weightReg = 1.0  # balance lossLocalLabel and lossLocalVAT

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_fl_algo_cfg)


def assert_fl_algo_cfg(cfg):
    if cfg.personalization.local_update_steps == -1:
        # By default, use the same step to normal mode
        cfg.personalization.local_update_steps = \
            cfg.train.local_update_steps
        cfg.personalization.local_update_steps = \
            cfg.train.local_update_steps

    if cfg.personalization.lr <= 0.0:
        # By default, use the same lr to normal mode
        cfg.personalization.lr = cfg.train.optimizer.lr


register_config("fl_algo", extend_fl_algo_cfg)
