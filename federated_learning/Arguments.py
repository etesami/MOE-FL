class Arguments():
    def __init__(
        self, batch_size=None, test_batch_size=None, rounds=None, epochs=None, 
        lr=None, momentum=None, weight_decay=None, shards_num=None, shards_per_worker_num=None, total_users_num=None, selected_users_num=None, server_data_fraction=None, 
        server_pure=None, mode=None, attack_type=None, attackers_num=None, 
        use_cuda=None, device=None, seed=None, log_interval=None, 
        log_level=None, log_format=None, log_dir=None, neptune_log=None, local_log=None):

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.rounds = rounds
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.shards_num = shards_num
        self.shards_per_worker_num = shards_per_worker_num
        self.total_users_num = total_users_num
        self.selected_users_num = selected_users_num
        self.server_data_fraction = server_data_fraction
        self.server_pure = server_pure
        self.mode = mode
        self.attack_type = attack_type
        self.attackers_num = attackers_num
        self.use_cuda = use_cuda
        self.device = device
        self.seed = seed
        self.log_interval = log_interval
        self.log_level = log_level
        self.log_format = log_format
        self.log_dir = log_dir
        self.neptune_log = neptune_log
        self.local_log = local_log