class Arguments():
    def __init__(
        self, batch_size, test_batch_size, rounds, epochs, 
        lr, momentum, weight_decay, shards_num, shards_per_worker_num, attack_type, attackers_num, use_cuda, device, seed, log_interval, 
        log_level, log_format, log_dir, neptune_log, local_log):

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.rounds = rounds
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.shards_num = shards_num
        self.shards_per_worker_num = shards_per_worker_num
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