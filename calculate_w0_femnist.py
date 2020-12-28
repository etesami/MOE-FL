"""
Usage: 
    run-study-iid.py --pure --output-prefix=NAME [--log] [--nep-log]
    run-study-iid.py --not-pure --attack=<ATTACK-TYPE> --output-prefix=NAME [--log] [--nep-log]
"""
from docopt import docopt
import logging
import random
import neptune
import numpy as np
import syft as sy
from torch import load
from torchvision import transforms
from torch.utils.data import DataLoader
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.FederatedLearning import FederatedLearning
from federated_learning.helper import utils
arguments = docopt(__doc__)
CONFIG_PATH = 'configs/defaults.yml'


if __name__ == '__main__':
    configs = utils.load_config(CONFIG_PATH)
    logging.basicConfig(format='%(asctime)s %(message)s', level=configs['log']['level'])
    random.seed(configs['runtime']['random_seed'])

    # Logging initialization
    log_enable = True if arguments['--log'] else False
    output_dir = None
    if log_enable:
        output_dir = utils.make_output_dir(
            configs['log']['root_output_dir'], arguments['--output-prefix'])
        utils.save_configs(output_dir, configs)
    neptune_enable = True if arguments['--nep-log'] else False
    
    # Neptune logging initialization
    if neptune_enable:
        neptune.init(configs['log']['neptune_init'])
        neptune.create_experiment(name = configs['log']['neptune_exp'])
        
    epochs_num = configs['runtime']['epochs']
    rounds_num = configs['runtime']['rounds']

    fl = FederatedLearning(
        configs['runtime']['server_w0_batch_size'], 
        configs['runtime']['test_batch_size'], 
        configs['runtime']['lr'], 
        configs['runtime']['reg'],
        configs['runtime']['momentum'], 
        neptune_enable, log_enable, 
        configs['log']['interval'], 
        output_dir, 
        configs['runtime']['random_seed'])

    fl.create_server()
    fl.create_server_model()

    raw_train_data = utils.preprocess_leaf_data(
        utils.load_leaf_train(configs['data']['FEMNIST_PATH']), only_digits=True
    )
    raw_test_data = utils.preprocess_leaf_data(
        utils.load_leaf_test(configs['data']['FEMNIST_PATH']), min_num_samples=configs['runtime']['test_batch_size'], only_digits=True
    )

    # common users in processed test/train dataset
    workers_idx_all = sorted(list(set(raw_test_data.keys()).intersection(raw_train_data.keys())))
    logging.info("Total of {} workers are in the dataset.".format(len(workers_idx_all)))

    workers_idx_to_be_used = utils.get_workers_idx(
        workers_idx_all,
        configs['runtime']['femnist_users_num'],
        []
    )
    logging.info("Select {} workers to be used from the dataset.".format(len(workers_idx_to_be_used)))

    fl.create_workers(workers_idx_to_be_used)
    fl.create_workers_model(workers_idx_to_be_used)

    trusted_idx = utils.get_workers_idx(
        workers_idx_to_be_used, configs['runtime']['femnist_trusted_num'], [])
    eavesdroppers_idx = utils.get_workers_idx(
        workers_idx_to_be_used, configs['runtime']['femnist_eavesdropper_num'], trusted_idx)
    normal_idx = utils.get_workers_idx(
        workers_idx_to_be_used, 
        len(workers_idx_to_be_used) - 
        (int(configs['runtime']['femnist_eavesdropper_num']) + int(configs['runtime']['femnist_trusted_num'])),
        eavesdroppers_idx + trusted_idx)
        
    logging.info("Trusted [{}]: {}".format(len(trusted_idx), trusted_idx))
    logging.info("Eavesdroppers [{}]: {}".format(len(eavesdroppers_idx), eavesdroppers_idx))
    logging.info("Normal [{}]: {}".format(len(normal_idx), normal_idx))
    if log_enable:
        utils.write_to_file(output_dir, "all_users", workers_idx_all)
        utils.write_to_file(output_dir, "eavesdroppers", eavesdroppers_idx)
        utils.write_to_file(output_dir, "normal", normal_idx)
        utils.write_to_file(output_dir, "trusted", trusted_idx)
        
    # Create test dataloader from all normal and eveasdroppers
    test_dataset = fl.create_femnist_dataset(
        raw_test_data, workers_idx_to_be_used)
    logging.info("Aggregated test dataset: len: {}".format(len(test_dataset)))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=configs['runtime']['test_batch_size'],
        shuffle=True,
        drop_last=True)
    logging.info("Aggregated test dataloader: Batch Num: {}, Total samples: {}".format(
            len(test_dataloader), len(test_dataloader) * test_dataloader.batch_size))
            
    # Train on 25 users.
    # Test on the 30 users dataset.

    fed_train_dataset = fl.create_femnist_fed_dataset(raw_train_data, normal_idx, 75)
    fed_train_dataloader = sy.FederatedDataLoader(
            fed_train_dataset, batch_size=configs['runtime']['server_w0_batch_size'], 
            shuffle=True, drop_last=True)
            
    # if arguments["--not-pure"]:
    #     # Performing attack on the dataset before sending to the server
    #     logging.info("Performing attack on the dataset before sending to the server")
    #     train_dataset = utils.perfrom_attack(
    #         train_dataset, 
    #         int(arguments["--attack"]), 
    #         configs['runtime']['mnist_eavesdropper_num'], 
    #         configs['runtime']['mnist_workers_num'], 
    #         100)

    for round_no in range(rounds_num):
        fl.train_server(fed_train_dataloader, round_no, epochs_num)
        # Apply the server model to the test dataset
        fl.test(fl.server_model, test_dataloader, "server", round_no)
        fl.save_model(
            fl.server_model, 
            "{}_{}".format("server_model", round_no))
        print("")
    
