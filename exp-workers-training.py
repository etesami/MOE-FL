"""
Usage: 
    exp-workers-training.py\n\t\t(--avg | --opt)\n\t\t--no-attack --output-prefix=NAME [--log] [--nep-log]
    exp-workers-training.py\n\t\t(--avg | --opt)\n\t\t--attack=ATTACK-TYPE --output-prefix=NAME [--log] [--nep-log]
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
        configs['runtime']['batch_size'], 
        configs['runtime']['test_batch_size'], 
        configs['runtime']['lr'], 
        configs['runtime']['reg'],
        configs['runtime']['momentum'], 
        neptune_enable, log_enable, 
        configs['log']['interval'], 
        output_dir, 
        configs['runtime']['random_seed'])

    # fl.create_server()
    # fl.create_server_model()

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
    # fed_test_dataloader = fl.create_femnist_server_test_dataloader(
    #     raw_test_data, workers_idx_to_be_used)

    # W0 model
    # trained_w0_model = load(configs['runtime']['W0_pure_path'])
    fed_train_datasets = None
    if arguments["--no-attack"]:
        logging.info("No Attack will be performed.")
        fed_train_datasets = fl.create_femnist_fed_datasets(raw_train_data, workers_idx_to_be_used)
    elif arguments["--attack"] == "99": # Combines
        logging.info("Perform combined attacks 1, 2, 3")
        dataset = utils.perfrom_attack_femnist(
                raw_train_data, 1, workers_idx_to_be_used, eavesdroppers_idx)
        dataset = utils.perfrom_attack_femnist(
                dataset, 3, workers_idx_to_be_used, eavesdroppers_idx)
        fed_train_datasets = fl.create_femnist_fed_datasets(dataset, workers_idx_to_be_used)
    else:
        logging.info("Perform attack type: {}".format(arguments["--attack"]))
        fed_train_datasets = fl.create_femnist_fed_datasets(
            utils.perfrom_attack_femnist(
                raw_train_data, 
                int(arguments["--attack"]),
                workers_idx_to_be_used,
                eavesdroppers_idx
            ), workers_idx_to_be_used)

    fed_train_dataloaders = dict()
    for ww_id, fed_dataset in fed_train_datasets.items():
        dataloader = sy.FederatedDataLoader(
            fed_dataset, batch_size=configs['runtime']['batch_size'], shuffle=False, drop_last=True)
        fed_train_dataloaders[ww_id] = dataloader

    logging.info("Creating test dataset for each worker...")
    test_datasets = fl.create_femnist_datasets(raw_test_data, workers_idx_to_be_used)
    test_dataloaders = dict()
    for ww_id, test_dataset in test_datasets.items():
        dataloader = DataLoader(
            test_dataset, 
            batch_size=configs['runtime']['test_batch_size'], 
            shuffle=False, drop_last=True)
        test_dataloaders[ww_id] = dataloader
        
    for round_no in range(rounds_num):
        for counter, worker_id in enumerate(workers_idx_to_be_used):
            logging.info("Training worker {} out of {} workers...".format(
                counter+1, len(workers_idx_to_be_used)))
            fl.train_workers(fed_train_dataloaders[worker_id], [worker_id], round_no, epochs_num)
            fl.test(fl.workers_model[worker_id], test_dataloaders[worker_id], worker_id, round_no)

        # Find the best weights and update the server model
        # weights = None
        # if arguments['--avg']:
        #     weights = [1.0 / len(workers_idx_to_be_used)] * len(workers_idx_to_be_used)
        # elif arguments['--opt']:
        #     trusted_weights = [1.0 / len(trusted_idx)] * len(trusted_idx)
        #     avg_trusted_model = fl.wieghted_avg_model(trusted_weights, trusted_idx)
        #     weights = fl.find_best_weights(avg_trusted_model, normal_idx + eavesdroppers_idx)
        
        if log_enable:
            fl.save_workers_model(workers_idx_to_be_used, str(round_no))
            # fl.save_model(
            #     fl.get_average_model(trusted_idx),
            #     "R{}_{}".format(round_no, "avg_trusted_model")
            # )

        # weighted_avg_model = fl.wieghted_avg_model(weights, normal_idx + eavesdroppers_idx)
        # # Update the server model
        # fl.update_models(workers_idx_to_be_used, weighted_avg_model)

        # # Apply the server model to the test dataset
        # fl.test(weighted_avg_model, fed_test_dataloader, round_no)

        # if log_enable:
        #     fl.save_model(
        #         weighted_avg_model, 
        #         "R{}_{}".format(round_no, "weighted_avg_model")
        #     )

        print("")
    