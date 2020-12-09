"""
Usage: 
    run-study-iid.py\n\t\t(--avg | --opt)\n\t\t--no-attack --output-prefix=NAME [--log] [--nep-log]
    run-study-iid.py\n\t\t(--avg | --opt)\n\t\t--attack=ATTACK-TYPE --output-prefix=NAME [--log] [--nep-log]
"""
from docopt import docopt
import logging
import neptune
import numpy as np
import syft as sy
from torch import load
from random import seed, sample
from torchvision import transforms
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.FederatedLearning import FederatedLearning
from federated_learning.helper import utils
arguments = docopt(__doc__)
CONFIG_PATH = 'configs/defaults.yml'

############ TEMPORARILY ################
# arguments = dict()
############ TEMPORARILY ################


if __name__ == '__main__':
    configs = utils.load_config(CONFIG_PATH)
    logging.basicConfig(format='%(asctime)s %(message)s', level=configs['log']['level'])
    seed(configs['runtime']['random_seed'])

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

    fl.create_server()
    fl.create_server_model()
    
    total_num_workers = \
        configs['runtime']['mnist_normal_users_num'] + \
        configs['runtime']['mnist_eavesdropper_num'] 
        # configs['runtime']['mnist_trusted_users_num']

    workers_idx = ["worker_" + str(i) for i in range(total_num_workers)]
    fl.create_workers(workers_idx)
    fl.create_workers_model(workers_idx)

    # trusted_idx = utils.get_workers_idx(
    #     range(total_num_workers), configs['runtime']['mnist_trusted_users_num'], [])
    eavesdroppers_idx = utils.get_workers_idx(
        range(total_num_workers), configs['runtime']['mnist_eavesdropper_num'], [])
    normal_idx = utils.get_workers_idx(
        range(total_num_workers), configs['runtime']['mnist_normal_users_num'], eavesdroppers_idx)

    # trusted_idx = [workers_idx[ii] for ii in trusted_idx]
    eavesdroppers_idx = [workers_idx[ii] for ii in eavesdroppers_idx]
    normal_idx = [workers_idx[ii] for ii in normal_idx]

    # logging.info("Trusted: {}".format(trusted_idx))
    logging.info("Eavesdroppers: {}".format(eavesdroppers_idx))
    logging.info("Normal: {}".format(normal_idx))
    if log_enable:
        # utils.write_to_file(output_dir, "trusted", trusted_idx)
        utils.write_to_file(output_dir, "eavesdroppers", eavesdroppers_idx)
        utils.write_to_file(output_dir, "normal", normal_idx)

    train_raw_dataset = utils.preprocess_mnist(
        utils.load_mnist_data_train(
            configs['data']['MNIST_PATH'], 
            configs['runtime']['mnist_data_percentage']))
    train_dataset = utils.get_mnist_dataset(train_raw_dataset)

    test_data = utils.load_mnist_data_test(configs['data']['MNIST_PATH'])
    test_dataset = utils.get_mnist_dataset(test_data)
    test_dataloader = utils.get_dataloader(
        test_dataset, configs['runtime']['test_batch_size'], shuffle=True, drop_last=False)
    
    # W0 model
    trained_w0_model = load(configs['runtime']['W0_pure_path'])

    federated_train_dataloader = None
    if arguments["--no-attack"]:
        logging.info("No Attack will be performed.")
        federated_train_dataloader = fl.create_federated_mnist(
            train_dataset, workers_idx, configs['runtime']['batch_size'], shuffle=False)
    else:
        logging.info("Perform attack type: {}".format(arguments["--attack"]))
        federated_train_dataloader = fl.create_federated_mnist(
            utils.perfrom_attack(
                train_dataset, 
                int(arguments["--attack"]), 
                workers_idx, 
                eavesdroppers_idx, 
                100), 
            workers_idx, configs['runtime']['batch_size'], shuffle=False)
        
    for round_no in range(rounds_num):
        fl.train_workers(federated_train_dataloader, workers_idx, round_no, epochs_num)
        
        # Find the best weights and update the server model
        weights = None
        if arguments['--avg']:
            weights = [1.0 / len(workers_idx)] * len(workers_idx)
        elif arguments['--opt']:
            # weights = fl.find_best_weights(trained_server_model, workers_idx)
            weights = fl.find_best_weights(trained_w0_model, workers_idx)
        
        if log_enable:
            fl.save_workers_model(workers_idx, str(round_no))
            # fl.save_model(
            #     fl.get_average_model(trusted_idx),
            #     "R{}_{}".format(round_no, "avg_trusted_model")
            # )

        weighted_avg_model = fl.wieghted_avg_model(weights, workers_idx)
        # Update the server model
        fl.update_models(workers_idx, weighted_avg_model)

        # Apply the server model to the test dataset
        fl.test(weighted_avg_model, test_dataloader, round_no)

        if log_enable:
            fl.save_model(
                weighted_avg_model, 
                "R{}_{}".format(round_no, "weighted_avg_model")
            )

        print("")
    