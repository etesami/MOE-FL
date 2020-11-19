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
from time import strftime
from federated_learning.FederatedLearning import FederatedLearning
from federated_learning.helper import utils
arguments = docopt(__doc__)
CONFIG_PATH = 'configs/defaults.yml'

# arguments = docopt(__doc__)
############ TEMPORARILY ################
# arguments = dict()
arguments['--reg'] = 0.0
arguments['--epoch'] = 5 
arguments['--round'] = 100 
############ TEMPORARILY ################


if __name__ == '__main__':
    configs = utils.load_config(CONFIG_PATH)
    logging.basicConfig(format='%(asctime)s %(message)s', level=configs['log']['level'])
    random.seed(configs['runtime']['random_seed'])

    # From command line
    epochs_num = int(arguments["--epoch"])
    rounds_num = int(arguments["--round"])

    log_enable = True if arguments['--log'] else False
    
    output_dir = None
    if log_enable:
        output_dir = utils.make_output_dir(
            configs['log']['root_output_dir'], arguments['--output-prefix'])
                
    neptune_enable = True if arguments['--nep-log'] else False

    fl = FederatedLearning(
        configs['runtime']['server_w0_batch_size'], 
        configs['runtime']['test_batch_size'], 
        configs['runtime']['lr'], 
        float(arguments['--reg']) if arguments['--reg'] is not None else 0.0,
        configs['runtime']['momentum'], 
        neptune_enable, log_enable, 
        configs['log']['interval'], 
        output_dir, 
        configs['runtime']['random_seed'])

    # Neptune logging, initialization
    if neptune_enable:
        neptune.init(configs['log']['neptune_init'])
        neptune.create_experiment(name = configs['log']['neptune_exp'])

    fl.create_server()
    fl.create_server_model()


    raw_data = utils.preprocess_leaf_data(
        utils.load_leaf_train(configs['data']['FEMNIST_PATH']),
        min_num_samples=100,only_digits=True)


    logging.info("Select data from only {} workers to be used...".format(configs['runtime']['femnist_total_users_num']))
    workers_idx = list(raw_data.keys())[:configs['runtime']['femnist_total_users_num']]
    logging.info(len(workers_idx))

    server_train_images, server_train_labels = fl.create_server_femnist_dataset(
        raw_data, workers_idx, configs['runtime']['public_data_percentage'])
    server_federated_train_dataloader = fl.create_federated_server_leaf_dataloader(
        server_train_images, server_train_labels, configs['runtime']['server_w0_batch_size'], True)

    test_raw_data = utils.preprocess_leaf_data(
        utils.load_leaf_test(configs['data']['FEMNIST_PATH']),
        min_num_samples=0,only_digits=True)
    server_test_images, server_test_labels = fl.create_server_femnist_dataset(
        test_raw_data, workers_idx, 100)

    server_test_dataloader = fl.create_federated_server_leaf_dataloader(
        server_test_images, server_test_labels, configs['runtime']['test_batch_size'], True)
    
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
        fl.train_server(server_federated_train_dataloader, round_no, epochs_num)
        # Apply the server model to the test dataset
        fl.test(fl.server_model, server_federated_test_dataloader, round_no)
        fl.save_model(
            fl.server_model, 
            "{}_{}".format("server_model", round_no))
        print("")
    
