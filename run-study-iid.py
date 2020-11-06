"""
Usage: 
    run-study-iid.py --epoch=<epoch-num>\n\t\t--round=<round-num>\n\t\t(--avg | --opt)
"""
from docopt import docopt
import logging
import random
import torch
import neptune
import numpy as np
import syft as sy
from torchvision import transforms
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.FederatedLearning import FederatedLearning
from federated_learning.helper import utils
arguments = docopt(__doc__)
CONFIG_PATH = 'configs/defaults.yml'
MNIST_PATH = "/home/ubuntu/data/MNIST"

# arguments = docopt(__doc__)
############ TEMPORARILY ################
# arguments = dict()
arguments['--reg'] = 0.0
arguments['--output-prefix'] = "test_"
arguments['--local-log'] = False
arguments['--neptune-log'] = False
neptune_enable = False
log_enable = False
############ TEMPORARILY ################


if __name__ == '__main__':
    configs = utils.load_config(CONFIG_PATH)

    # From config file
    train_workers_num = configs['runtime']['train_workers_num']
    emnist_google_path = configs['data']['EMNIST_GOOGLE_PATH']
    model_path = configs['runtime']['model_path']
    model_path_ = "{}{}".format(configs['runtime']['model_path'], "server_model_7")
    # trained_server_model = torch.load(model_path_)

    # From command line
    epochs_num = int(arguments["--epoch"])
    rounds_num = int(arguments["--round"])
    # reg = float(arguments['--reg']) if arguments['--reg'] is not None else 0.0
    # output_file = arguments['--output-file']
    log_enable = True if arguments['--local-log'] == "True" or \
        arguments['--local-log'] == "true" else False
    neptune_enable = True if arguments['--neptune-log'] == "True" or \
        arguments['--neptune-log'] == "true" else False
    # output_prefix = arguments['--output-prefix']
    # percentage = float(arguments['--percentage'])

    # model_path = model_path + output_prefix

    fl = FederatedLearning(
        configs['runtime']['batch_size'], 
        configs['runtime']['test_batch_size'], 
        configs['runtime']['lr'], 
        float(arguments['--reg']) if arguments['--reg'] is not None else 0.0,
        configs['runtime']['momentum'], 
        neptune_enable, log_enable, 
        configs['log']['interval'], 
        configs['log']['level'], 
        configs['log']['output_dir'], 
        arguments['--output-prefix'], 
        configs['runtime']['random_seed'], 
        configs['runtime']['save_model'])

    # Neptune logging, initialization
    if neptune_enable:
        neptune.init(configs['log']['neptune_init'])
        neptune.create_experiment(name = configs['log']['neptune_exp'])

    train_data = utils.load_mnist_data_train(MNIST_PATH)
    train_data = utils.preprocess_mnist(train_data)
    test_data = utils.load_mnist_data_test(MNIST_PATH)
    # train_dataloader = utils.get_mnist_dataloader(fl.train_data, configs['runtime']['batch_size'])
    test_dataloader = utils.get_mnist_dataloader(test_data, configs['runtime']['test_batch_size'])

    workers_idx = ["worker_" + str(i) for i in range(configs['runtime']['mnist_workers_num'])]
    fl.create_workers(workers_idx)
    fl.create_server()
    fl.create_workers_model(workers_idx)
    fl.create_server_model()

    fed_train_dataloader = fl.create_federated_mnist(train_data, configs['runtime']['batch_size'], True)
    for round_no in range(rounds_num):
        fl.train_workers(fed_train_dataloader, workers_idx, round_no, epochs_num)
        wieghts = None
        mode = None
        if arguments['--avg']:
            wieghts = [0.1] * int(configs['runtime']['mnist_workers_num'])
            mode = "AVG"
        elif arguments['--opt']:
            wieghts = fl.find_best_weights(workers_idx)
            mode = "OPT"

        fl.update_models(
            configs['runtime']['alpha'],
            wieghts, 
            fl.server_model, 
            fl.workers_model, workers_idx)

        # Apply the server model to the test dataset
        fl.test(fl.server_model, test_dataloader, "[Server][{}]".format(mode), round_no)
    
