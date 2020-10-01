"""
Usage: run-study.py \n\t\t(--avg | --opt) \n\t\t--attack=<attack-type> \n\t\t[--workers-percentage=<workers-percentage>] \n\t\t[--data-percentage=<data-percentage>]  \n\t\t--local-log=<true/false> \n\t\t --neptune-log=<true/false>\n\t\t--output-file=<output-filename>
"""
from docopt import docopt
from federated_learning.FederatedLearning import FederatedLearning
import logging
import random
import torch
import ast
import sys
import yaml
import os
arguments = docopt(__doc__)
import neptune

# print(arguments)

CONFIG_PATH = 'configs/defaults.yml'

def loadConfig(configPath):
    """ Load configuration files.

    Args:
        configPath (str): Path to the config file

    Returns:
        Obj: Corresponding python object
    """    
    configPathAbsolute = os.path.abspath(configPath)
    configs = None
    try:
        with open(configPathAbsolute, 'r') as f:
            configs = yaml.full_load(f)
    except FileNotFoundError:
        print("Config file does not exist.")
        exit(1)
    return configs

if __name__ == '__main__':
    
    configs = loadConfig(CONFIG_PATH)
    epochs_num = configs['runtime']['epochs_num']
    rounds_num = configs['runtime']['rounds_num']
    batch_size = configs['runtime']['batch_size']
    test_batch_size = configs['runtime']['test_batch_size']
    lr = configs['runtime']['lr']
    momentum = configs['runtime']['momentum']
    save_model = configs['runtime']['save_model']
    model_path = configs['runtime']['model_path']
    random_seed = configs['runtime']['random_seed']
    train_workers_num = configs['runtime']['train_workers_num']
    
    output_file = arguments['--output-file']
    log_enable = True if arguments['--local-log'] == "True" or \
        arguments['--local-log'] == "true" else False
    neptune_enable = True if arguments['--neptune-log'] == "True" or \
        arguments['--neptune-log'] == "true" else False

    log_interval = configs['log']['interval']
    output_dir = configs['log']['output_dir']
    output_prefix = configs['log']['output_prefix']
    log_level = configs['log']['level']
    neptune_init = configs['log']['neptune_init']
    neptune_name = configs['log']['neptune_exp']
    emnist_google_path = configs['data']['EMNIST_GOOGLE_PATH']
    

    fl = FederatedLearning(batch_size, test_batch_size, lr, momentum, neptune_enable, log_enable, log_interval, log_level, output_prefix, random_seed, save_model)

    if neptune_enable:
        neptune.init(neptune_init)
        neptune.create_experiment(name = neptune_name)

    fl.load_femnist_train_digits(emnist_google_path)
    fl.load_femnist_test_digits(emnist_google_path)
    logging.info("Total workers size: {}".format(len(fl.workers_id)))

    random.seed(random_seed)

    fl.create_server()

    # [1, 3, 6, 9]
    workers_to_be_used_idx = random.sample(range(len(fl.workers_id)), train_workers_num)
    # ['f_353', 'f_345']
    workers_to_be_used = [fl.workers_id[i] for i in workers_to_be_used_idx]
    fl.create_workers(workers_to_be_used)

    print("Some sample train labels for user {}: {}".format(workers_to_be_used[0], fl.train_data[workers_to_be_used[0]]['y'][0:10]))

    # test_data_loaders = {}
    fl.create_server_model()
    fl.create_workers_model(workers_to_be_used)

    server_data_loader = fl.create_aggregated_data(workers_to_be_used)
    train_data_loader, test_data_loader = fl.create_datasets(workers_to_be_used)


    for round_no in range(0, rounds_num):
        fl.train_server(server_data_loader, round_no, epochs_num)
        fl.test(fl.server_model, test_data_loader, "server")
    

    logging.info("Saving the server model...")
    torch.save(fl.server_model, model_path)


