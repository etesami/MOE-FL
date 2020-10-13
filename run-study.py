"""
Usage: 
    run-study.py \n\t\t--model-id=<model-id>\n\t\t--local-log=<true/false> \n\t\t --neptune-log=<true/false>\n\t\t--output-prefix=<output-filename>
    run-study.py \n\t\t--local-log=<true/false> \n\t\t --neptune-log=<true/false>\n\t\t--output-prefix=<output-prefix>
    run-study.py \n\t\t--attack\n\t\t--local-log=<true/false>\n\t\t--neptune-log=<true/false>\n\t\t--output-prefix=<output-prefix>
"""
from docopt import docopt
from federated_learning.FederatedLearning import FederatedLearning
import logging
import random
import torch
import ast
import sys
import yaml
import numpy as np
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


def get_distance(model, server_model):
    distance = 0
    distance += np.linalg.norm(model.conv1.weight.data - server_model.conv1.weight.data)
    distance += np.linalg.norm(model.conv1.bias.data - server_model.conv1.bias.data)
    distance += np.linalg.norm(model.conv2.weight.data - server_model.conv2.weight.data)
    distance += np.linalg.norm(model.conv2.bias.data - server_model.conv2.bias.data)
    distance += np.linalg.norm(model.fc1.weight.data - server_model.fc1.weight.data)
    distance += np.linalg.norm(model.fc1.bias.data - server_model.fc1.bias.data)
    distance += np.linalg.norm(model.fc2.weight.data - server_model.fc2.weight.data)
    distance += np.linalg.norm(model.fc2.bias.data - server_model.fc2.bias.data)
    return distance

def get_models_list(file_path):
    with open(file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

if __name__ == '__main__':
    
    configs = loadConfig(CONFIG_PATH)
    # epochs_num = int(arguments['--epoch'])
    # rounds_num = int(arguments['--round'])
    batch_size = configs['runtime']['batch_size']
    test_batch_size = configs['runtime']['test_batch_size']
    lr = configs['runtime']['lr']
    momentum = configs['runtime']['momentum']
    save_model = configs['runtime']['save_model']
    model_path = configs['runtime']['model_path']
    random_seed = configs['runtime']['random_seed']
    train_workers_num = configs['runtime']['train_workers_num']
    
    # output_file = arguments['--output-file']
    log_enable = True if arguments['--local-log'] == "True" or \
        arguments['--local-log'] == "true" else False
    neptune_enable = True if arguments['--neptune-log'] == "True" or \
        arguments['--neptune-log'] == "true" else False

    log_interval = configs['log']['interval']
    output_dir = configs['log']['output_dir']
    output_prefix = arguments['--output-prefix']
    log_file_path = output_dir + "/" + output_prefix
    log_level = configs['log']['level']
    neptune_init = configs['log']['neptune_init']
    neptune_name = configs['log']['neptune_exp']
    emnist_google_path = configs['data']['EMNIST_GOOGLE_PATH']
    
    model_path_ = "{}{}".format(model_path, "server_model_7")
    trained_server_model = torch.load(model_path_)
    # print("{} {}".format("server", get_distance(trained_server_model, trained_server_model)))


    fl = FederatedLearning(batch_size, test_batch_size, lr, momentum, neptune_enable, log_enable, log_interval, log_level, output_dir, output_prefix, random_seed, save_model)
    
    fl.load_femnist_train_digits(emnist_google_path)
    fl.load_femnist_test_digits(emnist_google_path)
    logging.info("Total workers size: {}".format(len(fl.workers_id)))

    random.seed(random_seed)

    fl.create_server()
    fl.create_server_model()

    # test_data_loader = fl.create_test_dataset()
    
    if log_enable:
        file = open(log_file_path + "_models_distance", "a")
    if neptune_enable:
        neptune.init(neptune_init)
        neptune.create_experiment(name = neptune_name)

    bad_workers_idx = None
    if arguments['--attack']:
        bad_workers_idx = fl.attack_permute_labels_randomly(1, 100)
        print(bad_workers_idx)
    else:
        model_path_ = "{}{}".format(model_path, "models_list")
        trained_models_list = get_models_list(model_path_)
        for model_ in trained_models_list:
            model_path_ = "{}{}".format(model_path, model_)
            trained_model = torch.load(model_path_)
            distance = round(get_distance(trained_model, trained_server_model),2)
            TO_FILE = '{} {}\n'.format(model_, distance)
            print("{} {}".format(model_, distance))
            if log_enable:
                file.write(TO_FILE)
            if neptune_enable:
                neptune.log_metric('distances', distance)
    
    if log_enable:
        file.close()

    

    
