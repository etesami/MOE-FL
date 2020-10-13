"""
Usage: 
    run-study.py \n\t\t--server \n\t\t--epoch=<epoch-num>\n\t\t--round=<round-num>\n\t\t--local-log=<true/false> \n\t\t --neptune-log=<true/false>\n\t\t--output-file=<output-filename>
    run-study.py \n\t\t--clients \n\t\t--start=<start> \n\t\t--end=<end>\n\t\t--epoch=<epoch-num>\n\t\t--round=<round-num>\n\t\t--local-log=<true/false> \n\t\t --neptune-log=<true/false>\n\t\t--output-prefix=<output-prefix>
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
    epochs_num = int(arguments['--epoch'])
    rounds_num = int(arguments['--round'])
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
    output_prefix = arguments['--output-prefix']
    log_level = configs['log']['level']
    neptune_init = configs['log']['neptune_init']
    neptune_name = configs['log']['neptune_exp']
    emnist_google_path = configs['data']['EMNIST_GOOGLE_PATH']
    

    fl = FederatedLearning(batch_size, test_batch_size, lr, momentum, neptune_enable, log_enable, log_interval, log_level, output_dir, output_prefix, random_seed, save_model)

    if neptune_enable:
        neptune.init(neptune_init)
        neptune.create_experiment(name = neptune_name)

    fl.load_femnist_train_digits(emnist_google_path)
    fl.load_femnist_test_digits(emnist_google_path)
    logging.info("Total workers size: {}".format(len(fl.workers_id)))

    random.seed(random_seed)

    if arguments['--server']:
        fl.create_server()
        # [1, 3, 6, 9]
        workers_to_be_used_idx = random.sample(range(len(fl.workers_id)), train_workers_num)
        # ['f_353', 'f_345']
        workers_to_be_used = [fl.workers_id[i] for i in workers_to_be_used_idx]
        fl.create_workers(workers_to_be_used)

        logging.debug("Some sample train labels for user {}: {}".format(workers_to_be_used[0], fl.train_data[workers_to_be_used[0]]['y'][0:10]))

        fl.create_server_model()
        fl.create_workers_model(workers_to_be_used)

        server_data_loader = fl.create_aggregated_data(workers_to_be_used)
        train_data_loader, test_data_loader = fl.create_datasets(workers_to_be_used)

        for round_no in range(0, rounds_num):
            fl.train_server(server_data_loader, round_no, epochs_num)
            test_acc = fl.test(fl.server_model, test_data_loader, "server", round_no)
            logging.info("Saving the server model...")
            torch.save(fl.server_model, model_path + "_" + str(round_no))
            if test_acc >= 98.0:
                logging.info("Test accuracy is {}. Stopping the experiment...".format(test_acc))
                break

    elif arguments['--clients']:
        start_idx = int(arguments['--start']) 
        end_idx = int(arguments['--end']) 
        worker_idx = start_idx
        
        while worker_idx <= end_idx:
            # contains sth like this ['f_353']
            workers_to_be_used = [fl.workers_id[worker_idx]]
            
            fl.create_workers(workers_to_be_used)

            logging.debug("Some sample train labels for user {}: {}".format(workers_to_be_used[0], fl.train_data[workers_to_be_used[0]]['y'][0:10]))
            fl.create_workers_model(workers_to_be_used)
            train_data_loader, test_data_loader = fl.create_datasets(workers_to_be_used)

            test_acc = 0
            for round_no in range(0, rounds_num):
                fl.train_workers(train_data_loader, workers_to_be_used, round_no, epochs_num)
                test_acc = fl.test(fl.workers_model[fl.workers_id[worker_idx]], test_data_loader, fl.workers_id[worker_idx], round_no)
                model_path_ = model_path + str(fl.workers_id[worker_idx]) + "_" + str(round_no) + "_" + str(round(test_acc,2))
                logging.info("Saving the worker model: {} ...".format(model_path_))
                if test_acc >= 98.5:
                    logging.info("Test accuracy is {}. Stopping the experiment...\n".format(test_acc))
                    break
            if neptune_enable:
                neptune.log_metric("accuracy_overal", test_acc)
            torch.save(fl.workers_model[fl.workers_id[worker_idx]], model_path_)
            worker_idx += 1
            
    else:
        raise Exception("Wrong arguments!")


