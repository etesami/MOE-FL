"""
Usage: 
    run-study-iid.py --epoch=<epoch-num>\n\t\t--round=<round-num>
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
MNIST_PATH = "/home/ubuntu/data/MNIST"

# arguments = docopt(__doc__)
############ TEMPORARILY ################
# arguments = dict()
arguments['--reg'] = 0.0
arguments['--output-prefix'] = "mnist_w0"
arguments['--local-log'] = "True"
arguments['--neptune-log'] = False
############ TEMPORARILY ################


if __name__ == '__main__':
    configs = utils.load_config(CONFIG_PATH)
    logging.basicConfig(format='%(asctime)s %(message)s', level=configs['log']['level'])
    random.seed(configs['runtime']['random_seed'])

    # From command line
    epochs_num = int(arguments["--epoch"])
    rounds_num = int(arguments["--round"])

    log_enable = True if arguments['--local-log'] == "True" or \
        arguments['--local-log'] == "true" else False
    
    output_dir = None
    if log_enable:
        output_dir = utils.make_output_dir(
            configs['log']['root_output_dir'], arguments['--output-prefix'])
                
    neptune_enable = True if arguments['--neptune-log'] == "True" or \
        arguments['--neptune-log'] == "true" else False

    # From config file
    train_workers_num = configs['runtime']['train_workers_num']

    fl = FederatedLearning(
        configs['runtime']['batch_size'], 
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

    train_data = utils.preprocess_mnist(
        utils.load_mnist_data_train(MNIST_PATH))
    train_dataloader = utils.get_mnist_dataloader(
        train_data, configs['runtime']['batch_size'])

    test_data = utils.load_mnist_data_test(MNIST_PATH)
    test_dataloader = utils.get_mnist_dataloader(
        test_data, configs['runtime']['test_batch_size'])
    
    server_dataset = utils.get_server_dataloader(
        train_dataloader, configs['runtime']['public_data_percentage'])

    federated_server_dataloader = fl.create_federated_mnist(
        server_dataset, ["server"], configs['runtime']['batch_size'], shuffle=False)

    for round_no in range(rounds_num):
        fl.train_server(federated_server_dataloader, round_no, epochs_num)
        # Apply the server model to the test dataset
        fl.test(fl.server_model, test_dataloader, round_no)
        fl.save_model(
            fl.server_model, 
            "{}/{}_{}".format(output_dir, "server_model", round_no))
        print("")
    
