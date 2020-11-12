"""
Usage: 
    run-study-iid.py --pure 
    run-study-iid.py --not-pure --attack=<ATTACK-TYPE>
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
arguments['--output-prefix'] = "mnist_w0_not_pure" 
arguments['--local-log'] = "True"
arguments['--neptune-log'] = "True"
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

    train_raw_dataset = utils.preprocess_mnist(
        utils.load_mnist_data_train(
            configs['data']['MNIST_PATH'], 
            configs['runtime']['mnist_data_percentage']))
    train_dataset = utils.get_mnist_dataset(train_raw_dataset)
    # train_dataloader = utils.get_dataloader(
    #     train_dataset, configs['runtime']['batch_size'], shuffle=True)

    test_data = utils.load_mnist_data_test(configs['data']['MNIST_PATH'])
    test_dataset = utils.get_mnist_dataset(test_data)
    test_dataloader = utils.get_dataloader(
        test_dataset, configs['runtime']['test_batch_size'], shuffle=True)
    
    if arguments["--not-pure"]:
        # Performing attack on the dataset before sending to the server
        logging.info("Performing attack on the dataset before sending to the server")
        train_dataset = utils.perfrom_attack(
            train_dataset, 
            int(arguments["--attack"]), 
            configs['runtime']['mnist_eavesdropper_num'], 
            configs['runtime']['mnist_workers_num'], 
            100)

    server_dataset = utils.get_server_mnist_dataset(
        train_dataset, 
        configs['runtime']['mnist_workers_num'],
        configs['runtime']['public_data_percentage'])
    
    federated_server_dataloader = fl.create_federated_mnist(
        server_dataset, ["server"], configs['runtime']['server_w0_batch_size'], shuffle=False)

    for round_no in range(rounds_num):
        fl.train_server(federated_server_dataloader, round_no, epochs_num)
        # Apply the server model to the test dataset
        fl.test(fl.server_model, test_dataloader, round_no)
        fl.save_model(
            fl.server_model, 
            "{}_{}".format("server_model", round_no))
        print("")
    
