"""
Usage: 
    run-study-iid.py\n\t\t(--avg | --opt)\n\t\t--no-attack --output-prefix=NAME [--log] [--nep-log]
    run-study-iid.py\n\t\t(--avg | --opt)\n\t\t--attack=ATTACK-TYPE --output-prefix=NAME [--log] [--nep-log]
"""
from docopt import docopt
import logging
import random
import neptune
import numpy as np
import syft as sy
from torch import load
from torchvision import transforms
from federated_learning.FLCustomDataset import FLCustomDataset
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
# arguments['--output-prefix'] = "mnist_attack_2_opt"
arguments['--server_model'] = "data_output/20201108_225254_mnist_w0/models/server_model_49"
# arguments['--local-log'] = "True"
# arguments['--neptune-log'] = "True"
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
    workers_idx = ["worker_" + str(i) for i in range(configs['runtime']['mnist_workers_num'])]
    fl.create_workers(workers_idx)
    fl.create_workers_model(workers_idx)

    train_raw_dataset = utils.preprocess_mnist(
        utils.load_mnist_data_train(
            configs['data']['MNIST_PATH'], 
            configs['runtime']['mnist_data_percentage']))
    train_dataset = utils.get_mnist_dataset(train_raw_dataset)
    # train_dataloader = utils.get_dataloader(
    #     train_raw_dataset, configs['runtime']['batch_size'])

    test_data = utils.load_mnist_data_test(configs['data']['MNIST_PATH'])
    test_dataset = utils.get_mnist_dataset(test_data)
    test_dataloader = utils.get_dataloader(
        test_dataset, configs['runtime']['test_batch_size'], shuffle=True)
    
    # Federated dataset is not used here. There is no server training
    trained_server_model = load(arguments['--server_model'])
    # server_dataset = utils.get_server_mnist_dataset(
    #     train_dataloader, configs['runtime']['public_data_percentage'])
    # federated_server_dataloader = fl.create_federated_mnist(
    #     server_dataset, ["server"], configs['runtime']['batch_size'], shuffle=False)

    federated_train_dataloader = None
    if arguments["--no-attack"]:
        logging.info("No Attack will be performed.")
        federated_train_dataloader = fl.create_federated_mnist(
            train_dataset, workers_idx, configs['runtime']['batch_size'], shuffle=False)
    else:
        logging.info("Perform attack type: {}".format(arguments["--attack"]))
        eavesdroppers_idx = utils.get_eavesdroppers_idx(len(workers_idx), configs['runtime']['mnist_eavesdropper_num'])
        if log_enable:
            file = open(output_dir + "eavesdroppers", "a")
            file.write('{}'.format(eavesdroppers_idx))
            file.close()
        federated_train_dataloader = fl.create_federated_mnist(
            utils.perfrom_attack(
                train_dataset, 
                int(arguments["--attack"]), 
                len(workers_idx), 
                eavesdroppers_idx, 
                100), 
            workers_idx, configs['runtime']['batch_size'], shuffle=False)
        
    for round_no in range(rounds_num):
        fl.train_workers(federated_train_dataloader, workers_idx, round_no, epochs_num)
        
        # Find the best weights and update the server model
        wieghts, mode = None, None
        if arguments['--avg']:
            wieghts = [1.0 / float(configs['runtime']['mnist_workers_num'])] * int(configs['runtime']['mnist_workers_num'])
            mode = "AVG"
        elif arguments['--opt']:
            wieghts = fl.find_best_weights(trained_server_model, workers_idx)
            mode = "OPT"
        
        fl.save_workers_model(workers_idx, str(round_no))

        # Update the server model
        fl.update_models(
            configs['runtime']['alpha'],
            wieghts, 
            workers_idx,
            workers_update=True,
            server_update=True)

        # Apply the server model to the test dataset
        fl.test(fl.server_model, test_dataloader, round_no)

        fl.save_model(
            fl.server_model, 
            "{}_{}".format("server_model", round_no))

        print("")
    