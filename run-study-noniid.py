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

    fl.create_server()
    fl.create_server_model()

    raw_train_data = utils.preprocess_leaf_data(
        utils.load_leaf_train(configs['data']['FEMNIST_PATH'])
    )
    raw_test_data = utils.preprocess_leaf_data(
        utils.load_leaf_test(configs['data']['FEMNIST_PATH']), min_num_samples=configs['runtime']['batch_size']
    )

    workers_idx_all = list(raw_train_data.keys())
    workers_idx_to_be_used = utils.get_workers_idx(
        workers_idx_all,
        configs['runtime']['femnist_users_num'],
        []
    )

    fl.create_workers(workers_idx_to_be_used)
    fl.create_workers_model(workers_idx_to_be_used)

    # trusted_idx = utils.get_workers_idx(
    #     range(total_num_workers), configs['runtime']['mnist_trusted_users_num'], [])
    eavesdroppers_idx = utils.get_workers_idx(
                            workers_idx_to_be_used, configs['runtime']['femnist_eavesdropper_num'], [])
    normal_idx = utils.get_workers_idx(
                            workers_idx_to_be_used, 
                            len(workers_idx_to_be_used) - int(configs['runtime']['femnist_eavesdropper_num']),
                            eavesdroppers_idx)
        
    # logging.info("Trusted: {}".format(trusted_idx))
    logging.info("Eavesdroppers: {}".format(eavesdroppers_idx))
    logging.info("Normal: {}".format(normal_idx))
    if log_enable:
        utils.write_to_file(output_dir, "all_users", workers_idx_all)
        utils.write_to_file(output_dir, "eavesdroppers", eavesdroppers_idx)
        utils.write_to_file(output_dir, "normal", normal_idx)
        
    test_dataloader = fl.create_femnist_server_test_dataloader(raw_test_data, workers_idx_to_be_used)
    
    # W0 model
    trained_w0_model = load(configs['runtime']['W0_pure_path'])

    # federated_train_dataloader = None
    # if arguments["--no-attack"]:
    #     logging.info("No Attack will be performed.")
    #     federated_train_dataloader = fl.create_federated_mnist(
    #         train_dataset, workers_idx, configs['runtime']['batch_size'], shuffle=False)
    # else:
    #     logging.info("Perform attack type: {}".format(arguments["--attack"]))
    #     federated_train_dataloader = fl.create_federated_mnist(
    #         utils.perfrom_attack(
    #             train_dataset, 
    #             int(arguments["--attack"]), 
    #             workers_idx, 
    #             eavesdroppers_idx, 
    #             100), 
    #         workers_idx, configs['runtime']['batch_size'], shuffle=False)
        
    # for round_no in range(rounds_num):
    #     fl.train_workers(federated_train_dataloader, workers_idx, round_no, epochs_num)
        
    #     # Find the best weights and update the server model
    #     wieghts, mode = None, None
    #     if arguments['--avg']:
    #         wieghts = [1.0 / float(configs['runtime']['mnist_workers_num'])] * int(configs['runtime']['mnist_workers_num'])
    #         mode = "AVG"
    #     elif arguments['--opt']:
    #         wieghts = fl.find_best_weights(trained_server_model, workers_idx)
    #         mode = "OPT"
        
    #     fl.save_workers_model(workers_idx, str(round_no))

    #     # Update the server model
    #     fl.update_models(
    #         configs['runtime']['alpha'],
    #         wieghts, 
    #         workers_idx,
    #         workers_update=True,
    #         server_update=True)

    #     # Apply the server model to the test dataset
    #     fl.test(fl.server_model, test_dataloader, round_no)

    #     fl.save_model(
    #         fl.server_model, 
    #         "{}_{}".format("server_model", round_no))

    #     print("")
    