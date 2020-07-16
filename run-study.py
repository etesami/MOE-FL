"""
Usage: run-study.py \n\t\t(--avg | --opt) \n\t\t--attack=<attack-type> \n\t\t--mal-workers=<list-of-uesr-ids> \n\t\t[--data-percentage=<data-percentage>]  \n\t\t--output-file=<output-filename>
"""
from docopt import docopt
from federated_learning.FederatedLearning import FederatedLearning
import logging, ast, torch
from FLNet import FLNet
import numpy as np
arguments = docopt(__doc__)

# '''
# 01_avg: average, normal situation
# 02_opt: wieghted average, normal situation
# 03_att1_avg: average, attack 1
# 04_att1_opt: wieghted average, attack 1
# '''

# print(arguments)

OUTPUT_PATH_PREFIX = "data_tmp/"
MNIST_PATH = "/home/ubuntu/data/MNIST"
WORKERS_NUM = 10
EPOCH_NUM = 2

fl = FederatedLearning(
        workers_num = WORKERS_NUM, 
        epochs_num = EPOCH_NUM, 
        output_prefix = OUTPUT_PATH_PREFIX + arguments['--output-file'],
        mnist_path = MNIST_PATH, 
        log_level = logging.DEBUG)

fl.create_workers()
fl.create_server()
fl.load_data()

count = {}
for i in range(0,10):
    count[i] = 0
for d in fl.train_labels:
    count[d] = count[d] + 1

logging.info("Percentage of digits in whole training dataset: {}".format([round(d*100.0/len(fl.train_labels),2) for _, d in count.items()]))

logging.debug(fl.train_labels[6000:6025])
mal_users_list = ast.literal_eval(arguments['--mal-workers'])
data_percentage = int(arguments['--data-percentage'])
if arguments['--attack'] == "1":
    fl.attack_permute_labels_randomly(mal_users_list)
elif arguments['--attack'] == "2":
    a = fl.attack_permute_labels_collaborative(mal_users_list, 40)

server_data_loader = fl.create_aggregated_data()
train_data_loader, test_data_loader = fl.create_datasets()

fl.create_server_model()
fl.create_workers_model()

for epoch in range(1, EPOCH_NUM + 1):
    fl.train_server(server_data_loader, epoch)
    fl.train_workers(train_data_loader, epoch)
    print()
    # W = fl.find_best_weights(epoch)
    W = [0.1] * 10

    # base model is meant nothing in this scenario
    fl.update_models(W, fl.server_model, fl.workers_model)

    # Apply the server model to the test dataset
    fl.test(fl.server_model, test_data_loader, epoch)


