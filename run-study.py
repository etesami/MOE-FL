"""
Usage: run-study.py \n\t\t(--avg | --opt) \n\t\t--attack=<attack-type> \n\t\t[--workers-percentage=<workers-percentage>] \n\t\t[--data-percentage=<data-percentage>]  \n\t\t--output-file=<output-filename>
"""
from docopt import docopt
from federated_learning.FederatedLearning import FederatedLearning
import logging, ast
arguments = docopt(__doc__)

# print(arguments)

OUTPUT_PATH_PREFIX = "data_tmp/"
MNIST_PATH = "/home/ubuntu/data/MNIST"
WORKERS_NUM = 10
EPOCH_NUM = 5

fl = FederatedLearning(
        workers_num = WORKERS_NUM, 
        epochs_num = EPOCH_NUM, 
        output_prefix = OUTPUT_PATH_PREFIX + arguments['--output-file'],
        mnist_path = MNIST_PATH, 
        log_level = logging.INFO)

fl.create_workers()
fl.create_server()
fl.load_data()

# Count the number of digits
# fl.count_digits()

logging.debug("Some sample data for user 1: {}".format(fl.train_labels[6000:6025]))

if arguments['--attack'] == "1":
    mal_users_list = ast.literal_eval(arguments['--workers-percentage'])
    fl.attack_permute_labels_randomly(mal_users_list)
elif arguments['--attack'] == "2":
    mal_users_list = ast.literal_eval(arguments['--workers-percentage'])
    data_percentage = int(arguments['--data-percentage'])
    a = fl.attack_permute_labels_collaborative(mal_users_list, 40)
else:
    logging.info("** No attack is performed on the dataset.")

server_data_loader = fl.create_aggregated_data()
train_data_loader, test_data_loader = fl.create_datasets()

fl.create_server_model()
fl.create_workers_model()

for epoch in range(1, EPOCH_NUM + 1):
    fl.train_server(server_data_loader, epoch)
    fl.train_workers(train_data_loader, epoch)
    print()
    W = None
    if arguments['--avg']:
        W = [0.1] * 10
    elif arguments['--opt']:
        W = fl.find_best_weights(epoch)
    else:
        logging.error("Not expected this mode!")
        exit(1)
    
    # base model is meant nothing in this scenario
    fl.update_models(W, fl.server_model, fl.workers_model)

    # Apply the server model to the test dataset
    fl.test(fl.server_model, test_data_loader, epoch)


