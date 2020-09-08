"""
Usage: run-study.py \n\t\t(--avg | --opt) \n\t\t--attack=<attack-type> \n\t\t[--workers-percentage=<workers-percentage>] \n\t\t[--data-percentage=<data-percentage>]  \n\t\t--log=<True/False> \n\t\t--output-file=<output-filename>
"""
from docopt import docopt
from federated_learning.FederatedLearning import FederatedLearning
import logging
import random
import ast
arguments = docopt(__doc__)
import neptune

# print(arguments)

OUTPUT_PATH_PREFIX = "data_tmp/"
MNIST_PATH = "/home/ubuntu/data/MNIST/processed_manual"
EMNIST_PATH = "/home/ubuntu/data/EMNIST/processed"
FEMNIST_PATH="/home/ubuntu/leaf/data/femnist/data"
NUM_TRAIN_WORKERS = 30 # Total number of users which participate in training in each round
EPOCH_NUM = 5
ROUNDS = 1

fl = FederatedLearning(
        epochs_num = EPOCH_NUM, 
        output_prefix = OUTPUT_PATH_PREFIX + arguments['--output-file'],
        data_path = MNIST_PATH, 
        write_to_file = True if arguments['--log'] == "True" else False,
        log_level = logging.DEBUG)

if fl.write_to_file:
    neptune.init('ehsan/sandbox')
    neptune.create_experiment(name='evaluation_01')

fl.load_femnist_train(FEMNIST_PATH)
fl.load_femnist_test(FEMNIST_PATH)
logging.debug("Some sample data for user {}: {}".format(fl.workers_id[0], fl.train_data[fl.workers_id[0]]['y'][0:15]))
print(fl.workers_id[0])
print(len(fl.train_data[fl.workers_id[0]]['y']))

fl.create_server()

for round_no in range(0, ROUNDS):
    # [1, 3, 6, 9]
    workers_to_be_used_idx = random.sample(range(len(fl.workers_id)), NUM_TRAIN_WORKERS)
    # ['f_353', 'f_345']
    workers_to_be_used = [fl.workers_id[i] for i in workers_to_be_used_idx]
    fl.create_workers(workers_to_be_used)
    server_data_loader = fl.create_aggregated_data(workers_to_be_used)
    train_data_loader, test_data_loader = fl.create_datasets(workers_to_be_used)

    # fl.create_server_model()
    # fl.create_workers_model(workers_to_be_used)

    # for epoch_no in range(1, EPOCH_NUM + 1):
    #     fl.train_server(server_data_loader, round_no, epoch_no)
    #     # fl.train_workers(train_data_loader, workers_to_be_used, round_no, epoch_no)
    
    #     W = None
    #     if arguments['--avg']:
    #         W = [0.1] * 10
    #     # elif arguments['--opt']:
    #     #     W = fl.find_best_weights(epoch)
    #     else:
    #         logging.error("Not expected this mode!")
    #         exit(1)
    #     # Apply the server model to the test dataset
    #     fl.test(fl.server_model, test_data_loader, epoch_no, "trained")
    #     # fl.update_models(W, fl.server_model, fl.workers_model. workers_to_be_used)
    #     # fl.test(fl.server_model, test_data_loader, epoch_no, "averaged")




# if arguments['--attack'] == "1":
#     mal_users_list = ast.literal_eval(arguments['--workers-percentage'])
#     fl.attack_permute_labels_randomly(mal_users_list)
# elif arguments['--attack'] == "2":
#     mal_users_list = ast.literal_eval(arguments['--workers-percentage'])
#     data_percentage = int(arguments['--data-percentage'])
#     a = fl.attack_permute_labels_collaborative(mal_users_list, 40)
# else:
#     logging.info("** No attack is performed on the dataset.")

# fl.create_server_model()
# fl.create_workers_model()

# for epoch in range(1, EPOCH_NUM + 1):
#     fl.train_server(server_data_loader, epoch)
#     fl.train_workers(train_data_loader, epoch)
#     print()
#     W = None
#     if arguments['--avg']:
#         W = [0.1] * 10
#     elif arguments['--opt']:
#         W = fl.find_best_weights(epoch)
#     else:
#         logging.error("Not expected this mode!")
#         exit(1)
    
#     # base model is meant nothing in this scenario
#     fl.update_models(W, fl.server_model, fl.workers_model)

#     # Apply the server model to the test dataset
#     fl.test(fl.server_model, test_data_loader, epoch)


