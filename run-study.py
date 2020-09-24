"""
Usage: run-study.py \n\t\t(--avg | --opt) \n\t\t--attack=<attack-type> \n\t\t[--workers-percentage=<workers-percentage>] \n\t\t[--data-percentage=<data-percentage>]  \n\t\t--log=<True/False> \n\t\t--output-file=<output-filename>
"""
from docopt import docopt
from federated_learning.FederatedLearning import FederatedLearning
import logging
import random
import ast
import sys
arguments = docopt(__doc__)
import neptune
from multiprocessing import Process
from multiprocessing import Queue as mq
import queue
import time

# print(arguments)

OUTPUT_PATH_PREFIX = "data_tmp/"
MNIST_PATH = "/home/ubuntu/data/MNIST/processed_manual"
EMNIST_PATH = "/home/ubuntu/EMNIST/"
FEMNIST_PATH="/home/ubuntu/leaf/data/femnist/data"
FEMNIST_PATH_GOOGLE="/home/ubuntu/data/fed_google"
NUM_TRAIN_WORKERS = 30 # Total number of users which participate in training in each round
EPOCH_NUM = 5
ROUNDS = 20

models = dict()

fl = FederatedLearning(
        epochs_num = EPOCH_NUM, 
        output_prefix = OUTPUT_PATH_PREFIX + arguments['--output-file'],
        data_path = EMNIST_PATH, 
        write_to_file = True if arguments['--log'] == "true" or arguments['--log'] == "True" else False,
        log_level = logging.DEBUG)

if fl.write_to_file:
    neptune.init('ehsan/sandbox')
    neptune.create_experiment(name='evaluation_01')

train_data = fl.load_femnist_train_digits(FEMNIST_PATH_GOOGLE)
logging.info("Total workers size: {}".format(len(fl.workers_id)))
test_data = fl.load_femnist_test_digits(FEMNIST_PATH_GOOGLE)

random.seed("12345")
fl.create_server()

# # [1, 3, 6, 9]
workers_to_be_used_idx = random.sample(range(len(fl.workers_id)), NUM_TRAIN_WORKERS)
# # ['f_353', 'f_345']
workers_to_be_used = [fl.workers_id[i] for i in workers_to_be_used_idx]
fl.create_workers(workers_to_be_used)

logging.debug("First 10 labels of the first selected user: {}".format(train_data[workers_to_be_used[0]]['y'][0:10]))

# test_data_loaders = {}
models['server'] = fl.create_server_model()
for worker_id in workers_to_be_used:
    if worker_id not in models:
        models[worker_id] = fl.create_worker_model(worker_id)
    else:
        logging.debug("The model for worker {} exists".format(worker_id))

server_data_loader = fl.create_aggregated_data(workers_to_be_used, train_data)
train_data_loaders, test_data_loader = fl.create_datasets_mp(workers_to_be_used, train_data, test_data)

# if arguments['--attack'] == "1":
#     mal_users_list = ast.literal_eval(arguments['--workers-percentage'])
#     data_percentage = int(arguments['--data-percentage'])
#     fl.attack_permute_labels_randomly(mal_users_list, data_percentage)
# elif arguments['--attack'] == "2":
#     sys.exit(1)
#     # mal_users_list = ast.literal_eval(arguments['--workers-percentage'])
#     # data_percentage = int(arguments['--data-percentage'])
#     # a = fl.attack_permute_labels_collaborative(mal_users_list, 40)
# else:
#     logging.info("** No attack is performed on the dataset.")
#     sys.exit(1)

m_queue = mq()
for round_no in range(0, ROUNDS):
    processes = []
    # for epoch_no in range(1, EPOCH_NUM + 1):
    server_process = Process(target=fl.train_server_mp, args=(models['server'], server_data_loader, m_queue, round_no, EPOCH_NUM,))
    server_process.start()
    processes.append(server_process)
    for worker_id in workers_to_be_used:
        worker_p = Process(target=fl.train_worker_mp, args=(worker_id, models[worker_id], train_data_loaders[worker_id], m_queue, round_no, EPOCH_NUM,))
        worker_p.start()
        processes.append(worker_p)

    counter = (NUM_TRAIN_WORKERS + 1)
    models = dict()
    while counter > 0:
        try:
            (id, model) = m_queue.get()
            models[id] = model
            counter -= 1
            logging.info("Model id: {}, counter: {}".format(id, counter))
        except queue.Empty:
            logging.debug("Empty queue! Not should have been here!")

    for w_process in processes:
        w_process.join()
        w_process.terminate()
    

    
    W = None
    if arguments['--avg']:
        W = [0.1] * len(workers_to_be_used)
    elif arguments['--opt']:
        W = fl.find_best_weights(workers_to_be_used, models, round_no)
    else:
        logging.error("Not expected this mode!")
        sys.exit(1)
    # Apply the server model to the test dataset
    fl.update_models(0.7, W, fl.server_model, fl.workers_model, workers_to_be_used)
    fl.test(fl.server_model, test_data_loader, epoch_no, "server")
        # fl.test(fl.server_model, test_data_loader, epoch_no, "averaged")

    # # for worker_id in workers_to_be_used_:
    # #     fl.getback_model(fl.workers_model[worker_id])
    # #     fl.test_workers(fl.workers_model[worker_id], test_data_loaders[worker_id], epoch_no, "Test: " + worker_id)





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











######################### BACKUP CODES ############################


# fl.load_feminst_train_digits(FEMNIST_PATH_GOOGLE)
# print("Workers size: {}".format(len(fl.workers_id)))
# # for a, b in data.items():
# #     print("User: {}, data: {}".format(a, b))
# #     break

# fl.load_feminst_test_digits(FEMNIST_PATH_GOOGLE)
# # print("Workers size: {}".format(len/)
# # for a, b in data_test.items():
# #     print("User: {}, data: {}".format(a, b))
# #     break



# # fl.load_femnist_train(FEMNIST_PATH)
# # fl.load_femnist_test(FEMNIST_PATH)
# # logging.debug("Some sample data for user {}: {}".format(fl.workers_id[0], fl.train_data[fl.workers_id[0]]['y'][0:15]))


# random.seed("12345")
# fl.create_server()

# # # [1, 3, 6, 9]
# workers_to_be_used_idx = random.sample(range(len(fl.workers_id)), NUM_TRAIN_WORKERS)
# # # ['f_353', 'f_345']
# workers_to_be_used = [fl.workers_id[i] for i in workers_to_be_used_idx]
# fl.create_workers(workers_to_be_used)
# test_data_loaders = {}
# fl.create_server_model()
# fl.create_workers_model(workers_to_be_used)
# for worker_id in workers_to_be_used:
#     workers_to_be_used_ = []
#     workers_to_be_used_.append(worker_id)

#     # server_data_loader = fl.create_aggregated_data(workers_to_be_used)
#     train_data_loader, test_data_loader = fl.create_datasets(workers_to_be_used_)
#     test_data_loaders[worker_id] = test_data_loader


#     for round_no in range(0, ROUNDS):
        
#         for epoch_no in range(1, EPOCH_NUM + 1):
#             # fl.train_server(server_data_loader, round_no, epoch_no)
#             fl.train_workers(train_data_loader, workers_to_be_used_, round_no, epoch_no)
#             # W = None
#             # if arguments['--avg']:
#             #     W = [0.1] * 10
#             # # elif arguments['--opt']:
#             # #     W = fl.find_best_weights(epoch)
#             # else:
#             #     logging.error("Not expected this mode!")
#             #     exit(1)
#             # Apply the server model to the test dataset
#             # fl.update_models(W, fl.server_model, fl.workers_model, workers_to_be_used)
#             # fl.test(fl.server_model, test_data_loader, epoch_no, "test1")
#             # fl.test(fl.server_model, test_data_loader, epoch_no, "averaged")

# #         # print("Conv2 bias data: {}".format(fl.getback_model(fl.workers_model[worker_id]).conv2.bias.data))
#         for worker_id in workers_to_be_used_:
#             fl.getback_model(fl.workers_model[worker_id])
#             fl.test_workers(fl.workers_model[worker_id], test_data_loaders[worker_id], epoch_no, "Test: " + worker_id)

