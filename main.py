from federated_learning.FederatedLearning import FederatedLearning
import logging
import torch
from FLNet import FLNet

'''
01_avg: average, normal situation
02_opt: wieghted average, normal situation
03_att1_avg: average, attack 1
04_att1_opt: wieghted average, attack 1
'''

OUTPUT_PATH = "/home/savi/FederatedLearning/data_tmp/04_att1_opt"
MNIST_PATH = "/home/savi/data/MNIST"
WORKERS_NUM = 10
EPOCH_NUM = 5

fl = FederatedLearning(
        workers_num = WORKERS_NUM, 
        epochs_num = EPOCH_NUM, 
        output_prefix = OUTPUT_PATH,
        mnist_path = MNIST_PATH, 
        log_level = logging.INFO)

fl.create_workers()
fl.create_server()
fl.load_data()
train_data_loader, test_data_loader = fl.create_datasets()

server_data_loader = fl.create_aggregated_data()

fl.attack_premute_labels([5,6,7,8,9])

'''
Clients' models are being trained locally.
Server model is being updated only
'''
fl.create_server_model()
fl.create_workers_model()

for epoch in range(1, EPOCH_NUM + 1):
    fl.train_server(server_data_loader, epoch)
    fl.train_workers(train_data_loader, epoch)
    print()
    W = fl.find_best_weights(fl.server_model, fl.workers_model, epoch)
#     W = [0.1] * 10

    # base model is meant nothing in this scenario
    fl.update_models(W, fl.server_model, fl.workers_model)

    # Apply the server model to the test dataset
    fl.test(fl.server_model, test_data_loader, epoch)


