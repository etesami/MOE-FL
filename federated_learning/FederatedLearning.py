import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
import cvxpy as cp
import tensorflow as tf
import numpy as np
import logging
import idx2numpy
from mnist import MNIST
import os
import random
import math
import json
from collections import defaultdict
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.FLNet import FLNet
import neptune

class Arguments():
    def __init__(self, epochs_num, use_cuda = False):
        self.batch_size = 12
        self.test_batch_size = 1000
        self.epochs = epochs_num
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 1
        self.save_model = False
        # self.workers_num = workers_num


class FederatedLearning():

    # Initializing variables
    def __init__(self, epochs_num = 5, output_prefix = None, data_path = None, use_cuda = False, write_to_file = False, log_level = logging.INFO):
        logging.basicConfig(format='%(asctime)s %(message)s', level=log_level)
        logging.info("Initializing Federated Learning class.")

        self.workers = dict()
        self.workers_id = []
        self.workers_model = dict()
        self.server = None
        self.server_model = None

        self.train_data = None
        self.test_data = None
        self.write_to_file = write_to_file
        self.hook = sy.TorchHook(torch)
        
        # use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.args = Arguments(epochs_num = epochs_num, use_cuda = use_cuda)
        
        torch.manual_seed(self.args.seed)

        if output_prefix is None or data_path is None:
            raise Exception("Sorry, you should specify the output/mnist path!")

        self.output_prefix = output_prefix
        self.data_path = data_path


    def create_workers(self, workers_id_list):
        logging.info("Randomly select {} workers".format(len(workers_id_list)))
        for worker_id in workers_id_list:
            if worker_id not in self.workers:
                logging.debug("Creating the worker: {}".format(worker_id))
                self.workers[worker_id] = sy.VirtualWorker(self.hook, id=worker_id)
            else:
                logging.debug("Worker {} exists. Skip creating this worker".format(worker_id))


    def create_server(self):
        logging.info("Creating the server")
        self.server = sy.VirtualWorker(self.hook, id="server")


    def load_mnist_data_training(self):
        file_path = "/train-images-idx3-ubyte"
        train_images = idx2numpy.convert_from_file(self.data_path + file_path)
        
        file_path = "/train-labels-idx1-ubyte"
        train_labels = idx2numpy.convert_from_file(self.data_path + file_path)
        
        self.train_images, self.train_labels = train_images.copy(), train_labels.copy()
        # indices = np.arange(self.train_images.shape[0])
        # np.random.shuffle(indices)
        # self.train_images = self.train_images[indices]
        # self.train_labels = self.train_labels[indices]


    def load_emnist_data_training(self):
        mndata = MNIST(self.data_path + '/digits')
        logging.info("Loading the EMNIST dataset")
        train_images, train_labels = mndata.load_training()
        self.train_images = np.asarray(train_images, dtype=np.uint8).reshape(-1, 28, 28)
        self.train_labels = np.asarray(train_labels)

        indices = np.arange(self.train_images.shape[0])
        np.random.shuffle(indices)
        self.train_images = self.train_images[indices]
        self.train_labels = self.train_labels[indices]


    def read_raw_data(self, data_path, get_workers = False):
        logging.debug("Reading raw data")
        workers_id = []
        groups = []
        data = defaultdict(lambda : None)

        files = os.listdir(data_path)
        files = [f for f in files if f.endswith('.json')]
        
        for f in files:
            file_path = os.path.join(data_path, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            data.update(cdata['user_data'])
            if get_workers:
                workers_id.extend(cdata['users'])

        if get_workers:
            workers_id = list(sorted(data.keys()))

        return workers_id, data


    def load_femnist_train(self, data_dir):
        logging.info("Loading train dataset")
        self.workers_id, self.train_data = self.read_raw_data(data_dir + "/train", get_workers = True)


    def load_femnist_test(self, data_dir):
        logging.info("Loading test dataset")
        _, self.test_data = self.read_raw_data(data_dir + "/test", get_workers = False)


    def count_digits(self):
        count = {}
        for i in range(0,10):
            count[i] = 0
        for d in self.train_labels:
            count[d] = count[d] + 1

        logging.info("Percentage of digits in whole training dataset: {}".format(
            [round(d*100.0/len(self.train_labels),2) for _, d in count.items()]))
        with open(self.output_prefix + "_digits", "w") as f:
            f.write(' '.join([str(round(d*100.0/len(self.train_labels),2)) for _, d in count.items()]))
            f.close()


    def create_datasets(self, selected_workers_id):
        logging.info("Creating federated dataset for selected workers")
        
        train_datasets = []
        # test_datasets = []
        test_data_images = torch.Tensor()
        test_data_labels = torch.Tensor()

        for worker_id in selected_workers_id:
            worker_record_num = len(self.train_data[worker_id]['y'])

            logging.debug("Worker {} has {} records".format(worker_id, worker_record_num))
            train_images = torch.Tensor(np.array(self.train_data[worker_id]['x'], dtype = np.single).reshape(-1, 1, 28, 28))
            train_labels = torch.Tensor(np.array(self.train_data[worker_id]['y'], dtype = np.single))
            print("Number of training data for user {} is {}".format(worker_id, len(train_labels)))

            # transform=transforms.Compose([transforms.ToTensor()])
            train_dataset = sy.BaseDataset(train_images, train_labels)\
                .send(self.workers[worker_id])
            train_datasets.append(train_dataset)

            print("Number of training data in the BaseDataset class is {}".format(len(train_dataset.targets)))

            test_images = torch.Tensor(np.array(self.test_data[worker_id]['x'], dtype = np.single).reshape(-1 , 1, 28, 28))
            test_labels = torch.Tensor(np.array(self.test_data[worker_id]['y'], dtype = np.single))
            print("Number of testing data for user {} is {}".format(worker_id, len(test_labels)))

            test_data_images = torch.cat((test_data_images, test_images))
            test_data_labels = torch.cat((test_data_labels, test_labels))

        train_dataset_loader = sy.FederatedDataLoader(
            sy.FederatedDataset(train_datasets), batch_size = self.args.batch_size, shuffle=False, drop_last = True, **self.kwargs)

        print("Length of Federated Dataset (Total number of records for all workers): {}".format(len(train_dataset_loader.federated_dataset)))
        
        test_dataset = sy.BaseDataset(test_data_images, test_data_labels)
        print("Length of the test dataset (Basedataset): {}".format(len(test_dataset)))

        test_dataset_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.args.test_batch_size, shuffle=True, drop_last = True, **self.kwargs)
        
        print("Length of the test data loader (datasets): {}".format(len(test_dataset_loader.dataset)))

        return train_dataset_loader, test_dataset_loader
        

    # Create aggregation in server from all users.
    def create_aggregated_data(self, workers_id_list):
        logging.info("Creating the aggregated data for the server from previously selected users")
        # Fraction of public data of each user, which be shared by the server
        aggregated_image = np.array([], dtype = np.single).reshape(-1, 28, 28)
        aggregated_label = np.array([], dtype = np.single)
        fraction = 0.2 
        total_samples_count = 0
        for worker_id in workers_id_list:
            worker_samples_count = len(self.train_data[worker_id]['y'])
            
            num_samples_for_server = math.floor(fraction * len(self.train_data[worker_id]['y']))
            logging.info("Sending {} from client {} with total {}".format(
                num_samples_for_server, worker_id, worker_samples_count
            ))
            total_samples_count = total_samples_count + num_samples_for_server
            indices = random.sample(range(worker_samples_count), num_samples_for_server)
            
            images = np.array([self.train_data[worker_id]['x'][i] for i in indices], dtype = np.single).reshape(-1, 28, 28)
            labels = np.array([self.train_data[worker_id]['y'][i] for i in indices], dtype = np.single)
            aggregated_image = np.concatenate((aggregated_image, images))
            aggregated_label = np.concatenate((aggregated_label, labels))

        logging.info("Selected {} samples in total for the server from all users.".format(total_samples_count))
        logging.debug("Aggregated train images shape: {}, dtype: {}".format(
            aggregated_image.shape, aggregated_image.dtype))
        logging.debug("Aggregated train images label: {}, dtype: {}".format(
            aggregated_label.shape, aggregated_label.dtype))

        aggregated_dataset = sy.BaseDataset(torch.Tensor(aggregated_image),\
            torch.Tensor(aggregated_label), \
            transform=transforms.Compose([transforms.ToTensor()]))
        
        aggregated_dataloader = sy.FederatedDataLoader(
            aggregated_dataset.federate([self.server]), batch_size = self.args.batch_size, shuffle = True, drop_last = True, **self.kwargs)

        return aggregated_dataloader


    def send_model(self, model, location, location_id):
        if isinstance(model, dict):
            for ww_id, ww in model.items():
                if ww.location is None:
                    model.send(location)
                elif ww.location.id != location_id:
                    model.move(location)
        elif model.location is None:
            model.send(location)
        elif model.location.id != location_id:
            model.move(location)


    def getback_model(self, model, selected_client_ids = None):
        if isinstance(model, dict):
            if selected_client_ids is not None:
                for ww_id in selected_client_ids:
                    if model[ww_id].location is not None:
                        model[ww_id].get()
            else:
                for ww_id, ww in model.items():
                    if ww.location is not None:
                        ww.get()
        elif model.location is not None:
            model.get()

    # '''
    # Attack 1
    # Permute all labels for given workers' id
    # workers_id_list: the list of workers' id (zero-based)
    # '''
    def attack_permute_labels_randomly(self, workers_percentage):
        logging.info("ATTACK 1: Permute labels of {} percentage of workers".format(workers_percentage))
        
        # Find workers which are counted as malicious users
        workers_id_list = None
        if 20 <= workers_percentage and workers_percentage < 40:
            workers_id_list = np.array([0, 1])
        elif 40 <= workers_percentage and workers_percentage < 50:
            workers_id_list = np.array([0, 1, 2, 3])
        elif 50 <= workers_percentage and workers_percentage < 60:
            workers_id_list = np.array([0, 1, 2, 3, 4])
        elif 60 <= workers_percentage and workers_percentage < 80:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5])
        elif 80 <= workers_percentage and workers_percentage < 100:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        elif workers_percentage == 100:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        logging.debug("Affected workers: {}".format(workers_id_list))  
        
        step = int(len(self.train_labels) / len(self.workers))
        for i in workers_id_list: 
            logging.debug("-- Permute all labels from index {} to {}".format(
                i * step, (i + 1) * step))
            self.train_labels[i * step:(i + 1) * step] = \
                        np.random.permutation(self.train_labels[i * step:(i + 1) * step])

    def attack_permute_labels_collaborative(self, workers_percentage, data_percentage):
        logging.info("ATTACK 2: Permute {} percentage of labels of the {} percentage of workers".format(data_percentage, workers_percentage))
        
        # Find workers which are counted as malicious users
        workers_id_list = None
        if 20 <= workers_percentage and workers_percentage < 40:
            workers_id_list = np.array([0, 1])
        elif 40 <= workers_percentage and workers_percentage < 50:
            workers_id_list = np.array([0, 1, 2, 3])
        elif 50 <= workers_percentage and workers_percentage < 60:
            workers_id_list = np.array([0, 1, 2, 3, 4])
        elif 60 <= workers_percentage and workers_percentage < 80:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5])
        elif 80 <= workers_percentage and workers_percentage < 100:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        elif workers_percentage == 100:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        logging.debug("Affected workers: {}".format(workers_id_list))        

        # Find labels which are going to be permuted base on the value of the percentage
        labels_to_be_changed = None
        if 20 <= data_percentage and data_percentage < 40:
            labels_to_be_changed = np.array([0, 1])
        elif 40 <= data_percentage and data_percentage < 60:
            labels_to_be_changed = np.array([0, 1, 2, 3])
        # I tried to add an option for 50% changes in data but it is not 
        # rasy with the current implementation.
        elif 60 <= data_percentage and data_percentage < 80:
            labels_to_be_changed = np.array([0, 1, 2, 3, 4, 5])
        elif 80 <= data_percentage and data_percentage < 100:
            labels_to_be_changed = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        elif data_percentage == 100:
            labels_to_be_changed = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        logging.debug("Affected labels: {}".format(labels_to_be_changed))

        # Find index of each number in the train_label and store them
        # into a dic named labels_indexes
        labels_indexes = {}
        for i in range(0, 10):
            labels_indexes[i] = np.array([], dtype = np.int64)
        
        # Initialization of indexes
        index = 0
        for n in self.train_labels:
            labels_indexes[n] = np.concatenate((labels_indexes[n], [index]))
            index = index + 1

        step = int(len(self.train_labels) / len(self.workers))
        # Start permutation
        for i in workers_id_list:
            for l in range(0, len(labels_to_be_changed), 2):
                
                # ex.
                # labels_to_be_changed = [0, 1]
                # labels_to_be_changed[l] = 0
                # labels_to_be_changed[l + 1] = 1 
                # labels_indexes[0] = list if indexes of 0
                logging.debug("-- Permute {} with {} from index {} to {}".format(
                    labels_to_be_changed[l], labels_to_be_changed[l+1],
                    i * step, (i + 1) * step
                    ))

                indexes_first_digit = np.where(
                    (i * step <= labels_indexes[labels_to_be_changed[l]]) &
                    (labels_indexes[labels_to_be_changed[l]] < (i + 1) * step)
                )[0]
                logging.debug("-- To be verified: Some indexes of {}: {}".format(
                    labels_to_be_changed[l], 
                    labels_indexes[labels_to_be_changed[l]][indexes_first_digit][0:10])
                )
                
                indexes_sec_digit = np.where(
                    (i * step <= labels_indexes[labels_to_be_changed[l + 1]]) &
                    (labels_indexes[labels_to_be_changed[l + 1]] < (i + 1) * step)
                )[0]
                logging.debug("-- To be verified: Some indexes of {}: {}".format(
                    labels_to_be_changed[l + 1], 
                    labels_indexes[labels_to_be_changed[l + 1]][indexes_sec_digit][0:10])
                )

                self.train_labels[labels_indexes[labels_to_be_changed[l]][indexes_first_digit]] = labels_to_be_changed[l + 1]
                self.train_labels[labels_indexes[labels_to_be_changed[l + 1]][indexes_sec_digit]]= labels_to_be_changed[l]


    def train_workers(self, federated_train_loader, workers_id_list, round_no, epoch_no):
        workers_opt = {}
        file = None
        # if self.write_to_file:
        #     file = open(self.output_prefix + "_train", "a")
        for ww_id in workers_id_list:
        # for ww_id, ww in self.workers.items():
            if self.workers_model[ww_id].location is None \
                    or self.workers_model[ww_id].location.id != ww_id:
                self.workers_model[ww_id].send(self.workers[ww_id])
            workers_opt[ww_id] = optim.SGD(params=self.workers_model[ww_id].parameters(), lr=self.args.lr)

        for batch_idx, (data, target) in enumerate(federated_train_loader):
            worker_id = data.location.id
            worker_model = self.workers_model[worker_id]
            worker_opt = workers_opt[worker_id]
            worker_model.train()
            data, target = data.to(self.device), target.to(self.device, dtype = torch.int64)
            worker_opt.zero_grad()
            output = worker_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            worker_opt.step()

            if batch_idx % self.args.log_interval == 0:
                loss = loss.get()
                if self.write_to_file:
                    neptune.log_metric(worker_id, loss)
                    # TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch_no, batch_idx, data.location.id, loss)
                    # file.write(TO_FILE)
                logging.info('Train Round: {}, Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    round_no, epoch_no, worker_id, batch_idx, 
                    batch_idx * self.args.batch_size, 
                    len(federated_train_loader) * self.args.batch_size,
                    100. * batch_idx / len(federated_train_loader), loss.item()))
        # Need to getback the self.workers_model
        # if self.write_to_file:
        #     file.close()
        print()


    def train_server(self, train_server_loader, round_no, epoch_no):
        file = None
        # if self.write_to_file:
        #     file = open(self.output_prefix + "_train_server", "a")

        self.send_model(self.server_model, self.server, "server")
        server_opt = optim.SGD(self.server_model.parameters(), lr=self.args.lr)
        for batch_idx, (data, target) in enumerate(train_server_loader):
            self.server_model.train()
            data, target = data.to(self.device), target.to(self.device, dtype = torch.int64)
            server_opt.zero_grad()
            output = self.server_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            server_opt.step()

            if batch_idx % self.args.log_interval == 0:
                loss = loss.get()
                if self.write_to_file:
                    neptune.log_metric('loss_server', loss)
                    # TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch_no, batch_idx, data.location.id, loss)
                    # file.write(TO_FILE)
                logging.info('Train Round: {}, Epoch: {} [server] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    round_no, epoch_no, batch_idx, 
                    batch_idx * self.args.batch_size, 
                    len(train_server_loader) * self.args.batch_size,
                    100. * batch_idx / len(train_server_loader), loss.item()))
        # if self.write_to_file:
        #     file.close()
        # Always need to get back the model
        self.getback_model(self.server_model)
        print()


    def test(self, model, test_loader, epoch, test_name):
        # if fl.write_to_file:
            # file = open(self.output_prefix + "_test", "a")
        model.eval()
        test_loss = 0
        correct = 0
        print(dir(test_loader))
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        if self.write_to_file:
            neptune.log_metric("server_test_loss_" + test_name, test_loss)
            neptune.log_metric("server_test_acc_" + test_name, 100. * correct / len(test_loader.dataset))
            # file = open(self.output_prefix + "_test", "a")
            # TO_FILE = '{} {} "{{/*0.80 Accuracy:}}\\n{}%" {}\n'.format(
            #     epoch, test_loss, 
            #     100. * correct / len(test_loader.dataset),
            #     100. * correct / len(test_loader.dataset))
            # file.write(TO_FILE)
            # file.close()
        logging.info('Test set [{}]: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_name, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    def find_best_weights(self, epoch):
        # reference_model = self.server_model
        # workers_model = self.workers_model
        file = open(self.output_prefix + "_weights", "a")
        self.getback_model(self.server_model)
        with torch.no_grad():
            reference_layers = [None] * 8
            reference_layers[0] = self.server_model.conv1.weight.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[1] = self.server_model.conv1.bias.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[2] = self.server_model.conv2.weight.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[3] = self.server_model.conv2.bias.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[4] = self.server_model.fc1.weight.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[5] = self.server_model.fc1.bias.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[6] = self.server_model.fc2.weight.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[7] = self.server_model.fc2.bias.data.numpy().copy().reshape(-1, 1).ravel()

            workers_params = {}
            """
            --> conv1.weight
            workers_params['worker0'][0] =
                convW0_11
                convW0_12
                convW0_21
                convW0_22

            --> conv1.bias
            workers_params['worker0'][1] =
                convW0_11
                convW0_12
                convW0_21
                convW0_22
            """
            self.getback_model(self.workers_model)
            for worker_id, worker_model in self.workers_model.items():
                workers_params[worker_id] = [[] for i in range(8)]
                workers_params[worker_id][0] = worker_model.conv1.weight.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][1] = worker_model.conv1.bias.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][2] = worker_model.conv2.weight.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][3] = worker_model.conv2.bias.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][4] = worker_model.fc1.weight.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][5] = worker_model.fc1.bias.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][6] = worker_model.fc2.weight.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][7] = worker_model.fc2.bias.data.numpy().copy().reshape(-1, 1)

            # logging.debug("workers_param shape: {}".format(len(workers_params)))
            # for key in workers_params:
            #     logging.debug("workers_param[{}]: {}".format(key, len(workers_params[key])))
            #     if key == "worker0":
            #         for i in range(0, len(workers_params[key])):
            #             logging.debug("workers_param[{}][{}]: {}".format(key, i, len(workers_params[key][i])))
            """
            --> conv1.weight
            workers_all_params[0] =
                [workers_param[worker0][0], workers_param[worker1][0], workers_param[worker2][0]]
            --> conv1.bias
            workers_all_params[1] =
                [workers_param[worker0][1], workers_param[worker1][1], workers_param[worker2][1]]
            """

            workers_all_params = []
            logging.info("Start the optimization....")
            workers_all_params.append(np.array([]).reshape(workers_params["worker0"][0].shape[0], 0))
            workers_all_params.append(np.array([]).reshape(workers_params["worker0"][1].shape[0], 0))
            workers_all_params.append(np.array([]).reshape(workers_params["worker0"][2].shape[0], 0))
            workers_all_params.append(np.array([]).reshape(workers_params["worker0"][3].shape[0], 0))
            workers_all_params.append(np.array([]).reshape(workers_params["worker0"][4].shape[0], 0))
            workers_all_params.append(np.array([]).reshape(workers_params["worker0"][5].shape[0], 0))
            workers_all_params.append(np.array([]).reshape(workers_params["worker0"][6].shape[0], 0))
            workers_all_params.append(np.array([]).reshape(workers_params["worker0"][7].shape[0], 0))

            for worker_id, worker_model in workers_params.items():
                workers_all_params[0] = np.concatenate((workers_all_params[0], workers_params[worker_id][0]), 1)
                workers_all_params[1] = np.concatenate((workers_all_params[1], workers_params[worker_id][1]), 1)
                workers_all_params[2] = np.concatenate((workers_all_params[2], workers_params[worker_id][2]), 1)
                workers_all_params[3] = np.concatenate((workers_all_params[3], workers_params[worker_id][3]), 1)
                workers_all_params[4] = np.concatenate((workers_all_params[4], workers_params[worker_id][4]), 1)
                workers_all_params[5] = np.concatenate((workers_all_params[5], workers_params[worker_id][5]), 1)
                workers_all_params[6] = np.concatenate((workers_all_params[6], workers_params[worker_id][6]), 1)
                workers_all_params[7] = np.concatenate((workers_all_params[7], workers_params[worker_id][7]), 1)

            # logging.debug("workers_all_param: {}".format(len(workers_all_params)))
            # for i in range(len(workers_all_params)):
            #     logging.debug("workers_all_params[{}]: {}".format(i, workers_all_params[i].shape))

            W = cp.Variable(len(self.workers_model))

            objective = cp.Minimize(cp.norm2(cp.matmul(workers_all_params[0], W) - reference_layers[0]) +
                                    cp.norm2(cp.matmul(workers_all_params[1], W) - reference_layers[1]) +
                                    cp.norm2(cp.matmul(workers_all_params[2], W) - reference_layers[2]) +
                                    cp.norm2(cp.matmul(workers_all_params[3], W) - reference_layers[3]) +
                                    cp.norm2(cp.matmul(workers_all_params[4], W) - reference_layers[4]) +
                                    cp.norm2(cp.matmul(workers_all_params[5], W) - reference_layers[5]) +
                                    cp.norm2(cp.matmul(workers_all_params[6], W) - reference_layers[6]) +
                                    cp.norm2(cp.matmul(workers_all_params[7], W) - reference_layers[7]))

            for i in range(len(workers_all_params)):
                logging.debug("Mean [{}]: {}".format(i, np.round(np.mean(workers_all_params[i],0) - np.mean(reference_layers[i],0),6)))
                logging.debug("")

            constraints = [0 <= W, W <= 1, sum(W) == 1]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.MOSEK)
            logging.info(W.value)
            logging.info("")
            TO_FILE = '{} {}\n'.format(epoch, np.array2string(W.value).replace('\n',''))
            file.write(TO_FILE)
            # file.close()
            return W.value


    def update_models(self, W, server_model, workers_model, workers_to_be_used):
        self.getback_model(workers_model, workers_to_be_used)
        self.getback_model(server_model)
        tmp_model = FLNet().to(self.device)

        with torch.no_grad():
            tmp_model.conv1.weight.data.fill_(0)
            tmp_model.conv1.bias.data.fill_(0)
            tmp_model.conv2.weight.data.fill_(0)
            tmp_model.conv2.bias.data.fill_(0)
            tmp_model.fc1.weight.data.fill_(0)
            tmp_model.fc1.bias.data.fill_(0)
            tmp_model.fc2.weight.data.fill_(0)
            tmp_model.fc2.bias.data.fill_(0)

            counter = 0
            # for worker_id, worker_model in workers_model.items():
            for worker_id in workers_to_be_used:
                worker_model = workers_model[worker_id]
                tmp_model.conv1.weight.data = (
                        tmp_model.conv1.weight.data + W[counter] * worker_model.conv1.weight.data)
                tmp_model.conv1.bias.data = (
                        tmp_model.conv1.bias.data + W[counter] * worker_model.conv1.bias.data)
                tmp_model.conv2.weight.data = (
                        tmp_model.conv2.weight.data + W[counter] * worker_model.conv2.weight.data)
                tmp_model.conv2.bias.data = (
                        tmp_model.conv2.bias.data + W[counter] * worker_model.conv2.bias.data)
                tmp_model.fc1.weight.data = (
                        tmp_model.fc1.weight.data + W[counter] * worker_model.fc1.weight.data)
                tmp_model.fc1.bias.data = (
                        tmp_model.fc1.bias.data + W[counter] * worker_model.fc1.bias.data)
                tmp_model.fc2.weight.data = (
                        tmp_model.fc2.weight.data + W[counter] * worker_model.fc2.weight.data)
                tmp_model.fc2.bias.data = (
                        tmp_model.fc2.bias.data + W[counter] * worker_model.fc2.bias.data)
                counter = counter + 1

            # base_model.conv1.weight.data = tmp_model.conv1.weight.data
            # base_model.conv1.bias.data = tmp_model.conv1.bias.data
            # base_model.conv2.weight.data = tmp_model.conv2.weight.data
            # base_model.conv2.bias.data = tmp_model.conv2.bias.data
            # base_model.fc1.weight.data = tmp_model.fc1.weight.data
            # base_model.fc1.bias.data = tmp_model.fc1.bias.data
            # base_model.fc2.weight.data = tmp_model.fc2.weight.data
            # base_model.fc2.bias.data = tmp_model.fc2.bias.data

            server_model.conv1.weight.data = tmp_model.conv1.weight.data
            server_model.conv1.bias.data = tmp_model.conv1.bias.data
            server_model.conv2.weight.data = tmp_model.conv2.weight.data
            server_model.conv2.bias.data = tmp_model.conv2.bias.data
            server_model.fc1.weight.data = tmp_model.fc1.weight.data
            server_model.fc1.bias.data = tmp_model.fc1.bias.data
            server_model.fc2.weight.data = tmp_model.fc2.weight.data
            server_model.fc2.bias.data = tmp_model.fc2.bias.data

            # for worker_id in workers_model.keys():
            for worker_id in workers_to_be_used:
                workers_model[worker_id].conv1.weight.data = tmp_model.conv1.weight.data
                workers_model[worker_id].conv1.bias.data = tmp_model.conv1.bias.data
                workers_model[worker_id].conv2.weight.data = tmp_model.conv2.weight.data
                workers_model[worker_id].conv2.bias.data = tmp_model.conv2.bias.data
                workers_model[worker_id].fc1.weight.data = tmp_model.fc1.weight.data
                workers_model[worker_id].fc1.bias.data = tmp_model.fc1.bias.data
                workers_model[worker_id].fc2.weight.data = tmp_model.fc2.weight.data
                workers_model[worker_id].fc2.bias.data = tmp_model.fc2.bias.data

    def create_server_model(self):
        logging.info("Creating a model for the server")
        self.server_model = FLNet().to(self.device)

    def create_workers_model(self, selected_workers_id):
        logging.info("Creating a model for workers")
        for worker_id in selected_workers_id:
            if worker_id not in self.workers_model:
                logging.debug("Creating a model for worker {}".format(worker_id))
                self.workers_model[worker_id] = FLNet().to(self.device)
            else:
                logging.debug("The model for worker {} exists".format(worker_id))
