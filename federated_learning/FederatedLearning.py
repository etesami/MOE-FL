import torch
import h5py
import logging
import os
import math
import json
import neptune
import syft as sy
import cvxpy as cp
import numpy as np
from copy import deepcopy
from random import sample
from collections import defaultdict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim, float32, int64, tensor, cat
from torchvision import datasets, transforms
from sklearn.preprocessing import MinMaxScaler
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.helper import utils
from federated_learning.FLNet import FLNet
# from federated_learning.FLNetComplex import FLNetComplex as FLNet

class FederatedLearning():

    # Initializing variables
    # batch_size, test_batch_size, lr, wieght_decay, momentum, 
    def __init__(
        self, neptune_enable, 
        log_enable, log_interval, output_dir, random_seed):
        
        logging.info("Initializing Federated Learning class...")

        self.hook = sy.TorchHook(torch)
        use_cuda = False
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(random_seed)

        self.workers = dict()
        # self.workers_model = dict()
        # self.server = None
        # self.server_model = None
        # self.batch_size = batch_size
        # self.test_batch_size = test_batch_size
        # self.lr = lr
        # self.momentum = momentum
        # self.log_interval = log_interval
        # self.weight_decay = wieght_decay
        self.seed = random_seed
        self.log_enable = log_enable
        self.neptune_enable = neptune_enable
        self.log_file_path = output_dir


    def create_workers(self, workers_id_list):
        logging.info("Creating workers...")
        for worker_id in workers_id_list:
            if worker_id not in self.workers:
                logging.debug("Creating the worker: {}".format(worker_id))
                self.workers[worker_id] = sy.VirtualWorker(self.hook, id=worker_id)
            else:
                logging.debug("Worker {} exists. Skip creating this worker".format(worker_id))


    # def create_server(self):
    #     logging.info("Creating the server...")
    #     self.server = sy.VirtualWorker(self.hook, id="server")


    def create_server_model(self):
        logging.info("Creating a model for the server...")
        self.server_model = FLNet().to(self.device)


    def create_model(self):
        logging.info("Creating a model...")
        return FLNet().to(self.device)


    def create_workers_model(self, selected_workers_id):
        logging.info("Creating a model for {} worker(s)...".format(len(selected_workers_id)))
        for worker_id in selected_workers_id:
            if worker_id not in self.workers_model:
                logging.debug("Creating a (copy) model of server for worker {}".format(worker_id))
                self.workers_model[worker_id] = deepcopy(self.server_model)
            else:
                logging.debug("The model for worker {} exists".format(worker_id))

    ############################ MNIST RELATED FUNCS ###############################
    
    def create_federated_mnist(self, dataset, destination_idx, batch_size, shuffle):
        """ 

        Args:
            dataset (FLCustomDataset): Dataset to be federated
            destination_idx (list[str]): Path to the config file
        Returns:
            Obj: Corresponding python object
        """    
        workers = []
        if "server" in destination_idx:
            workers.append(self.server)
        else:
            for worker_id, worker in self.workers.items():
                worker_id in destination_idx and workers.append(worker)

        fed_dataloader = sy.FederatedDataLoader(
            dataset.federate(workers),
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True)

        return fed_dataloader

    def create_mnist_fed_datasets(self, raw_dataset):
        """
        raw_datasets (dict)
        ex.
            data: raw_datasets['worker_1']['x']
            label: raw_datasets['worker_1']['y']
        """
        fed_datasets = dict()

        for ww_id, ww_data in raw_dataset.items():
            images = tensor(ww_data['x'], dtype=float32)
            labels = tensor(ww_data['y'].ravel(), dtype=int64)
            dataset = sy.BaseDataset(
                    images,
                    labels,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((ww_data['x'].mean(),), (ww_data['x'].std(),))])
                ).federate([self.workers[ww_id]])
            fed_datasets[ww_id] = dataset

        return fed_datasets

    ############################ FEMNIST RELATED FUNCS ###############################

    def create_femnist_dataset(self, raw_data, workers_idx, shuffle=True, drop_last=True):
        """ 

        Args:
            raw_data (dict of str): dict contains processed train and test data categorized based on user id
                # raw_data['f0_12345']['x'], raw_data['f0_12345']['y'] 
        Returns:
            Dataloader for the server
        """    
        logging.info("Creating 1 test dataset from {} workers".format(len(workers_idx)))
        # raw_data = utils.extract_data(raw_data, workers_idx)
        server_images = np.array([], dtype = np.float32).reshape(-1, 28, 28)
        server_labels = np.array([], dtype = np.int64)

        for worker_id in workers_idx:
            images = np.array(raw_data[worker_id]['x'], dtype = np.float32).reshape(-1, 28, 28)
            labels = np.array(raw_data[worker_id]['x'], dtype = np.int64).ravel()
            server_images = np.concatenate((server_images, images))
            server_labels = np.concatenate((server_labels, labels))

        test_dataset = FLCustomDataset(
            server_images,
            server_labels,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((server_images.mean(),), (server_images.std(),))])
        )

        return test_dataset


    def create_femnist_fed_dataset(self, raw_data, workers_idx, percentage):
        """ 
        Assume this only used for preparing aggregated dataset for the server
        Args:
            raw_data (dict): 
            workers_idx (list(int)): 
            percentage (float): Out of 100, amount of public data of each user
        Returns:
        """  
        logging.info("Creating the dataset from {}% of {} selected users' data...".format(
            percentage, len(workers_idx)))
        # Fraction of public data of each user, which be shared by the server
        server_images = tensor([], dtype=float32).view(-1, 28, 28)
        server_labels = tensor([], dtype=int64)
        # server_images = np.array([], dtype = np.float32).reshape(-1, 28, 28)
        # server_labels = np.array([], dtype = np.int64)
        for worker_id in workers_idx:
            worker_samples_num = len(raw_data[worker_id]['y'])
            num_samples_for_server = math.floor((percentage / 100.0) * worker_samples_num)
            logging.debug("Sending {} samples from worker {} with total {}".format(
                num_samples_for_server, worker_id, worker_samples_num))
            indices = sample(range(worker_samples_num), num_samples_for_server)
            images = tensor(
                [raw_data[worker_id]['x'][i] for i in indices], dtype = float32).view(-1, 28, 28)
            labels = tensor([raw_data[worker_id]['y'][i] for i in indices], dtype = int64).view(-1)
            server_images = cat((server_images, images))
            server_labels = cat((server_labels, labels))

        logging.info("Selected {} samples in total for the server from {} users.".format(server_images.shape, len(workers_idx)))

        return sy.BaseDataset(
            server_images,
            server_labels,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((server_images.mean().item(),), (server_images.std().item(),))])
        ).federate([self.server])


    def create_femnist_fed_datasets(self, raw_dataset, workers_idx):
        fed_datasets = dict()

        for worker_id in workers_idx:
            images = tensor(raw_dataset[worker_id]['x'], dtype=float32)
            labels = tensor(raw_dataset[worker_id]['y'].ravel(), dtype=int64)
            dataset = sy.BaseDataset(
                    images,
                    labels,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((raw_dataset[worker_id]['x'].mean(),), (raw_dataset[worker_id]['x'].std(),))])
                ).federate([self.workers[worker_id]])
            fed_datasets[worker_id] = dataset

        return fed_datasets


    def create_femnist_datasets(self, raw_dataset, workers_idx):
        datasets = dict()

        for worker_id in workers_idx:
            images = tensor(raw_dataset[worker_id]['x'], dtype=float32)
            labels = tensor(raw_dataset[worker_id]['y'].ravel(), dtype=int64)
            dataset = sy.BaseDataset(
                    images,
                    labels,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((raw_dataset[worker_id]['x'].mean(),), (raw_dataset[worker_id]['x'].std(),))])
                )
            datasets[worker_id] = dataset

        return datasets


    ############################ GENERAL FUNC ################################

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

    
    def train_server(self, server_dataloader, round_no, epochs_num):
        self.send_model(self.server_model, self.server, "server")
        server_opt = optim.SGD(self.server_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch_no in range(epochs_num):
            for batch_idx, (data, target) in enumerate(server_dataloader):
                self.server_model.train()
                data, target = data.to(self.device), target.to(self.device)
                server_opt.zero_grad()
                output = self.server_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                server_opt.step()

                if batch_idx % self.log_interval == 0:
                    loss = loss.get()
                    if self.neptune_enable:
                        neptune.log_metric('train_w0_loss', loss)
                    if self.log_enable:
                        file = open(self.log_file_path + "server_train", "a")
                        TO_FILE = '{} {} {} [server] {}\n'.format(round_no, epoch_no, batch_idx, loss)
                        file.write(TO_FILE)
                        file.close()
                    logging.info('Train Round: {}, Epoch: {} [server] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        round_no, epoch_no, batch_idx, 
                        batch_idx * server_dataloader.batch_size, 
                        len(server_dataloader) * server_dataloader.batch_size,
                        100. * batch_idx / len(server_dataloader), loss.item()))
        # Always need to get back the model
        # self.getback_model(self.server_model)
        print()

    
    def train_workers(self, federated_train_loader, workers_model, round_no, epochs_num):
        workers_opt = {}
        for ww_id, ww_model in workers_model.items():
            if ww_model.location is None \
                    or ww_model.location.id != ww_id:
                ww_model.send(self.workers[ww_id])
            workers_opt[ww_id] = optim.SGD(
                params=ww_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        for epoch_no in range(epochs_num):
            for ww_id, fed_dataloader in federated_train_loader.items():
                if ww_id in workers_model.keys():
                    for batch_idx, (data, target) in enumerate(fed_dataloader):
                        worker_id = data.location.id
                        worker_opt = workers_opt[worker_id]
                        workers_model[worker_id].train()
                        data, target = data.to(self.device), target.to(self.device)
                        worker_opt.zero_grad()
                        output = workers_model[worker_id](data)
                        loss = F.nll_loss(output, target)
                        loss.backward()
                        worker_opt.step()

                        if batch_idx % self.log_interval == 0:
                            loss = loss.get()
                            if self.neptune_enable:
                                neptune.log_metric("train_loss_" + str(worker_id), loss)
                            if self.log_enable:
                                file = open(self.log_file_path + str(worker_id) + "_train", "a")
                                TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch_no, batch_idx, worker_id, loss)
                                file.write(TO_FILE)
                                file.close()
                            logging.info('Train Round: {}, Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                round_no, epoch_no, worker_id, batch_idx, 
                                batch_idx * fed_dataloader.batch_size, 
                                len(fed_dataloader) * fed_dataloader.batch_size,
                                100. * batch_idx / len(fed_dataloader), loss.item()))
        print()


    def save_workers_model(self, workers_idx, round_no):
        self.getback_model(self.workers_model, workers_idx)
        logging.info("Saving models {}".format(workers_idx))
        for worker_id, worker_model in self.workers_model.items():
            if worker_id in workers_idx:
                self.save_model(worker_model, "R{}_{}".format(round_no, worker_id))
        

    def save_model(self, model, name):
        parent_dir = "{}{}".format(self.log_file_path, "models")
        if not os.path.isdir(parent_dir):
            logging.debug("Create a directory for model(s).")
            os.mkdir(parent_dir)
        full_path = "{}/{}".format(parent_dir, name)
        logging.debug("Saving the model into " + full_path)
        torch.save(model, full_path)


    def test(self, model, test_loader, worker_id, round_no):
        self.getback_model(model)
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)

        if self.neptune_enable:
            neptune.log_metric("test_loss_" + str(worker_id), test_loss)
            neptune.log_metric("test_acc_"  + str(worker_id), test_acc)
        if self.log_enable:
            file = open(self.log_file_path + str(worker_id) + "_test", "a")
            TO_FILE = '{} {} "{{/*Accuracy:}}\\n{}%" {}\n'.format(
                round_no, test_loss, test_acc, test_acc)
            file.write(TO_FILE)
            file.close()
        
        logging.info('Test Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_acc))
        return test_acc


    def wieghted_avg_model(self, W, workers_model):
        self.getback_model(workers_model)
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

            for counter, (ww_id, worker_model) in enumerate(workers_model.items()):
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

        return tmp_model


    def update_models(self, workers_idx, weighted_avg_model):
        self.getback_model(weighted_avg_model)
        with torch.no_grad():
            for worker_id in workers_idx:
                model = None
                if worker_id == "server":
                    self.getback_model(self.server_model)
                    self.server_model.conv1.weight.set_(weighted_avg_model.conv1.weight.data)
                    self.server_model.conv1.bias.set_(weighted_avg_model.conv1.bias.data)
                    self.server_model.conv2.weight.set_(weighted_avg_model.conv2.weight.data)
                    self.server_model.conv2.bias.set_(weighted_avg_model.conv2.bias.data)
                    self.server_model.fc1.weight.set_(weighted_avg_model.fc1.weight.data)
                    self.server_model.fc1.bias.set_(weighted_avg_model.fc1.bias.data)
                    self.server_model.fc2.weight.set_(weighted_avg_model.fc2.weight.data)
                    self.server_model.fc2.bias.set_(weighted_avg_model.fc2.bias.data)
                else:
                    self.getback_model(self.workers_model[worker_id])
                    self.workers_model[worker_id].conv1.weight.set_(weighted_avg_model.conv1.weight.data)
                    self.workers_model[worker_id].conv1.bias.set_(weighted_avg_model.conv1.bias.data)
                    self.workers_model[worker_id].conv2.weight.set_(weighted_avg_model.conv2.weight.data)
                    self.workers_model[worker_id].conv2.bias.set_(weighted_avg_model.conv2.bias.data)
                    self.workers_model[worker_id].fc1.weight.set_(weighted_avg_model.fc1.weight.data)
                    self.workers_model[worker_id].fc1.bias.set_(weighted_avg_model.fc1.bias.data)
                    self.workers_model[worker_id].fc2.weight.set_(weighted_avg_model.fc2.weight.data)
                    self.workers_model[worker_id].fc2.bias.set_(weighted_avg_model.fc2.bias.data)


    def normalize_weights(self, list_of_ids, **kwargs):
        self.getback_model(self.workers_model, list_of_ids)
        w0_model = None
        for model_id in kwargs:
            if model_id == "w0_model":
                w0_model = kwargs[model_id]

        workers_params = {}
        for worker_id in list_of_ids:
            worker_model = self.workers_model[worker_id]
            self.getback_model(worker_model)

            workers_params[worker_id] = [[] for i in range(8)]
            for layer_id, param in enumerate(worker_model.parameters()):
                workers_params[worker_id][layer_id] = param.data.numpy().reshape(-1, 1)

        if w0_model is not None:
            workers_params['w0_model'] = [[] for i in range(8)]
            for layer_id, param in enumerate(w0_model.parameters()):
                workers_params['w0_model'][layer_id] = param.data.numpy().reshape(-1, 1)

        workers_all_params = []
        for ii in range(8):
            workers_all_params.append(np.array([]).reshape(workers_params[list_of_ids[0]][ii].shape[0], 0))
            logging.debug("all_dparams: {}".format(workers_all_params[ii].shape))

        for worker_id, worker_model in workers_params.items():
            workers_all_params[0] = np.concatenate((workers_all_params[0], workers_params[worker_id][0]), 1)
            workers_all_params[1] = np.concatenate((workers_all_params[1], workers_params[worker_id][1]), 1)
            workers_all_params[2] = np.concatenate((workers_all_params[2], workers_params[worker_id][2]), 1)
            workers_all_params[3] = np.concatenate((workers_all_params[3], workers_params[worker_id][3]), 1)
            workers_all_params[4] = np.concatenate((workers_all_params[4], workers_params[worker_id][4]), 1)
            workers_all_params[5] = np.concatenate((workers_all_params[5], workers_params[worker_id][5]), 1)
            workers_all_params[6] = np.concatenate((workers_all_params[6], workers_params[worker_id][6]), 1)
            workers_all_params[7] = np.concatenate((workers_all_params[7], workers_params[worker_id][7]), 1)

        normalized_workers_all_params = []
        for ii in range(len(workers_all_params)):
            norm = MinMaxScaler().fit(workers_all_params[ii])
            normalized_workers_all_params.append(norm.transform(workers_all_params[ii]))

        return normalized_workers_all_params


    def find_best_weights(self, referenced_model, workers_to_be_used):

        # last column of normalized_weights is corresponding to the w0_model:
        normalized_weights = self.normalize_weights(workers_to_be_used, w0_model=referenced_model)

        reference_layer = []
        workers_all_params = []
        for ii in range(len(normalized_weights)):
            reference_layer.append(normalized_weights[ii][:,-1].reshape(-1, 1))
            workers_all_params.append(normalized_weights[ii][:,:normalized_weights[ii].shape[1] - 1])

        reference_layers = []
        for ii in range(len(reference_layer)):
            tmp = np.array([]).reshape(reference_layer[ii].shape[0], 0)
            for jj in range(len(workers_to_be_used)):
                tmp = np.concatenate((tmp, reference_layer[ii]), axis=1)
            reference_layers.append(tmp)

        W = cp.Variable(len(workers_to_be_used))
        objective = cp.Minimize(
                cp.matmul(cp.norm2(workers_all_params[0] - reference_layers[0], axis=0), W) +
                cp.matmul(cp.norm2(workers_all_params[1] - reference_layers[1], axis=0), W) +
                cp.matmul(cp.norm2(workers_all_params[2] - reference_layers[2], axis=0), W) +
                cp.matmul(cp.norm2(workers_all_params[3] - reference_layers[3], axis=0), W) +
                cp.matmul(cp.norm2(workers_all_params[4] - reference_layers[4], axis=0), W) +
                cp.matmul(cp.norm2(workers_all_params[5] - reference_layers[5], axis=0), W) +
                cp.matmul(cp.norm2(workers_all_params[6] - reference_layers[6], axis=0), W) +
                cp.matmul(cp.norm2(workers_all_params[7] - reference_layers[7], axis=0), W)
            )

        # for i in range(len(workers_all_params)):
        #     logging.debug("Mean [{}]: {}".format(i, np.round(np.mean(workers_all_params[i],0) - np.mean(reference_layers[i],0),6)))
        #     logging.debug("")

        constraints = [0 <= W, W <= 1, sum(W) == 1]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK)
        logging.info(W.value)
        logging.info("")
        if self.log_enable:
            file = open(self.log_file_path + "opt_weights", "a")
            TO_FILE = '{}\n'.format(np.array2string(W.value).replace('\n',''))
            file.write(TO_FILE)
            file.close()
        return W.value


########################################################################
##################### Trusted Users ####################################

    def get_average_param(self, all_params, indexes):
        """ Find the average of all parameters of all layers of given workers
        Args:
        Returns:
            
        """    
        logging.info("Finding an average model for {} workers.".format(len(indexes)))
        avg_param = []
        for ii in range(len(all_params)):
            params = np.transpose(np.array([all_params[ii][:,i] for i in indexes]))
            logging.debug("params[{}] shape: {}".format(ii, params.shape))
            avg_param.append(params.mean(axis=1))
            
        return avg_param


    def get_index_number(self, workers_idx, selected_idx):
        positions = []
        for ii, jj in enumerate(workers_idx):
            if jj in selected_idx:
                positions.append(ii)
        return positions


    def find_best_weights_from_trusted_idx_normalized_last_layer(self, workers_idx, trusted_idx):
        """
        Args:
            workers_idx (list[str])
            trusted_idx (list[str])
        """
        trusted_workers_position = self.get_index_number(workers_idx, trusted_idx)
        all_params = self.normalize_weights(workers_idx)
        """
        len(all_params) = 8
        all_params[0].shape = (num of elements in layer 0 of cnn, num of users)
        all_params[1].shape = (num of elements in layer 1 of cnn, num of users)
        """
        avg_param = self.get_average_param(all_params, trusted_workers_position)
        
        # Not trusted users (i.e. Normal users + attackers)
        workers_to_be_used = list(set(workers_idx) - set(trusted_idx))
        workers_to_be_used_position = self.get_index_number(workers_idx, workers_to_be_used)

        workers_all_params = []
        for ii in range(len(all_params)):
            workers_all_params.append(np.transpose(np.array([all_params[ii][:,i] for i in workers_to_be_used_position])))

        W = cp.Variable(len(workers_to_be_used))

        objective = cp.Minimize(cp.norm2(cp.matmul(workers_all_params[7], W) - avg_param[7]))

        constraints = [0 <= W, W <= 1, sum(W) == 1]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK)
        logging.info(W.value)
        logging.info("")
        if self.log_enable:
            file = open(self.log_file_path + "opt_weights", "a")
            TO_FILE = '{}\n'.format(np.array2string(W.value).replace('\n',''))
            file.write(TO_FILE)
            file.close()
        return W.value
        

    def get_average_model(self, workers_idx):
        """ Find the average of all parameters of all layers of given workers. The average for each 
            layer is stored as an element of a list.

        Args:
            workers_idx (list[str]): List of workers ids
        Returns:
            
        """    
        logging.info("Finding an average model for {} workers.".format(len(workers_idx)))
        tmp_model = FLNet().to(self.device)            
        self.getback_model(self.workers_model, workers_idx)

        with torch.no_grad():
            for id_, ww_id in enumerate(workers_idx):
                worker_model = self.workers_model[ww_id]
                if id_ == 0:
                    tmp_model.conv1.weight.set_(worker_model.conv1.weight.data)
                    tmp_model.conv1.bias.set_(worker_model.conv1.bias.data)
                    tmp_model.conv2.weight.set_(worker_model.conv2.weight.data)
                    tmp_model.conv2.bias.set_(worker_model.conv2.bias.data)
                    tmp_model.fc1.weight.set_(worker_model.fc1.weight.data)
                    tmp_model.fc1.bias.set_(worker_model.fc1.bias.data)
                    tmp_model.fc2.weight.set_(worker_model.fc2.weight.data)
                    tmp_model.fc2.bias.set_(worker_model.fc2.bias.data)
                else:
                    tmp_model.conv1.weight.set_(
                        tmp_model.conv1.weight.data + worker_model.conv1.weight.data)
                    tmp_model.conv1.bias.set_(
                        tmp_model.conv1.bias.data + worker_model.conv1.bias.data)
                    tmp_model.conv2.weight.set_(
                        tmp_model.conv2.weight.data + worker_model.conv2.weight.data)
                    tmp_model.conv2.bias.set_(
                        tmp_model.conv2.bias.data + worker_model.conv2.bias.data)
                    tmp_model.fc1.weight.set_(
                        tmp_model.fc1.weight.data + worker_model.fc1.weight.data)
                    tmp_model.fc1.bias.set_(
                        tmp_model.fc1.bias.data + worker_model.fc1.bias.data)
                    tmp_model.fc2.weight.set_(
                        tmp_model.fc2.weight.data + worker_model.fc2.weight.data)
                    tmp_model.fc2.bias.set_(
                        tmp_model.fc2.bias.data + worker_model.fc2.bias.data)
                
        for param in tmp_model.parameters():
            param.data = param.data / len(workers_idx)

        return tmp_model
        

    