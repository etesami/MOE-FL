import torch
import h5py
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
# import cvxpy as cp
# import tensorflow as tf
import numpy as np
import logging
import idx2numpy
# from mnist import MNIST
import os
import random
import math
import json
from collections import defaultdict
# from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.FLNet import FLNet
import neptune
import time


test_batch_size = 20
hook = sy.TorchHook(torch)    
kwargs = {}
device = torch.device("cpu")
batch_size = 20
epoch_num = 5
lr = 0.01
momentum = 0.5
seed = 1
log_interval = 1
save_model = False
torch.manual_seed(seed)

def load_femnist_train_digits(data_dir):
    logging.debug("Reading femnist (google) raw data")
    file_name_train = "fed_emnist_digitsonly_train.h5"
    file_path = data_dir + "/" + file_name_train
    workers_id = []
    groups = []
    data = defaultdict(lambda : None)
    with h5py.File(file_path, "r") as h5_file:
        for gname, group in h5_file.items():
            # gname: example
            for dname, ds in group.items():
                # dname: f0000_14
                workers_id.append(dname)
                data_ = []
                for a, b in ds.items():
                    # a: label, b: list(int)
                    # a: pixels, b: list(list(int))
                    data_.append(b[()])
                data[dname] = {'x' : data_[1], 'y' : data_[0]}
    # self.workers_id, self.train_data = workers_id, data
    # self.workers_id = workers_id
    return workers_id, data

def load_femnist_test_digits(data_dir):
    logging.debug("Reading femnist (google) raw data")
    file_name_test = "fed_emnist_digitsonly_test.h5"
    file_path = data_dir + "/" + file_name_test
    groups = []
    data = defaultdict(lambda : None)
    with h5py.File(file_path, "r") as h5_file:
        for gname, group in h5_file.items():
            # gname: example
            for dname, ds in group.items():
                # dname: f0000_14
                data_ = []
                for a, b in ds.items():
                    # a: label, b: list(int)
                    # a: pixels, b: list(list(int))
                    data_.append(b[()])
                data[dname] = {'x' : data_[1], 'y' : data_[0]}
    return data

def create_aggregated_data(workers_id_list, train_data):
    print("Creating the aggregated data for the server from previously selected users")
    # Fraction of public data of each user, which be shared by the server
    aggregated_image = np.array([], dtype = np.single).reshape(-1, 28, 28)
    aggregated_label = np.array([], dtype = np.single)
    fraction = 0.2 
    total_samples_count = 0
    for worker_id in workers_id_list:
        worker_samples_count = len(train_data[worker_id]['y'])
        
        num_samples_for_server = math.floor(fraction * len(train_data[worker_id]['y']))
        print("Sending {} from client {} with total {}".format(
            num_samples_for_server, worker_id, worker_samples_count
        ))
        total_samples_count = total_samples_count + num_samples_for_server
        indices = random.sample(range(worker_samples_count), num_samples_for_server)
        
        images = np.array([train_data[worker_id]['x'][i] for i in indices], dtype = np.single).reshape(-1, 28, 28)
        labels = np.array([train_data[worker_id]['y'][i] for i in indices], dtype = np.single)
        aggregated_image = np.concatenate((aggregated_image, images))
        aggregated_label = np.concatenate((aggregated_label, labels))

    print("Selected {} samples in total for the server from all users.".format(total_samples_count))
    logging.debug("Aggregated train images shape: {}, dtype: {}".format(
        aggregated_image.shape, aggregated_image.dtype))
    logging.debug("Aggregated train images label: {}, dtype: {}".format(
        aggregated_label.shape, aggregated_label.dtype))

    aggregated_dataset = sy.BaseDataset(torch.Tensor(aggregated_image),\
        torch.Tensor(aggregated_label), \
        transform=transforms.Compose([transforms.ToTensor()]))
    
    return aggregated_dataset
    # aggregated_dataloader = sy.FederatedDataLoader(
    #     aggregated_dataset.federate([self.server]), batch_size = self.args.batch_size, shuffle = True, drop_last = True, **self.kwargs)
    # return aggregated_dataloader

def create_datasets_mp(selected_workers_id, train_data, test_data):
    
    print("Creating federated dataset for selected workers")
    
    train_data_loaders = dict()
    train_datasets = dict()
    test_data_images = torch.Tensor()
    test_data_labels = torch.Tensor()

    for worker_id in selected_workers_id:
        worker_record_num = len(train_data[worker_id]['y'])

        train_images = torch.Tensor(np.array(train_data[worker_id]['x'], dtype = np.single).reshape(-1, 1, 28, 28))
        train_labels = torch.Tensor(np.array(train_data[worker_id]['y'], dtype = np.single))

        # transform=transforms.Compose([transforms.ToTensor()])
        train_dataset = sy.BaseDataset(train_images, train_labels)
        train_datasets[worker_id] = train_dataset
        # .send(self.workers[worker_id])
        # train_data_loaders[worker_id] = sy.FederatedDataLoader(
        #     sy.FederatedDataset([train_dataset]), batch_size = self.args.batch_size, 
        #     shuffle=False, drop_last = False, **self.kwargs)

        logging.debug("Number of training data in the BaseDataset class is: {} and loader: {}".format(len(train_dataset.targets), len(train_datasets[worker_id])))

        test_images = torch.Tensor(np.array(test_data[worker_id]['x'], dtype = np.single).reshape(-1 , 1, 28, 28))
        test_labels = torch.Tensor(np.array(test_data[worker_id]['y'], dtype = np.single))
        logging.debug("Worker {} has {} training and {} test reocrds.".format(worker_id, len(train_labels), len(test_labels)))

        test_data_images = torch.cat((test_data_images, test_images))
        test_data_labels = torch.cat((test_data_labels, test_labels))
    
    test_dataset = sy.BaseDataset(test_data_images, test_data_labels)
    logging.debug("Lenght of targets for test: {}".format(len(test_data_labels)))
    logging.debug("Length of the test dataset (Basedataset): {}".format(len(test_dataset)))

    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, drop_last = False, **kwargs)
    
    logging.debug("Length of the test data loader (datasets): {}".format(len(test_dataset_loader.dataset)))

    return train_datasets, test_dataset_loader

def create_server():
    print("Creating the server")
    self.server = sy.VirtualWorker(self.hook, id="server")


def send_model(model, location, location_id):
    print("Model location: {}".format(model.location))
    print("Model location to be sent to: {}".format(location))
    if model.location is None:
        model.send(location)
    elif model.location.id != location_id:
        model.move(location)


def getback_model(model, selected_client_ids = None):
    if model.location is not None:
        model.get()

def train_worker_mp(worker_id, model_data, train_ds, m_queue, round_no, epoch_num):
    worker_model = FLNet().to(torch.device("cpu"))
    worker = sy.VirtualWorker(hook, id=worker_id)
    if model_data is not None:
        server_model = update_a_model(worker_model, model_data)
    print("start training worker...")
    send_model(worker_model, worker, worker_id)
    print("Sent model...")
    train_data_loader = sy.FederatedDataLoader(
                train_ds.federate([worker]), batch_size = batch_size, 
                shuffle=False, drop_last = False, **kwargs)
    worker_opt = optim.SGD(params=worker_model.parameters(), lr=lr)
    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(train_data_loader):
            worker_model.train()
            data, target = data.to(device), target.to(device, dtype = torch.int64)
            worker_opt.zero_grad()
            output = worker_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            worker_opt.step()

            if batch_idx % log_interval == 0:
                loss = loss.get()
                # if self.write_to_file:
                #     neptune.log_metric(worker_id, loss)
                    # TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch_no, batch_idx, data.location.id, loss)
                    # file.write(TO_FILE)
                print('[{}] Train Round: {}, Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    worker_id, round_no, epoch, worker_id, batch_idx, 
                    batch_idx * batch_size, 
                    len(train_data_loader) * batch_size,
                    100. * batch_idx / len(train_data_loader), loss.item()))
        
    getback_model(worker_model)
    m_queue.put((worker_id, compress_model_layers(worker_model)))
    # Need to getback the self.workers_model
    # if self.write_to_file:
    #     file.close()
    return 0

def train_server_mp(model_data, train_ds, m_queue, round_no, epoch_num):
    server_model = FLNet().to(torch.device("cpu"))
    server = sy.VirtualWorker(hook, id="server")
    if model_data is not None:
        server_model = update_a_model(server_model, model_data)
    print("start training server...")
    send_model(server_model, server, "server")
    print("Send model...")
    
    train_server_loader = sy.FederatedDataLoader(
        train_ds.federate([server]), batch_size = batch_size, shuffle = True, drop_last = True, **kwargs)

    server_opt = optim.SGD(server_model.parameters(), lr=lr)
    for epoch in range(epoch_num):
        # logging.debug("epoch {}".format(epoch))
        for batch_idx, (data, target) in enumerate(train_server_loader):
            # logging.debug("batch_idx {}".format(batch_idx))
            server_model.train()
            data, target = data.to(device), target.to(device, dtype = torch.int64)
            server_opt.zero_grad()
            output = server_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            server_opt.step()

            if batch_idx % log_interval == 0:
                loss = loss.get()
                # if self.write_to_file:
                #     neptune.log_metric('loss_server', loss)
                    # TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch_no, batch_idx, data.location.id, loss)
                    # file.write(TO_FILE)
                print('[{}] Train Round: {}, Epoch: {} [server] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    "server", round_no, epoch, batch_idx, 
                    batch_idx * batch_size, 
                    len(train_server_loader) * batch_size,
                    100. * batch_idx / len(train_server_loader), loss.item()))
    getback_model(server_model)
    print("Putting into the queue.")
    m_queue.put(("server", compress_model_layers(server_model)))
    print("Put into the queue!")
    return 0


def get_model_params_count(self, layer_no):
    tmp_model = FLNet().to(self.device)
    if layer_no == 0:
        return tmp_model.conv1.weight.data.numel()
    elif layer_no == 1:
        return tmp_model.conv1.bias.data.numel()
    elif layer_no == 2:
        return tmp_model.conv2.weight.data.numel()
    elif layer_no == 3:
        return tmp_model.conv2.bias.data.numel()
    elif layer_no == 4:
        return tmp_model.fc1.weight.data.numel()
    elif layer_no == 5:
        return tmp_model.fc1.bias.data.numel()
    elif layer_no == 6:
        return tmp_model.fc2.weight.data.numel()
    elif layer_no == 7:
        return tmp_model.fc2.bias.data.numel()
    else:
        raise Exception("Wrong layer number.")


def compress_model_layers(model):
    LAYERS = 8
    print("Prepare trained model to be returned and updated after all epochs")
    tmp_params = [[] for i in range(LAYERS)]
    tmp_params[0] = model.conv1.weight.data.numpy()
    tmp_params[1] = model.conv1.bias.data.numpy()
    tmp_params[2] = model.conv2.weight.data.numpy()
    tmp_params[3] = model.conv2.bias.data.numpy()
    tmp_params[4] = model.fc1.weight.data.numpy()
    tmp_params[5] = model.fc1.bias.data.numpy()
    tmp_params[6] = model.fc2.weight.data.numpy()
    tmp_params[7] = model.fc2.bias.data.numpy()
    for idx in range(len(tmp_params)):
        print("Compressing layer {} with {} items.".format(idx, tmp_params[idx].shape))
    return tmp_params


def test(model_data, test_loader, test_name):
    model = FLNet().to(torch.device("cpu"))
    server = sy.VirtualWorker(hook, id="server")
    model = update_a_model(model, model_data)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # if self.write_to_file:
    #     neptune.log_metric("test_loss_" + test_name, test_loss)
    #     neptune.log_metric("test_acc_" + test_name, 100. * correct / len(test_loader.dataset))
        # file = open(self.output_prefix + "_test", "a")
        # TO_FILE = '{} {} "{{/*0.80 Accuracy:}}\\n{}%" {}\n'.format(
        #     epoch, test_loss, 
        #     100. * correct / len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset))
        # file.write(TO_FILE)
        # file.close()
    print('Test set [{}]: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def update_a_model(model, model_data):
    model.conv1.weight.data = torch.Tensor(model_data[0])
    model.conv1.bias.data = torch.Tensor(model_data[1])
    model.conv2.weight.data = torch.Tensor(model_data[2])
    model.conv2.bias.data = torch.Tensor(model_data[3])
    model.fc1.weight.data = torch.Tensor(model_data[4])
    model.fc1.bias.data = torch.Tensor(model_data[5])
    model.fc2.weight.data = torch.Tensor(model_data[6])
    model.fc2.bias.data = torch.Tensor(model_data[7])
    return model


def create_server_model(self):
    print("Creating a model for the server")
    # models["server"] = FLNet().to(self.device)
    return FLNet().to(self.device)

def create_worker_model(self, worker_id):
    print("Creating a model for worker {}".format(worker_id))
    return FLNet().to(self.device)
