import torch
import h5py
import logging
import os
import math
import json
import syft as sy
import cvxpy as cp
import numpy as np
from random import sample
from collections import defaultdict
from torch.nn import functional as F
from torch import optim, float32, int64, tensor
from torchvision import datasets, transforms
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.FLNet import FLNet
import neptune

class FederatedLearning():

    # Initializing variables
    def __init__(
        self, batch_size, test_batch_size, lr, wieght_decay, momentum, neptune_enable, 
        log_enable, log_interval, output_dir, random_seed):
        
        logging.info("Initializing Federated Learning class...")

        self.workers = dict()
        self.workers_model = dict()
        self.server = None
        self.server_model = None
        self.train_data = None
        self.test_data = None
        # Example MNIST: (train and test look similar)
        #   train images: self.train_data['x']
        #   train labels: self.train_data['y']
        # Example EMNIST and FEMNIST: (train and test look similar)
        #   train images: self.train_data['f000_1']['x']
        #   train labels: self.train_data['f000_1']['y']
        
        self.hook = sy.TorchHook(torch)
        use_cuda = False
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.device = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(random_seed)
        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.momentum = momentum
        self.seed = random_seed
        self.log_interval = log_interval
        self.log_enable = log_enable
        self.neptune_enable = neptune_enable
        self.weight_decay = wieght_decay
        self.log_file_path = output_dir


    def create_workers(self, workers_id_list):
        logging.info("Creating workers...")
        for worker_id in workers_id_list:
            if worker_id not in self.workers:
                logging.debug("Creating the worker: {}".format(worker_id))
                self.workers[worker_id] = sy.VirtualWorker(self.hook, id=worker_id)
            else:
                logging.debug("Worker {} exists. Skip creating this worker".format(worker_id))


    def create_server(self):
        logging.info("Creating the server...")
        self.server = sy.VirtualWorker(self.hook, id="server")


    def create_server_model(self):
        logging.info("Creating a model for the server...")
        self.server_model = FLNet().to(self.device)


    def create_workers_model(self, selected_workers_id):
        logging.info("Creating a model for {} worker(s)...".format(len(selected_workers_id)))
        for worker_id in selected_workers_id:
            if worker_id not in self.workers_model:
                logging.debug("Creating a model for worker {}".format(worker_id))
                self.workers_model[worker_id] = FLNet().to(self.device)
            else:
                logging.debug("The model for worker {} exists".format(worker_id))

    
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


    # def read_raw_data(self, data_path, get_workers = False):
    #     logging.debug("Reading raw data")
    #     workers_id = []
    #     groups = []
    #     data = defaultdict(lambda : None)

    #     files = os.listdir(data_path)
    #     files = [f for f in files if f.endswith('.json')]
        
    #     for f in files:
    #         file_path = os.path.join(data_path, f)
    #         with open(file_path, 'r') as inf:
    #             cdata = json.load(inf)
    #         data.update(cdata['user_data'])
    #         if get_workers:
    #             workers_id.extend(cdata['users'])

    #     if get_workers:
    #         workers_id = list(sorted(data.keys()))

    #     return workers_id, data


    # def load_femnist_train(self, data_dir):
    #     logging.info("Loading train dataset")
    #     self.workers_id, self.train_data = self.read_raw_data(data_dir + "/train", get_workers = True)


    # def load_femnist_test(self, data_dir):
    #     logging.info("Loading test dataset")
    #     _, self.test_data = self.read_raw_data(data_dir + "/test", get_workers = False)
    

    def load_federated_femnist_train_datasets(self, raw_data, workers_idx):
        """ 

        Args:
            raw_data (dict of str): dict contains train and test data categorized based on user id
                # raw_data['f0_12345']['x'], raw_data['f0_12345']['y'] 
        Returns:
        """    
        logging.info("Creating federated dataset for {} workers...".format(len(workers_idx)))
        train_datasets = []
        test_data_images = tensor([], dtype=float32)
        test_data_labels = tensor([], dtype=int64)

        for worker_id in workers_idx:

            worker_record_num = len(raw_data[worker_id]['y'])
            train_images = tensor(raw_data[worker_id]['x'], dtype=float32).reshape(-1, 1, 28, 28)
            train_labels = tensor(raw_data[worker_id]['y'], dtype=int64)

            # transform=transforms.Compose([transforms.ToTensor()])
            train_dataset = sy.BaseDataset(train_images, train_labels)\
                .send(self.workers[worker_id])
            train_datasets.append(train_dataset)

            test_images = tensor(raw_data[worker_id]['x'], dtype=float32).reshape(-1 , 1, 28, 28)
            test_labels = tensor(raw_data[worker_id]['y'], dtype=int64)

            test_data_images = torch.cat((test_data_images, test_images))
            test_data_labels = torch.cat((test_data_labels, test_labels))

        train_dataloader = sy.FederatedDataLoader(
            sy.FederatedDataset(train_datasets), batch_size = self.batch_size, shuffle=False, drop_last = False, **self.kwargs)

        logging.info("Length of Federated Dataset (Total number of records for all workers): {}".format(
            len(train_dataloader.federated_dataset)))
        
        test_dataset = sy.BaseDataset(test_data_images, test_data_labels)
        logging.info("Lenght of targets for test dataset: {}".format(len(test_data_labels)))
        logging.debug("Length of the test dataset (Basedataset): {}".format(len(test_dataset)))

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.test_batch_size, shuffle=True, drop_last = False, **self.kwargs)
        
        logging.debug("Length of the test data loader (datasets): {}".format(len(test_dataloader.dataset)))

        return train_dataloader, test_dataloader


    def create_server_femnist_dataset(self, raw_data, workers_idx, percentage):
        """ 
        Args:
            raw_data (dict): 
            workers_idx (list(int)): 
            percentage (float): Out of 100, amount of public data of each user
        Returns:
        """  
        logging.info("Creating the server data from {} selected users...".format(len(workers_idx)))
        # Fraction of public data of each user, which be shared by the server
        server_images = np.array([], dtype = np.float32).reshape(-1, 28, 28)
        server_labels = np.array([], dtype = np.int64).reshape(0, 1)
        for worker_id in workers_idx:
            worker_samples_num = len(raw_data[worker_id]['y'])
            num_samples_for_server = math.floor((percentage / 100.0) * worker_samples_num)
            logging.debug("Sending {} samples from worker {} with total {}".format(
                num_samples_for_server, worker_id, worker_samples_num))
            indices = sample(range(worker_samples_num), num_samples_for_server)
            
            images = np.array([raw_data[worker_id]['x'][i] for i in indices], dtype = np.float32).reshape(-1, 28, 28)
            labels = np.array([raw_data[worker_id]['y'][i] for i in indices], dtype = np.int64)
            server_images = np.concatenate((server_images, images))
            server_labels = np.concatenate((server_labels, labels))

        logging.info("Selected {} samples in total for the server from {} users.".format(server_images.shape, len(workers_idx)))
        return server_images, server_labels


    def create_federated_server_leaf_dataloader(self, train_images, train_labels, batch_size, shuffle):
        logging.info("Creating federated server dataloader...")
        aggregated_dataset = sy.BaseDataset(
            tensor(train_images.reshape(-1, 1, 28, 28), dtype=float32),
            tensor(train_labels.reshape(-1), dtype=int64))
        
        return sy.FederatedDataLoader(
            aggregated_dataset.federate([self.server]), 
            batch_size = batch_size, shuffle = shuffle, drop_last = False, **self.kwargs)


    # def create_datasets(self, selected_workers_id):
    #     logging.info("Creating federated dataset for selected {} workers...".format(len(selected_workers_id)))
        
    #     train_datasets = []
    #     test_data_images = torch.Tensor()
    #     test_data_labels = torch.Tensor()

    #     for worker_id in selected_workers_id:
    #         worker_record_num = len(self.train_data[worker_id]['y'])

    #         logging.debug("Worker {} has {} records".format(worker_id, worker_record_num))
    #         train_images = torch.Tensor(np.array(self.train_data[worker_id]['x'], dtype = np.single).reshape(-1, 1, 28, 28))
    #         train_labels = torch.Tensor(np.array(self.train_data[worker_id]['y'], dtype = np.single))
    #         logging.debug("Number of training data for user {} is {}".format(worker_id, len(train_labels)))

    #         # transform=transforms.Compose([transforms.ToTensor()])
    #         train_dataset = sy.BaseDataset(train_images, train_labels)\
    #             .send(self.workers[worker_id])
    #         train_datasets.append(train_dataset)

    #         logging.debug("Number of training data in the BaseDataset class is {}".format(len(train_dataset.targets)))

    #         test_images = torch.Tensor(np.array(self.test_data[worker_id]['x'], dtype = np.single).reshape(-1 , 1, 28, 28))
    #         test_labels = torch.Tensor(np.array(self.test_data[worker_id]['y'], dtype = np.single))
    #         logging.debug("Number of testing data for user {} is {}".format(worker_id, len(test_labels)))

    #         test_data_images = torch.cat((test_data_images, test_images))
    #         test_data_labels = torch.cat((test_data_labels, test_labels))

    #     train_dataset_loader = sy.FederatedDataLoader(
    #         sy.FederatedDataset(train_datasets), batch_size = self.batch_size, shuffle=False, drop_last = True, **self.kwargs)

    #     logging.info("Length of Federated Dataset (Total number of records for all workers): {}".format(len(train_dataset_loader.federated_dataset)))
        
    #     test_dataset = sy.BaseDataset(test_data_images, test_data_labels)
    #     logging.info("Lenght of targets for test: {}".format(len(test_data_labels)))
    #     logging.debug("Length of the test dataset (Basedataset): {}".format(len(test_dataset)))

    #     test_dataset_loader = torch.utils.data.DataLoader(
    #         test_dataset, batch_size=self.test_batch_size, shuffle=True, drop_last = False, **self.kwargs)
        
    #     logging.debug("Length of the test data loader (datasets): {}".format(len(test_dataset_loader.dataset)))

    #     return train_dataset_loader, test_dataset_loader
        

    # # Create aggregation in server from all users.
    # def create_mnist_server_data(self, workers_id_list):
    #     logging.info("Creating the aggregated data for the server from {} selected users...".format(len(workers_id_list)))
    #     # Fraction of public data of each user, which be shared by the server
    #     aggregated_image = np.array([], dtype = np.single).reshape(-1, 28, 28)
    #     aggregated_label = np.array([], dtype = np.single)
    #     fraction = 0.2 
    #     total_samples_count = 0
    #     for worker_id in workers_id_list:
    #         worker_samples_count = len(self.train_data[worker_id]['y'])
            
    #         num_samples_for_server = math.floor(fraction * len(self.train_data[worker_id]['y']))
    #         logging.debug("Sending {} from client {} with total {}".format(
    #             num_samples_for_server, worker_id, worker_samples_count
    #         ))
    #         total_samples_count = total_samples_count + num_samples_for_server
    #         indices = sample(range(worker_samples_count), num_samples_for_server)
            
    #         images = np.array([self.train_data[worker_id]['x'][i] for i in indices], dtype = np.single).reshape(-1, 28, 28)
    #         labels = np.array([self.train_data[worker_id]['y'][i] for i in indices], dtype = np.single)
    #         aggregated_image = np.concatenate((aggregated_image, images))
    #         aggregated_label = np.concatenate((aggregated_label, labels))

    #     logging.info("Selected {} samples in total for the server from all users.".format(total_samples_count))
    #     logging.debug("Aggregated train images shape: {}, dtype: {}".format(
    #         aggregated_image.shape, aggregated_image.dtype))
    #     logging.debug("Aggregated train images label: {}, dtype: {}".format(
    #         aggregated_label.shape, aggregated_label.dtype))

    #     aggregated_dataset = sy.BaseDataset(torch.Tensor(aggregated_image),\
    #         torch.Tensor(aggregated_label), \
    #         transform=transforms.Compose([transforms.ToTensor()]))
        
    #     aggregated_dataloader = sy.FederatedDataLoader(
    #         aggregated_dataset.federate([self.server]), batch_size = self.batch_size, shuffle = True, drop_last = True, **self.kwargs)

    #     return aggregated_dataloader


    # # Create aggregation in server from all users.
    # def create_aggregated_data(self, workers_id_list):
    #     logging.info("Creating the aggregated data for the server from {} selected users...".format(len(workers_id_list)))
    #     # Fraction of public data of each user, which be shared by the server
    #     aggregated_image = np.array([], dtype = np.single).reshape(-1, 28, 28)
    #     aggregated_label = np.array([], dtype = np.single)
    #     fraction = 0.2 
    #     total_samples_count = 0
    #     for worker_id in workers_id_list:
    #         worker_samples_count = len(self.train_data[worker_id]['y'])
            
    #         num_samples_for_server = math.floor(fraction * len(self.train_data[worker_id]['y']))
    #         logging.debug("Sending {} from client {} with total {}".format(
    #             num_samples_for_server, worker_id, worker_samples_count
    #         ))
    #         total_samples_count = total_samples_count + num_samples_for_server
    #         indices = sample(range(worker_samples_count), num_samples_for_server)
            
    #         images = np.array([self.train_data[worker_id]['x'][i] for i in indices], dtype = np.single).reshape(-1, 28, 28)
    #         labels = np.array([self.train_data[worker_id]['y'][i] for i in indices], dtype = np.single)
    #         aggregated_image = np.concatenate((aggregated_image, images))
    #         aggregated_label = np.concatenate((aggregated_label, labels))

    #     logging.info("Selected {} samples in total for the server from all users.".format(total_samples_count))
    #     logging.debug("Aggregated train images shape: {}, dtype: {}".format(
    #         aggregated_image.shape, aggregated_image.dtype))
    #     logging.debug("Aggregated train images label: {}, dtype: {}".format(
    #         aggregated_label.shape, aggregated_label.dtype))

    #     aggregated_dataset = sy.BaseDataset(torch.Tensor(aggregated_image),\
    #         torch.Tensor(aggregated_label), \
    #         transform=transforms.Compose([transforms.ToTensor()]))
        
    #     aggregated_dataloader = sy.FederatedDataLoader(
    #         aggregated_dataset.federate([self.server]), batch_size = self.batch_size, shuffle = True, drop_last = True, **self.kwargs)

    #     return aggregated_dataloader


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

    def get_labels_from_data_percentage(self, data_percentage):
        # Find labels which are going to be permuted base on the value of the percentage
        labels_to_be_changed = None
        if 20 <= data_percentage and data_percentage < 40:
            labels_to_be_changed = np.array(sample(range(10), 2))
        elif 40 <= data_percentage and data_percentage < 60:
            labels_to_be_changed = np.array(sample(range(10), 4))
        elif 60 <= data_percentage and data_percentage < 80:
            labels_to_be_changed = np.array(sample(range(10), 6))
        elif 80 <= data_percentage and data_percentage < 100:
            labels_to_be_changed = np.array(sample(range(10), 8))
        elif data_percentage == 100:
            labels_to_be_changed = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        return labels_to_be_changed


    # # '''
    # # Attack convert to black
    # # '''
    # def attack_robust_aggregation(self, selected_workers_id):
    #     logging.info("ATTACK 1: {} of workers are affected".format(len(selected_workers_id)))
    #     logging.debug("Some Affected workers: {}...".format(selected_workers_id[0:5]))  

    #     for worker_id in selected_workers_id:
        
    #         for pixel_id in len(self.train_data[worker_id]['x']):
    #             self.train_data[worker_id]['x'][pixel_id] = 
    #             self.train_data[worker_id]['y'][indexes_first_digit] = labels_to_be_changed[l + 1]
    #             self.train_data[worker_id]['y'][indexes_sec_digit] = labels_to_be_changed[l]
            
    #         logging.debug("Worker Labels (After): {}\n".format(self.train_data[worker_id]['y'][0:10]))
    #     return selected_workers_id
        

    # def get_subset_of_workers(self, workers_percentage):
    #     NUM_MAL_WORKERS = round(workers_percentage * len(self.workers_id) / 100.0)
    #     selected_workers_idx = sample(range(len(self.workers_id)), NUM_MAL_WORKERS)
    #     selected_workers_id = [self.workers_id[i] for i in selected_workers_idx]
    #     logging.debug("Total selected workers: {}".format(len(selected_workers_id)))
    #     return selected_workers_id
    
    # '''
    # Attack 1
    # Permute all labels for given workers' id
    # workers_id_list: the list of workers' id (zero-based)
    # '''
    def attack_permute_labels_randomly(self, selected_workers_id, data_percentage):
        logging.info("ATTACK 1: Permute labels of {} workers".format(len(selected_workers_id)))
        logging.debug("Some Affected workers: {}...".format(selected_workers_id[0:5]))  

        for worker_id in selected_workers_id:
        
            # Find labels which are going to be permuted base on the value of the percentage
            labels_to_be_changed = self.get_labels_from_data_percentage(data_percentage)
            logging.debug("Affected labels: {}".format(labels_to_be_changed))

            # Find index of each number in the train_label and store them
            # into a dic named labels_indexes
            labels_indexes = {}
            for i in range(0, 10):
                labels_indexes[i] = np.array([], dtype = np.int64)
        
            # Initialization of indexes
            index = 0
            logging.debug("Worker Labels (Before): {}".format(self.train_data[worker_id]['y'][0:10]))
            for n in self.train_data[worker_id]['y']:
                labels_indexes[n] = np.concatenate((labels_indexes[n], [index]))
                index = index + 1
            
            for l in range(0, len(labels_to_be_changed), 2):
                
                # ex.
                # labels_to_be_changed = [0, 1]
                # labels_to_be_changed[l] = 0
                # labels_to_be_changed[l + 1] = 1 
                # labels_indexes[0] = list if indexes of 0
                logging.debug("-- Permute {} with {} from worker {}".format(
                    labels_to_be_changed[l], labels_to_be_changed[l+1],worker_id))

                indexes_first_digit = labels_indexes[labels_to_be_changed[l]]
                logging.debug("-- Some indexes of {}: {}".format(
                    labels_to_be_changed[l], 
                    labels_indexes[labels_to_be_changed[l]][0:10])
                )
                
                indexes_sec_digit = labels_indexes[labels_to_be_changed[l + 1]]
                logging.debug("-- Some indexes of {}: {}".format(
                    labels_to_be_changed[l + 1], 
                    labels_indexes[labels_to_be_changed[l + 1]][0:10])
                )

                self.train_data[worker_id]['y'][indexes_first_digit] = labels_to_be_changed[l + 1]
                self.train_data[worker_id]['y'][indexes_sec_digit] = labels_to_be_changed[l]
            
            logging.debug("Worker Labels (After): {}\n".format(self.train_data[worker_id]['y'][0:10]))
            

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
                        batch_idx * self.batch_size, 
                        len(server_dataloader) * self.batch_size,
                        100. * batch_idx / len(server_dataloader), loss.item()))
        # if self.log_enable:
        #     file.close()
        # Always need to get back the model
        self.getback_model(self.server_model)
        print()
    
    def train_workers(self, federated_train_loader, workers_id_list, round_no, epochs_num):
        workers_opt = {}
        for epoch_no in range(epochs_num):
            for ww_id in workers_id_list:
                if self.workers_model[ww_id].location is None \
                        or self.workers_model[ww_id].location.id != ww_id:
                    self.workers_model[ww_id].send(self.workers[ww_id])
                workers_opt[ww_id] = optim.SGD(
                    params=self.workers_model[ww_id].parameters(), lr=self.lr, weight_decay=self.weight_decay)

            for batch_idx, (data, target) in enumerate(federated_train_loader):
                worker_id = data.location.id
                worker_model = self.workers_model[worker_id]
                worker_opt = workers_opt[worker_id]
                worker_model.train()
                data, target = data.to(self.device), target.to(self.device)
                worker_opt.zero_grad()
                output = worker_model(data)
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
                        batch_idx * self.batch_size, 
                        len(federated_train_loader) * self.batch_size,
                        100. * batch_idx / len(federated_train_loader), loss.item()))
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


    def test(self, model, test_loader, round_no):
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
            neptune.log_metric("test_loss", test_loss)
            neptune.log_metric("test_acc", test_acc)
        if self.log_enable:
            file = open(self.log_file_path + "server_test", "a")
            TO_FILE = '{} {} "{{/*Accuracy:}}\\n{}%" {}\n'.format(
                round_no, test_loss, 
                test_acc,
                test_acc)
            file.write(TO_FILE)
            file.close()
        logging.info('Test Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_acc))
        return test_acc


    # def test_workers(self, model, test_loader, epoch, test_name):
    #     self.getback_model(model)
    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for batch_idx, (data, target) in enumerate(test_loader):
    #             # print("Batch {}".format(batch_idx))
    #             # print("Shape of Test data {}".format(data.shape))
    #             # print("Shape of Target data {}".format(target.shape))
    #             data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
    #             output = model(data)
    #             # print("Test Output: {} and the target is {}".format(output, target))
    #             # print("Output type: {}".format(type(output.data)))
    #             # print("Target type: {}".format(target.type()))
    #             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    #             pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
    #             # print("--> Pred: {}, Target: {}".format(pred, target))
    #             correct += pred.eq(target.view_as(pred)).sum().item()

    #     test_loss /= len(test_loader.dataset)
    #     if self.neptune_enable:
    #         neptune.log_metric("test_loss_" + test_name, test_loss)
    #         neptune.log_metric("test_acc_" + test_name, 100. * correct / len(test_loader.dataset))
    #         # file = open(self.output_prefix + "_test", "a")
    #         # TO_FILE = '{} {} "{{/*0.80 Accuracy:}}\\n{}%" {}\n'.format(
    #         #     epoch, test_loss, 
    #         #     100. * correct / len(test_loader.dataset),
    #         #     100. * correct / len(test_loader.dataset))
    #         # file.write(TO_FILE)
    #         # file.close()
    #     logging.info('Test set [{}]: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_name, test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))


    def find_best_weights(self, trained_model, workers_to_be_used):
        # reference_model = self.server_model
        # workers_model = self.workers_model
        # self.getback_model(self.server_model)
        with torch.no_grad():
            reference_layers = [None] * 8
            for layer_id, param_tensor in enumerate(trained_model.state_dict()):
                reference_layers[layer_id] = trained_model.state_dict()[param_tensor].data.numpy().copy().reshape(-1, 1).ravel()

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
            for worker_id in workers_to_be_used:
            # for worker_id, worker_model in self.workers_model.items():
                worker_model = self.workers_model[worker_id]
                self.getback_model(worker_model)
                workers_params[worker_id] = [[] for i in range(8)]

                for layer_id, param_tensor in enumerate(worker_model.state_dict()):
                    workers_params[worker_id][layer_id] = worker_model.state_dict()[param_tensor].data.numpy().copy().reshape(-1, 1)

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
            for ii in range(7):
                workers_all_params.append(np.array([]).reshape(workers_params[workers_to_be_used[0]][ii].shape[0], 0))

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

            objective = cp.Minimize(
                (1.0 / (len(self.workers_model) * len(self.workers_model))) *
                   (cp.norm2(cp.matmul(workers_all_params[0], W) - reference_layers[0]) +
                    cp.norm2(cp.matmul(workers_all_params[1], W) - reference_layers[1]) +
                    cp.norm2(cp.matmul(workers_all_params[2], W) - reference_layers[2]) +
                    cp.norm2(cp.matmul(workers_all_params[3], W) - reference_layers[3]) +
                    cp.norm2(cp.matmul(workers_all_params[4], W) - reference_layers[4]) +
                    cp.norm2(cp.matmul(workers_all_params[5], W) - reference_layers[5]) +
                    cp.norm2(cp.matmul(workers_all_params[6], W) - reference_layers[6]) +
                    cp.norm2(cp.matmul(workers_all_params[7], W) - reference_layers[7])
                    )
                )

            for i in range(len(workers_all_params)):
                logging.debug("Mean [{}]: {}".format(i, np.round(np.mean(workers_all_params[i],0) - np.mean(reference_layers[i],0),6)))
                logging.debug("")

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


    def find_best_weights_from_trusted_idx(self, workers_idx, trusted_idx):
        """

        Args:
            workers_idx (list[str])
            trusted_idx (list[str])
        """
        self.getback_model(self.workers_model, workers_idx)
        avg_model = self.get_average_model(trusted_idx)
        with torch.no_grad():
            reference_layers = [None] * 8
            for layer_id, param in enumerate(avg_model.parameters()):
                reference_layers[layer_id] = param.data.numpy().reshape(-1, 1).ravel()

            workers_params = {}
            # """
            # --> conv1.weight
            # workers_params['worker0'][0] =
            #     convW0_11
            #     convW0_12
            #     convW0_21
            #     convW0_22

            # --> conv1.bias
            # workers_params['worker0'][1] =
            #     convW0_11
            #     convW0_12
            #     convW0_21
            #     convW0_22
            # """
            workers_to_be_used = list(set(workers_idx) - set(trusted_idx))
            for worker_id in workers_to_be_used:
                worker_model = self.workers_model[worker_id]
                self.getback_model(worker_model)
                workers_params[worker_id] = [[] for i in range(8)]

                for layer_id, param in enumerate(worker_model.parameters()):
                    workers_params[worker_id][layer_id] = param.data.numpy().reshape(-1, 1)

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
            for ii in range(8):
                workers_all_params.append(np.array([]).reshape(workers_params[workers_to_be_used[0]][ii].shape[0], 0))
                logging.debug("all_params: {}".format(workers_all_params[ii].shape))

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

            W = cp.Variable(len(workers_to_be_used))

            objective = cp.Minimize(
                    cp.norm2(cp.matmul(workers_all_params[0], W) - reference_layers[0]) +
                    cp.norm2(cp.matmul(workers_all_params[1], W) - reference_layers[1]) +
                    cp.norm2(cp.matmul(workers_all_params[2], W) - reference_layers[2]) +
                    cp.norm2(cp.matmul(workers_all_params[3], W) - reference_layers[3]) +
                    cp.norm2(cp.matmul(workers_all_params[4], W) - reference_layers[4]) +
                    cp.norm2(cp.matmul(workers_all_params[5], W) - reference_layers[5]) +
                    cp.norm2(cp.matmul(workers_all_params[6], W) - reference_layers[6]) +
                    cp.norm2(cp.matmul(workers_all_params[7], W) - reference_layers[7])
                )

            for i in range(len(workers_all_params)):
                logging.debug("Mean [{}]: {}".format(i, np.round(np.mean(workers_all_params[i],0) - np.mean(reference_layers[i],0),6)))
                logging.debug("")

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
        

    def wieghted_avg_model(self, W, workers_idx):
        self.getback_model(self.workers_model, workers_idx)
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

            for counter, worker_id in enumerate(workers_idx):
                worker_model = self.workers_model[worker_id]
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
    
    
    def update_models(self, round_no, W, workers_idx, trusted_idx):
        self.getback_model(self.workers_model, workers_idx)
        self.getback_model(self.server_model)
        with torch.no_grad():
            tmp_model = self.wieghted_avg_model(W, list(set(workers_idx) - set(trusted_idx)))
            avg_model = self.get_average_model(trusted_idx)


            # if server_update:
            #     alpha = 0 if round_no == 0 else alpha
            self.server_model.conv1.weight.data = (tmp_model.conv1.weight.data + avg_model.conv1.weight.data) / 2.0
            self.server_model.conv1.bias.data = (tmp_model.conv1.bias.data + avg_model.conv1.bias.data) / 2.0
            self.server_model.conv2.weight.data = (tmp_model.conv2.weight.data + avg_model.conv2.weight.data) / 2.0
            self.server_model.conv2.bias.data = (tmp_model.conv2.bias.data + avg_model.conv2.bias.data) / 2.0
            self.server_model.fc1.weight.data = (tmp_model.fc1.weight.data + avg_model.fc1.weight.data) / 2.0
            self.server_model.fc1.bias.data = (tmp_model.fc1.bias.data + avg_model.fc1.bias.data) / 2.0
            self.server_model.fc2.weight.data = (tmp_model.fc2.weight.data + avg_model.fc2.weight.data) / 2.0
            self.server_model.fc2.bias.data = (tmp_model.fc2.bias.data + avg_model.fc2.bias.data) / 2.0

            # if workers_update:
            for worker_id in workers_idx:
                self.workers_model[worker_id].conv1.weight.set_(self.server_model.conv1.weight.data)
                self.workers_model[worker_id].conv1.bias.set_(self.server_model.conv1.bias.data)
                self.workers_model[worker_id].conv2.weight.set_(self.server_model.conv2.weight.data)
                self.workers_model[worker_id].conv2.bias.set_(self.server_model.conv2.bias.data)
                self.workers_model[worker_id].fc1.weight.set_(self.server_model.fc1.weight.data)
                self.workers_model[worker_id].fc1.bias.set_(self.server_model.fc1.bias.data)
                self.workers_model[worker_id].fc2.weight.set_(self.server_model.fc2.weight.data)
                self.workers_model[worker_id].fc2.bias.set_(self.server_model.fc2.bias.data)
    
    
