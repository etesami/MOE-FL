import os
import yaml
import idx2numpy
import numpy as np
import logging
from os import mkdir
from time import strftime
from math import floor
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import tensor, cat, float32, int64
from federated_learning.FLCustomDataset import FLCustomDataset

def load_config(configPath):
    """ Load configuration files.

    Args:
        configPath (str): Path to the config file

    Returns:
        Obj: Corresponding python object
    """    
    configPathAbsolute = os.path.abspath(configPath)
    configs = None
    try:
        with open(configPathAbsolute, 'r') as f:
            configs = yaml.full_load(f)
    except FileNotFoundError:
        print("Config file does not exist.")
        exit(1)
    return configs


def make_output_dir(root_dir, output_prefix):
    output_dir = "{}/{}_{}/".format(root_dir, strftime("%Y%m%d_%H%M%S"), output_prefix)
    logging.info("Creating the output direcotry as {}.".format(output_dir))
    mkdir(output_dir)
    return output_dir


def load_mnist_data_train(data_dir):
    logging.info("Loading train data from MNIST dataset.")
    file_path = "/train-images-idx3-ubyte"
    train_data = dict()
    train_data['x'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.float32)
    
    file_path = "/train-labels-idx1-ubyte"
    train_data['y'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.int64)

    return train_data
    

def load_mnist_data_test(data_dir):
    logging.info("Loading test data from MNIST dataset.")
    file_path = "/t10k-images-idx3-ubyte"
    test_data = dict()
    test_data['x'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.float32)
    
    file_path = "/t10k-labels-idx1-ubyte"
    test_data['y'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.int64)

    return test_data


def preprocess_mnist(dataset):
    logging.info("Preparing the MNIST dataset.")
    # dataset: dict() contains numpy array images and labels
    #    dataset['x'] images
    #    dataset['y'] labels
    max_pixel = dataset['x'].max()
    if max_pixel.max() > 1:
        images = dataset['x'] / 255.0
        dataset['x'] = images
    return dataset


def get_server_dataloader(dataloader, percentage):
    logging.info("Creating server MNIST data loader.")
    # Each batch is supposed to be assigned to a worker. 
    # We just take out a percentage of each batch and save it for the server
    batch_size = None
    server_dataset = dict()
    server_dataset['x'] = tensor([], dtype = float32).reshape(0, 1, 28, 28)
    server_dataset['y'] = tensor([], dtype = int64)
    
    for batch_idx, (data, target) in enumerate(dataloader):
        batch_size = len(data) if batch_size is None else batch_size
        if batch_idx % 100 == 0:
            logging.info('{:.2f}% Loaded...'.format(round((batch_idx * 100) / len(dataloader), 2)))
        server_dataset['x'] = cat((server_dataset['x'], data[:floor(len(data) * (percentage / 100.0))]))
        server_dataset['y'] = cat((server_dataset['y'], target[:floor(len(target) * (percentage / 100.0))]))
        logging.debug("Selecting {} out of {}, Total: [{}]".format(
            floor(len(data) * (percentage / 100.0)), 
            len(data), 
            server_dataset['y'].shape))
    
    return FLCustomDataset(server_dataset['x'], server_dataset['y'])


def get_mnist_dataloader(dataset, batch_size_):
    logging.info("Creating MNIST data loader.")
    return DataLoader(
        FLCustomDataset(
            dataset['x'],
            dataset['y'],
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_ , shuffle=True)
