import os
import yaml
import idx2numpy
import numpy as np
import logging
import json
import h5py
from os import mkdir
from random import sample, choice
from time import strftime
from math import floor
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import tensor, cat, float32, int64, randperm, unique
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


def write_to_file(output_dir, file_name, content):
    full_path = "{}/{}".format(output_dir, file_name)
    with open(full_path, "w") as f:
        f.write("{}".format(content))
    f.close

def save_configs(output_dir, configs):
    full_path = "{}/configs".format(output_dir)
    with open(full_path, "w") as f:
        yaml.dump(configs, f, default_flow_style=False)
    f.close


def make_output_dir(root_dir, output_prefix):
    output_dir = "{}/{}_{}/".format(root_dir, strftime("%Y%m%d_%H%M%S"), output_prefix)
    logging.info("Creating the output direcotry as {}.".format(output_dir))
    mkdir(output_dir)
    return output_dir


def get_workers_idx(population, num, excluded_idx):
    idx = []
    while len(idx) < num:
        idx_random = choice(population)
        if idx_random not in (excluded_idx + idx):
            idx.append(idx_random)
    return idx


################ Leaf related functions #################

def read_raw_data(data_path):
    logging.debug("Reading raw data from {}".format(data_path))
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_path)
    files = [f for f in files if f.endswith('.json')]
    
    counter = 1
    for f in files:
        logging.info("Loading {} out of {} files...".format(counter, len(files)))
        file_path = os.path.join(data_path, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.update(cdata['user_data'])
        counter += 1

    return data


def preprocess_leaf_data(data_raw, min_num_samples=100, only_digits=True):
    logging.info("Start processing of femnist data...")
    processed_data = dict()
    for user, user_data in data_raw.items():  
        data_y = np.array(user_data['y'], dtype = np.int64).reshape(-1 , 1)
        filtered_idx = None
        if only_digits:
            filtered_idx = np.where(data_y < 10)[0]
        else:
            filtered_idx = np.where(data_y)[0]
        if len(filtered_idx) < min_num_samples:
            continue
        data_x = np.array(user_data['x'], dtype = np.float32).reshape(-1 , 28, 28)
        processed_data[user] = dict()
        processed_data[user]['x'] = data_x[filtered_idx]
        processed_data[user]['y'] = data_y[filtered_idx]
    return processed_data


def load_leaf_train(data_dir):
    logging.info("Loading train dataset from {}".format(data_dir))
    return read_raw_data(data_dir + "/train")


def load_leaf_test(data_dir):
    logging.info("Loading test dataset from {}".format(data_dir))
    return read_raw_data(data_dir + "/test")


################ MNIST related functions #################

def get_server_mnist_dataset(dataset, workers_num, percentage):
    """ 
    Args:
        dataset (FLCustomDataset): 
        workers_num (int): Total number of workers
        percentage (float): Out of 100
    Returns:
        (FLCustomDataset)
    """  
    logging.info("Creating server MNIST data loader.")
    # Create a temporary DataLoader with adjusted batch_size according to the number of workers.
    # Each batch is supposed to be assigned to a worker. 
    # We just take out a percentage of each batch and save it for the server

    batch_size = int(len(dataset) / workers_num)
    tmp_dataloader = get_dataloader(dataset, batch_size, shuffle=False, drop_last=True)
    
    server_dataset = dict()
    server_dataset['x'] = tensor([], dtype=float32).reshape(0, 1, 28, 28)
    server_dataset['y'] = tensor([], dtype=int64)

    for batch_idx, (data, target) in enumerate(tmp_dataloader):
        # if batch_idx % 100 == 0:
        #     logging.info('{:.2f}% Loaded...'.format(round((batch_idx * 100) / len(dataloader), 2)))
        server_dataset['x'] = cat((server_dataset['x'], data[:floor(len(data) * (percentage / 100.0))]))
        server_dataset['y'] = cat((server_dataset['y'], target[:floor(len(target) * (percentage / 100.0))]))
        logging.debug("Taking {} out of {} from worker {}, Total: [{}]".format(
            floor(len(data) * (percentage / 100.0)), 
            len(data), 
            batch_idx,
            server_dataset['y'].shape))
    
    return FLCustomDataset(server_dataset['x'], server_dataset['y'])


def load_mnist_data_train(data_dir, percentage):
    """ 
    Args:
        data_dir (str): 
        percentage (float): Out of 100, how much of data is imported
    Returns:
        
    """  
    logging.info("Loading {}% of train data from MNIST dataset.".format(percentage))
    file_path = "/train-images-idx3-ubyte"
    train_data = dict()
    train_data['x'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.float32)
    
    # Save only some percentage of data
    train_data['x'] = train_data['x'][:int((float(percentage) / 100.0) * len(train_data['x']))]
    logging.debug("Train data loaded: {}".format(len(train_data['x'])))
    
    file_path = "/train-labels-idx1-ubyte"
    train_data['y'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.int64)
    train_data['y'] = train_data['y'][:int((float(percentage) / 100.0) * len(train_data['y']))]

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
    """ 
    Args:
        dataset (dict of str: numpy array):

    Returns:
        
    """ 
    logging.info("Preparing the MNIST dataset.")
    # dataset['x'] images
    # dataset['y'] labels
    max_pixel = dataset['x'].max()
    if max_pixel.max() > 1:
        images = dataset['x'] / 255.0
        dataset['x'] = images
    return dataset


def get_mnist_dataset(raw_dataset):
    """ 
    Args:
        raw_dataset (list[numpy array]): 
    Returns:
        
    """    
    logging.info("Creating MNIST dataset.")
    return FLCustomDataset(
        raw_dataset['x'],
        raw_dataset['y'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((raw_dataset['x'].mean(),), (raw_dataset['x'].std(),))])
    )


def get_dataloader(dataset, batch_size, shuffle, drop_last):
    """ 
    Args:
        dataset (FLCustomDataset): 
        batch_size (int):
    Returns:
        
    """    
    logging.info("Creating data loader.")
    return DataLoader(
        dataset,
        batch_size=batch_size , shuffle=shuffle, drop_last=drop_last)


def perfrom_attack(dataset, attack_id, workers_idx, evasdropers_idx, percentage=100):
    """ 
    Args:
        dataset (FLCustomDataset): 
        attack_id (int):
            1: shuffle
            2: negative_value
            3: labels
        
        evasdropers_idx (list(int))
        percentage (int): Amount of data affected in each eavesdropper
    Returns:
        dataset (FLCustomDataset)
    """  
    batch_size = int(len(dataset) / len(workers_idx))
    logging.debug("Batch size for {} workers: {}".format(len(workers_idx), batch_size))
    logging.info("Create a temporaily dataloader...")
    tmp_dataloader = get_dataloader(dataset, batch_size, shuffle=False, drop_last=True)
    data_x = tensor([], dtype=float32).reshape(0, 1, 28, 28)
    data_y = tensor([], dtype=int64)
    logging.debug("Attack ID: {}".format(attack_id))
    for idx, (data, target) in enumerate(tmp_dataloader):
        if workers_idx[idx] in evasdropers_idx:
            logging.debug("Find target [{}] for the attack.".format(idx))
            if attack_id == 1:
                logging.debug("Performing attack [shuffle pixels] for user {}...".format(idx))
                data = attack_shuffle_pixels(data)
            elif attack_id == 2:
                logging.debug("Performing attack [negative of pixels] for user {}...".format(idx))
                data = attack_negative_pixels(data)
            elif attack_id == 3:
                logging.debug("Performing attack [shuffle labels] for user {}...".format(idx))
                data = attack_shuffle_labels(target, percentage)
            else:
                logging.debug("NOT EXPECTED: NO VALID ATTACK ID!")
        data_x = cat((data_x, data))
        data_y = cat((data_y, target))
        
    return FLCustomDataset(data_x, data_y)


def attack_shuffle_pixels(data):
    for ii in range(len(data)):
        pixels_flatted = data[ii].reshape(-1)
        rand_idx = randperm(len(pixels_flatted))
        pixels_flatted = pixels_flatted[rand_idx]
        data[ii] = pixels_flatted.reshape(-1, 28, 28)
    return data

def attack_negative_pixels(data):
    for ii in range(len(data)):
        pixels_flatted = data[ii].reshape(-1)
        rand_idx = randperm(len(pixels_flatted))
        pixels_flatted = pixels_flatted[rand_idx]
        data[ii] = pixels_flatted.reshape(-1, 28, 28)
    return data

def attack_shuffle_labels(targets, percentage):
    num_categories = unique(targets)
    percentage = 50
    idx1 = np.array(sample(
        range(len(num_categories)), 
        int(percentage * 0.01 * len(num_categories))), dtype=np.int64)
    idx2 = np.random.permutation(idx1)

    target_map = dict()
    for ii in range(len(idx1)):
        target_map[idx1[ii]] = tensor(idx2[ii])

    logging.debug("Target map for attack: suffle labels:\n".format(target_map))
    logging.debug("target labels before: {}...".format(targets[:10]))
    for ii in range(len(targets)):
        if targets[ii].item() in target_map.keys():
            target[ii] = target_map[targets[ii].item()]
    logging.debug("target labels after: {}...".format(targets[:10]))
    return targets

################ EMNIST (google) related functions #################

def load_femnist_google_data_digits(file_path):
    logging.debug("Reading femnist (google) raw data for {}".format(file_path))
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


def load_femnist_google_test(data_dir):
    file_name = "fed_emnist_digitsonly_test.h5"
    full_path = "{}/{}".format(data_dir, file_name)
    return load_femnist_google_data_digits(full_path)


def load_femnist_google_train(data_dir):
    file_name = "fed_emnist_digitsonly_train.h5"
    full_path = "{}/{}".format(data_dir, file_name)
    return load_femnist_google_data_digits(full_path)