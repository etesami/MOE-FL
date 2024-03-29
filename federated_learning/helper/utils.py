import os
import yaml
import numpy as np
import logging
import json
import h5py
import cvxpy as cp
import pickle
import syft as sy
import subprocess
import random
from random import sample
from time import strftime
from math import floor
from collections import defaultdict
from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader
from torch import tensor, float32, int64, unique, norm, dot, numel
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.AddGaussianNoise import AddGaussianNoise

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


def print_model(name, model):
    print()
    for ii, jj in model.named_parameters():
        if ii == "conv1.bias":
            print("{}: {}".format(name, jj.data[:7]))
    print()


def wieghted_avg_model(weights, workers_model, workers_idx):
    layers = dict()
    for layer_name, layer in list(workers_model.values())[0].state_dict().items():
        layer_ = tensor([0.0] * numel(layer), dtype=float32).view(layer.shape)
        layers[layer_name] = layer_
    for ww_id in workers_idx:
        for layer_no, (layer_name, layer_data) in enumerate(workers_model[ww_id].state_dict().items()):
            layers[layer_name] += weights[ww_id] * layer_data
    return layers


def negative_parameters(model_params):
    for layer_name, layer_param in model_params.items():
        layer_param *= -1.0
    return model_params


def check_write_to_file(output_dir, file_name, content):
    full_path = "{}/{}".format(output_dir, file_name)
    if not os.path.isfile(full_path):
        write_to_file(output_dir, file_name, content)


def write_to_file(output_dir, file_name, content, round_no=None):
    full_path = "{}/{}".format(output_dir, file_name)
    output = "{}\n".format(content) if round_no is None else "{} {}\n".format(round_no, content)
    with open(full_path, "a") as f:
        f.write(output)
    f.close


def check_save_configs(output_dir, configs):
    full_path = "{}/configs".format(output_dir)
    if os.path.isfile(full_path):
        logging.info("Config file exists.")
    else:
        logging.info("Copying config file...")
        with open(full_path, "w") as f:
            yaml.dump(configs, f, default_flow_style=False)
        f.close


def get_last_round_num(output_dir, file_name):
    full_path = "{}/{}".format(output_dir, file_name)
    line = subprocess.check_output(['tail', '-1', full_path]).decode('utf-8')
    return line.split(' ')[0]


def check_create_output_dir(parent_log_dir_name, output_prefix):
    full_root_path = "{}/{}".format(get_app_root_dir(), parent_log_dir_name)
    target_folder = [ff for ff in os.listdir(full_root_path) if output_prefix in ff]
    if len(target_folder) == 1:
        logging.info("A prevoiusly created log directory was found: {}".format(target_folder[0]))
        output_dir = "{}/{}/".format(parent_log_dir_name, target_folder[0])
        return output_dir
    else:
        if len(target_folder) > 1:
            logging.warning("More than one prevoiusly created log directory were found!")
        else:
            return make_output_dir(parent_log_dir_name, output_prefix)


def get_app_root_dir():
    return os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))


def make_output_dir(parent_log_dir_name, output_prefix):
    output_dir = "{}/{}_{}/".format(parent_log_dir_name, strftime("%Y%m%d_%H%M%S"), output_prefix)
    logging.info("Creating the output direcotry as {}.".format(output_dir))
    os.mkdir(output_dir)
    return output_dir


def check_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        return True
    else:
        return False


def get_workers_idx(population, num, excluded_idx):
    return get_subset(population, num, excluded_idx)

def get_subset(population, num, excluded_idx):
    idx = []
    while len(idx) < num:
        idx_random = random.choice(population)
        if idx_random not in (excluded_idx + idx):
            idx.append(idx_random)
    return idx


def extract_data(raw_data, workers_idx):
    logging.info("Extract data from raw data for {} of users...".format(len(workers_idx)))
    data = dict()
    for ww_id, ww_data in raw_data.items():
        if ww_id in workers_idx:
            data[ww_id] = ww_data
    return data


def get_flattened_data(data):
    data_flattened_x = np.array([], dtype = np.float32).reshape(0, 28 * 28)
    tmp_array = [np.array(data_['x'], dtype = np.float32).reshape(-1, 28 * 28) for data_ in data.values()]
    for dd in tmp_array:
        data_flattened_x = np.concatenate((data_flattened_x, dd))
    data_flattened_y = np.array([], dtype = np.float32)
    tmp_array_y = [np.array(data_['y'], dtype = np.int64).reshape(-1) for data_ in data.values()]
    for dd in tmp_array_y:
        data_flattened_y = np.concatenate((data_flattened_y, dd))

    return data_flattened_x, data_flattened_y


def split_randomly_dataset(dataset, shards_num):
    """Split whole dataset into `shards_num` categories and 
    randomly return rearrange their position.

    Args:
        dataset ([torch.dataset]): 
        shards_num (int): Number of shards

    Yields:
        [Iterable[FLCustomDataset]]: Returning a generator of FLCustomDDataset
    """    
    samples_per_shards_num = int(len(dataset) / shards_num)
    logging.info(
        "Splitting the dataset into {} groups, each with {} samples...".format(shards_num, samples_per_shards_num))
    idx = [ii for ii in range(shards_num)]
    random.shuffle(idx)
    splitted_data = torch.split(dataset.data, samples_per_shards_num)
    splitted_targets = torch.split(dataset.targets, samples_per_shards_num)
    for ii in range(len(idx)):
        yield FLCustomDataset(
                splitted_data[idx[ii]],
                splitted_targets[idx[ii]],
                transform=transforms.Compose([
                    transforms.ToTensor()])
            )


def dataset_info(dataset):
    list_keys = list(dataset.keys())
    numbers = dict()
    # numbers[num_samples] = num_users
    for uu, dd in dataset.items():
        key = str(len(dd['y']))
        if key in numbers:
            numbers[key] += 1
        else:
            numbers[key] = 1
        
    total_samples = 0
    for uu in sorted(numbers.keys()):
        print("{}:\t{}".format(uu, numbers[uu]))
        total_samples += int(uu) * int(numbers[uu])

    print("Mean num of samples/user: {}".format(
        round(np.mean([int(ii) for ii in numbers])), 2))
    print("Total Samples:\t{}".format(total_samples))
    print("Total Users:\t{}".format(len(list_keys)))
    print("[{}]: Images: {}, Pixels: {}".format(
        list_keys[0], 
        len(dataset[list_keys[0]]['x']), 
        len(dataset[list_keys[0]]['x'][0])))
    data_flatted_x, data_flatted_y = get_flattened_data(dataset)
    print("mean: {}\nstd: {},\nmax: {}".format(
            data_flatted_x.mean(), 
            data_flatted_x.std(), 
            data_flatted_x.max()))
    print("-"*5)


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
        server_dataset['x'] = torch.cat((server_dataset['x'], data[:floor(len(data) * (percentage / 100.0))]))
        server_dataset['y'] = torch.cat((server_dataset['y'], target[:floor(len(target) * (percentage / 100.0))]))
        logging.debug("Taking {} out of {} from worker {}, Total: [{}]".format(
            floor(len(data) * (percentage / 100.0)), 
            len(data), 
            batch_idx,
            server_dataset['y'].shape))
    
    return FLCustomDataset(server_dataset['x'], server_dataset['y'])


# def load_mnist_data_train():
#     """ 
#     Args:
#         data_dir (str): 
#         fraction (float): Out of 1, how much of data is imported
#     Returns:
        
#     """  
#     logging.info("Loading train data from MNIST dataset...")
#     # file_path = "/train-images-idx3-ubyte"
#     # train_data = dict()
#     # train_data['x'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.float32)
    
#     # train_data['x'] = train_data['x'][:int(float(fraction) * len(train_data['x']))]
#     # file_path = "/train-labels-idx1-ubyte"
#     # train_data['y'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.int64)
#     # train_data['y'] = train_data['y'][:int(float(fraction) * len(train_data['y']))]

#     return datasets.MNIST("/tmp/data", train=True, download=True,
#                         transform=transforms.Compose([
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.1307,), (0.3081,))]))

def load_mnist_dataset(train=True, transform=None):
    """ 
    Args:
        data_dir (str): 
        fraction (float): Out of 1, how much of data is imported
    Returns:
        
    """  
    logging.info("Loading MNIST [train: {}] dataset...".format(train))
    return datasets.MNIST("/tmp/data", train=train, download=True, transform=transform)


# def load_mnist_data_test():
#     logging.info("Loading test data from MNIST dataset.")
#     # file_path = "/t10k-images-idx3-ubyte"
#     # test_data = dict()
#     # test_data['x'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.float32)
    
#     # file_path = "/t10k-labels-idx1-ubyte"
#     # test_data['y'] = idx2numpy.convert_from_file(data_dir + file_path).astype(np.int64)

#     return datasets.MNIST("/tmp/data", train=False, download=True, 
#                         transform=transforms.Compose([
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.1307,), (0.3081,))]))


# def preprocess_mnist(dataset):
#     """ 
#     Args:
#         dataset (dict of str: numpy array):

#     Returns:
        
#     """ 
#     logging.info("Preparing the MNIST dataset.")
#     # dataset['x'] images
#     # dataset['y'] labels
#     max_pixel = dataset['x'].max()
#     if max_pixel.max() > 1:
#         images = dataset['x'] / 255.0
#         dataset['x'] = images
#     logging.info("Preparing the MNIST dataset..... OK")
#     return dataset


def sort_mnist_dataset(dataset):
    """ 
    Args:
        dataset (torch.dataset):

    Returns:
        
    """ 
    logging.info("Sorting the MNIST dataset based on labels...")
    sorted_index = sorted(range(len(dataset.targets)), key=lambda k: dataset.targets[k])
    return FLCustomDataset(
        dataset.data[sorted_index],
        dataset.targets[sorted_index],
        transform=transforms.Compose([
            transforms.ToTensor()])
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


# def perfrom_attack(dataset, attack_id, workers_idx, evasdropers_idx, percentage=100):
#     """ 
#     Args:
#         dataset (FLCustomDataset): 
#         attack_id (int):
#             1: shuffle
#             2: negative_value
#             3: labels
        
#         evasdropers_idx (list(int))
#         percentage (int): Amount of data affected in each eavesdropper
#     Returns:
#         dataset (FLCustomDataset)
#     """  
#     batch_size = int(len(dataset) / len(workers_idx))
#     logging.debug("Batch size for {} workers: {}".format(len(workers_idx), batch_size))
#     logging.info("Create a temporaily dataloader...")
#     tmp_dataloader = get_dataloader(dataset, batch_size, shuffle=False, drop_last=True)
#     data_x = tensor([], dtype=float32).reshape(0, 1, 28, 28)
#     data_y = tensor([], dtype=int64)
#     logging.debug("Attack ID: {}".format(attack_id))
#     for idx, (data, target) in enumerate(tmp_dataloader):
#         if workers_idx[idx] in evasdropers_idx:
#             logging.debug("Find target [{}] for the attack.".format(idx))
#             if attack_id == 1:
#                 logging.debug("Performing attack [shuffle pixels] for user {}...".format(idx))
#                 data = attack_shuffle_pixels(data)
#             elif attack_id == 2:
#                 logging.debug("Performing attack [negative of pixels] for user {}...".format(idx))
#                 data = attack_negative_pixels(data)
#             elif attack_id == 3:
#                 logging.debug("Performing attack [shuffle labels] for user {}...".format(idx))
#                 data = attack_shuffle_labels(target, percentage)
#             else:
#                 logging.debug("NOT EXPECTED: NO VALID ATTACK ID!")
#         data_x = cat((data_x, data))
#         data_y = cat((data_y, target))
        
#     return FLCustomDataset(data_x, data_y)


# def perform_attack_noniid(datasets, workers_idx, attackers_idx, attack_type, attackers_num):
#     """ 
#     Args:
#         dataset (FLCustomDataset): 
#         attack_id (int):
#             0: No Attack
#             1: shuffle
#             2: negative_value
#             3: labels
        
#         evasdropers_idx (list(int))
#         percentage (int): Amount of data affected in each eavesdropper
#     Returns:
#         dataset (FLCustomDataset)
#     """  
#     new_datasets = dict()
#     logging.info("Attack ID: {}".format(attack_type))
#     if attack_type == 0:
#         logging.info("Attack ID: {}: No Attack is performed...".format(attack_type))
#         # data = attack_shuffle_pixels(data)
#     else:
#         for ww_id, dataset in datasets.items():
#             data = dataset.data
#             targets = dataset.targets
#             if ww_id in attackers_idx:
#                 logging.info("Performing attack on {}...".format(ww_id))
#                 if attack_type == 1:
#                     logging.debug("Performing attack [shuffle pixels] on {} workers...".format(attackers_num))
#                     # data = attack_shuffle_pixels(dataset.data)
#                 elif attack_type == 2:
#                     logging.debug("Performing attack [negative of pixels] on {} workers...".format(attackers_num))
#                     # data = attack_negative_pixels(dataset.data)
#                 elif attack_type == 3:
#                     logging.debug("Performing attack [shuffle labels] on {} workers...".format(attackers_num))
#                     # targets = attack_shuffle_labels_tensor(dataset.targets, 1.0)
#                     new_datasets[ww_id] = FLCustomDataset(
#                         data, targets, transform=transforms.Compose([
#                             transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), AddGaussianNoise(0., 3.0),]))
#                 else:
#                     logging.debug("NOT EXPECTED: NO VALID ATTACK ID!")
#             else:
#                 new_datasets[ww_id] = dataset
        
#     return new_datasets


# def normalize_weights(models_dicts):
#     logging.debug("")
#     logging.debug("Normalization of models states ------ ")

#     workers_params = {}
#     for worker_id, ww_states in models_dicts.items():
#         workers_params[worker_id] = [[] for i in range(8)]
#         for layer_id, (layer_name, params) in enumerate(ww_states.items()):
#             workers_params[worker_id][layer_id] = params.view(-1, 1)
#             logging.debug("workers_params[{}][{}]: {}".format(worker_id, layer_id, workers_params[worker_id][layer_id].shape))

#     logging.debug("")
#     workers_all_params = [[] for i in range(8)]
#     for ii in range(8):
#         for worker_id, worker_model in workers_params.items():
#             workers_all_params[ii].append(workers_params[worker_id][ii])
#         logging.debug("All Params[{}]: {} (num of workers)".format(ii, len(workers_all_params[ii])))

#     logging.debug("")
#     for ii in range(8):
#         workers_all_params[ii] = cat(workers_all_params[ii])
#         logging.debug("Concatenated All Params[{}]: {}, Mean: {}".format(
#             ii, workers_all_params[ii].shape, workers_all_params[ii].mean()))

#     logging.debug("")
#     #TODO Check other types of normalization
#     normalized_workers_all_params = []
#     for ii in range(len(workers_all_params)):
#         norm = MinMaxScaler().fit(workers_all_params[ii])
#         normalized_workers_all_params.append(norm.transform(workers_all_params[ii]))
#         logging.debug("Normalized Concatenated All Params[{}]: {}, Mean: {}".format(
#             ii, normalized_workers_all_params[ii].shape, normalized_workers_all_params[ii].mean()))

#     logging.debug("")
#     mapped_normalized = defaultdict(lambda: [])
#     for ww_no, worker_id in enumerate(workers_params.keys()):
#         tmp_state = dict()
#         for ii, (layer_name, params) in enumerate(models_dicts[worker_id].items()):
#             start_idx = ww_no * len(workers_params[worker_id][ii])
#             end_idx = (ww_no+1) * len(workers_params[worker_id][ii])
#             tmp_state[layer_name] = tensor(normalized_workers_all_params[ii][start_idx:end_idx, :]).view(params.shape)
#             logging.debug("Extracting for {}, Layer {}, from {}-{}: {}".format(
#                 worker_id, layer_name, start_idx, end_idx, tmp_state[layer_name].shape))
        
#         mapped_normalized[worker_id] = tmp_state

#     return mapped_normalized


def find_best_weights(referenced_model, workers_model):
    logging.debug("Finding Best Weigthe ----")

    ref_state = referenced_model.state_dict()

    exp_var = dict()
    for ww_id, ww_model in workers_model.items():
        a = 0
        for layer_name, layer_param in ww_model.state_dict().items():
            a += (dot(
                    layer_param.view(-1),  
                    ref_state[layer_name].view(-1) 
                )) / np.linalg.norm(layer_param)
        a = a/8.0
        exp_var[ww_id] = a.numpy()

    rho = dict()
    for ww_n in workers_model.keys():
        rho[ww_n] = np.exp(exp_var[ww_n])/sum(np.exp(list(exp_var.values())))
        # rho[ww_n] = (exp_var[ww_n]**2)/sum([ll**2 for ll in list(exp_var.values())])
        # print("{}:\t{}".format(ww_n, rho[ww_n]))

    # print("------ SUM: {}".format(sum(list(rho.values()))))
    return rho


def find_best_weights_opt(reference_model, workers_model):
    logging.debug("")
    logging.debug("Finding Best Weigthe ----")
    
    ref_state = reference_model.state_dict()

    workers_all_params = tensor([])
    for ww_n, (ww_id, model) in enumerate(workers_model.items()):
        tmp = []
        for ii, (layer_name, layer_param) in enumerate(model.state_dict().items()):
            if ii == 7:
                logging.debug("working on {} Layer {}, with {}".format(ww_id, layer_name, layer_param.shape))
                tmp.append(layer_param.view(-1, 1))
                # logging.info("working on {} Layer {}, with {}/{}".format(ww_id, layer_name, layer_param.shape, tmp[ii].shape))
        workers_all_params = torch.cat((workers_all_params, torch.cat(tmp, axis=0)), dim=1)
        logging.debug("-- workers_all_params[{}]:\t{}".format(ww_n, workers_all_params.shape))
    
    # logging.info("")

    # reference_layer = []
    tmp = []
    for ii, (layer_name, layer_param) in enumerate(ref_state.items()):
        if ii == 7:
            tmp.append(layer_param.view(-1, 1))
    reference_layer = torch.cat(tmp, axis=0)
    logging.debug("-- reference_layer: \t{}".format(reference_layer.shape))

    reference_layers = reference_layer.repeat(1, len(workers_model))
    logging.debug("-- reference_layers: \t{}".format(reference_layers.shape))

    # # #TODO: Check this
            # 1.0/len(workers_model) * 
    W = cp.Variable(len(workers_model))
    objective = cp.Minimize(
                (cp.matmul(
                    cp.power(
                        cp.norm2(workers_all_params - reference_layers, axis=0), 2), W))
            )

    constraints = [0.0 <= W, W <= 1.0, sum(W) == 1.0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.MOSEK)
    # logging.info("")
    rho = dict()
    for ii, ww_id in enumerate(workers_model.keys()):
        rho[ww_id] = W.value[ii]
        # logging.info("Optimized weights [{}]: {}".format(ww_id, W.value[ii]))
    return rho


def load_object(parent_path, file_name):
    full_path = "{}/{}".format(parent_path, file_name)
    with open(full_path, 'rb') as input:
        return pickle.load(input)


def save_model(model_state, parent_dir, name):
    # parent_dir = "{}/{}".format(output_dir, "models")
    if not os.path.isdir(parent_dir):
        logging.debug("Create a directory for model(s).")
        os.mkdir(parent_dir)
    full_path = "{}/{}".format(parent_dir, name)
    logging.debug("Saving the model into " + full_path)
    torch.save(model_state, full_path)


def save_object(output_dir, file_name, obj):
    logging.debug("Writing object to {}".format(file_name))
    full_path = "{}/{}".format(output_dir, file_name)
    with open(full_path, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file, pickle.HIGHEST_PROTOCOL)


def find_file(parent_path, file_name):
    full_path = "{}/{}".format(parent_path, file_name)
    return True if os.path.isfile(full_path) else False


def map_shards_to_worker(splitted_datasets, workers_idx, num_shards_per_worker):
    federated_datasets = defaultdict(lambda: [])
    for ii, ww_id in enumerate(workers_idx):
        images, labels = [], []
        # Two shard should be given to each worker
        for shard_idx in range(num_shards_per_worker):
            ds = next(splitted_datasets)
            images.append(ds.data)
            labels.append(ds.targets)
        if num_shards_per_worker > 1:
            for ii in range(num_shards_per_worker-1):
                images = torch.cat((images[ii], images[ii+1]))
                labels = torch.cat((labels[ii], labels[ii+1]))
        else:
            images = images[0]
            labels = labels[0]
        yield {ww_id: FLCustomDataset(
            images,labels, 
            transform=transforms.Compose([
                transforms.ToTensor()]))}


def fraction_of_datasets(datasets, fraction, attackers_idx=[]):
    """Extract a fraction of data from each dataset and return 
    the aggregated data as a FLCustomDataset.

    Args:
        datasets (dict[FLCustomDataset]): 
        fraction (float): Fraction between 0.0 and 1.0

    Returns:
        [FLCustomDataset]:
    """    
    logging.info("Extracting {}% of users data (total: {}) to be sent to the server...".format(
        fraction * 100.0, int(fraction * len(datasets) * len(list(datasets.values())[0].targets))))
    images, labels = [], []
    for ww_id, dataset in datasets.items():
        idx = torch.randperm(len(dataset.targets))[:int(fraction * len(dataset.targets))]
        if ww_id in attackers_idx:
            images.append(
                (dataset.data[idx.tolist()] +
                np.random.randint(0, 1024, (len(idx),28,28))).byte()
            )
        else:
            images.append(dataset.data[idx.tolist()])
        labels.append(dataset.targets[idx.tolist()])
    aggregate_dataset = FLCustomDataset(
        torch.cat(images), torch.cat(labels),
        transform=transforms.Compose([
            transforms.ToTensor()])
    )
    logging.info("Extracted... Ok, The size of the extracted data: {}".format(
        aggregate_dataset.data.shape))
    return aggregate_dataset


def merge_and_shuffle_dataset(datasets):
    images, labels = [], []
    for dataset in datasets:
        images.append(dataset.data)
        labels.append(dataset.targets)
    images, labels = torch.cat(images), torch.cat(labels)
    return shuffle_dataset(FLCustomDataset(
        images, labels, transform=transforms.Compose(
            [transforms.ToTensor()])))

    
def shuffle_dataset(dataset):
    new_data, new_labels = shuffle_data_labels(dataset.data, dataset.targets)
    return FLCustomDataset(
        new_data, new_labels, 
        transform=transforms.Compose([
            transforms.ToTensor()]))


def shuffle_data_labels(data, labels):
    # data.shape [x, 28, 28]
    # labels.shape [x]
    rand_idx = torch.randperm(len(labels))
    new_data = data[rand_idx]
    new_labels = labels[rand_idx]
    return new_data, new_labels
    

def attack_shuffle_pixels(data):
    for ii in range(len(data)):
        pixels_flattened = data[ii].view(-1)
        rand_idx = torch.randperm(len(pixels_flattened))
        pixels_flattened = pixels_flattened[rand_idx]
        data[ii] = pixels_flattened.reshape(-1, 28, 28)
    return data


def attack_negative_pixels(data):
    for ii in range(len(data)):
        pixels_flattened = data[ii].view(-1)
        negative_pixels = np.array([1 - pp for pp in pixels_flattened], dtype = np.float32)
        data[ii] = negative_pixels.reshape(-1, 28, 28)
    return data


def attack_black_pixels(data):
    data = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype = np.float32)
    return data


def attack_shuffle_labels(targets, fraction):
    num_categories = torch.unique(targets).tolist()
    num_categories = [int(ii) for ii in num_categories]
    idx1 = get_subset(num_categories, fraction*len(num_categories), [])
    
    # Permute idx1 as idx2
    idx2 = []
    while len(idx2) < len(idx1):
        ii = len(idx2)
        nn = random.choice(idx1)
        if idx1[ii] != nn and nn not in idx2:
            idx2.append(int(nn))
    idx2 = np.array(idx2)
    
    logging.debug("Attack shuffel idx1: {}".format(idx1))
    logging.debug("Attack shuffel idx2: {}".format(idx2))

    target_map = dict()
    for ii in range(len(idx1)):
        target_map[idx1[ii]] = idx2[ii]

    logging.debug("Target map for attack suffle labels: {}".format(target_map))
    logging.debug("target labels before:\t{}...".format(targets[:10]))
    targets_np = targets.detach().clone().numpy()
    for ii in range(len(targets)):
        if targets_np[ii] in target_map.keys():
            targets_np[ii] = target_map[targets_np[ii]]
    logging.debug("target labels after:\t{}...".format(targets_np[:10]))
    return torch.tensor(targets_np, dtype=float32)


# def attack_shuffle_labels(targets, fraction):
#     num_categories = unique(targets)
#     idx1 = get_subset(num_categories, fraction*len(num_categories), [])
#     logging.debug(idx1)

#     # Permute idx1 as idx2
#     idx2 = tensor(
#         get_subset(idx1, len(idx1), []), 
#         dtype=int64)
    
#     logging.debug("Attack shuffel idx1: {}".format(idx1))
#     logging.debug("Attack shuffel idx2: {}".format(idx2))

#     target_map = dict()
#     for ii in range(len(idx1)):
#         target_map[idx1[ii]] = idx2[ii]

#     logging.debug("Target map for attack suffle labels: {}".format(target_map))
#     logging.debug("target labels before:\t{}...".format(targets[:15].view(-1)))
#     for ii in range(len(targets)):
#         if targets[ii][0] in target_map.keys():
#             targets[ii] = target_map[targets[ii][0]]
#     logging.debug("target labels after:\t{}...".format(targets[:10].view(-1)))
#     return targets


def get_neptune_token():
    return "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNmRhYmZmM2YtZDc5Yi00ZGMyLWE5MWEtMjVkMDEwOGNjYTliIn0="

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


def perfrom_attack_femnist(raw_data, attack_id, workers_idx, evasdropers_idx, percentage=100):
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
    logging.info("Attack ID: {}".format(attack_id))
    logging.info("Performing attack on {}".format(evasdropers_idx))
    for worker_id in workers_idx:
        if worker_id in evasdropers_idx:
            if attack_id == 1:
                logging.debug("Performing attack [shuffle pixels] for user {}...".format(worker_id))
                raw_data[worker_id].update({'x': attack_shuffle_pixels(raw_data[worker_id]['x'])})
            elif attack_id == 2:
                logging.debug("Performing attack [negative of pixels] for user {}...".format(worker_id))
                raw_data[worker_id]['x'] = attack_negative_pixels(raw_data[worker_id]['x'])
            elif attack_id == 3:
                logging.debug("Performing attack [shuffle labels] for user {}...".format(worker_id))
                raw_data[worker_id]['y'] = attack_shuffle_labels(raw_data[worker_id]['y'], percentage)
            elif attack_id == 4:
                logging.debug("Performing attack [black pixels] for user {}...".format(worker_id))
                raw_data[worker_id]['x'] = attack_black_pixels(raw_data[worker_id]['x'])
            else:
                logging.debug("NOT EXPECTED: NO VALID ATTACK ID!")

    return raw_data


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