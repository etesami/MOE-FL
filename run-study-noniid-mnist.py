"""
Usage: 
    run-study-niid-mnist.py\n\t\t(--avg | --opt)\n\t\t--no-attack --output-prefix=NAME [--log] [--nep-log]
    run-study-niid-mnist.py\n\t\t(--avg | --opt)\n\t\t--attack=ATTACK-TYPE --output-prefix=NAME [--log] [--nep-log]
"""
from docopt import docopt
import os
import torch
import random
import neptune
import syft as sy
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import coloredlogs, logging
from torchvision import transforms
from collections import defaultdict
from torch.nn import functional as F
from federated_learning.FLNet import FLNet
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.FederatedLearning import FederatedLearning
from federated_learning.Arguments import Arguments
from federated_learning.helper import utils
CONFIG_PATH = 'configs/defaults.yml'

############ TEMPORARILY ################
arguments = docopt(__doc__)
############ TEMPORARILY ################
# arguments = dict()
# arguments['--log'] = False
arguments['--nep-log'] = False
# arguments['--output-prefix'] = "tmp"
# arguments["--no-attack"] = True
# arguments['--avg'] = True
############ TEMPORARILY ################


def print_model(model):
    for ii, jj in model.named_parameters():
        if ii == "conv1.bias":
            print(jj.data[:7])


def create_workers(hook, workers_idx):
    logging.info("Creating {} workers...".format(len(workers_idx)))
    workers = dict()
    for worker_id in workers_idx:
        logging.debug("Creating the worker: {}".format(worker_id))
        workers[worker_id] = sy.VirtualWorker(hook, id=worker_id)
    logging.info("Creating {} workers..... OK".format(len(workers_idx)))
    return workers

def create_mnist_federated_datasets(raw_dataset, workers):
        """
        raw_datasets (dict)
        ex.
            data: raw_datasets['worker_1']['x']
            label: raw_datasets['worker_1']['y']
        """
        logging.info("Creating the federated dataset for MNIST...")
        fed_datasets = dict()
        for ww_id, ww_data in raw_dataset.items():
            images = torch.tensor([], dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)
            for shard in ww_data:
                images_ = torch.tensor(shard['x'], dtype=torch.float32)
                labels_ = torch.tensor(shard['y'].ravel(), dtype=torch.int64)
                images = torch.cat((images, images_))
                labels = torch.cat((labels, labels_))
            dataset = sy.BaseDataset(
                    images,
                    labels,
                    transform=transforms.Compose([transforms.ToTensor()])
                ).federate([workers[ww_id]])
            fed_datasets[ww_id] = dataset
        logging.info("Creating the federated dataset for MNIST...... OK")
        return fed_datasets
        
def test(model, test_loader, round_no, args):
    model.eval()
    test_loss = 0
    correct = 0
    print()
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    if args.neptune_log:
        neptune.log_metric("test_loss", test_loss)
        neptune.log_metric("test_acc", test_acc)
    if args.local_log:
        file = open(args.log_dir +  "accuracy", "a")
        TO_FILE = '{} {} "{{/*Accuracy:}}\\n{}%" {}\n'.format(
            round_no, test_loss, test_acc, test_acc)
        file.write(TO_FILE)
        file.close()
    
    logging.info('Test Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    print()
    return test_acc



def federate_data(splitted_data, workers):
    idx = [ii for ii in range(len(splitted_data['y']))]
    random.shuffle(idx)
    federated_splitted_data = defaultdict(lambda: [])
    for ii, ww_id in enumerate(workers_idx):
        # Two shard should be given to each worker
        for shard_idx in range(2):
            shard = dict()
            shard['x'] = splitted_data['x'][ii*2 + shard_idx]
            shard['y'] = splitted_data['y'][ii*2 + shard_idx]
            federated_splitted_data[ww_id].append(shard)
    logging.info("Federated data to {} users..... OK".format(len(federated_splitted_data)))
    return federated_splitted_data     


def wieghted_avg_model(weights, models_state_dict):
    layers = dict()
    for layer_name, layer in models_state_dict[list(models_state_dict.keys())[0]].items():
        layer_ = torch.tensor([0.0] * torch.numel(layer), dtype=torch.float32).view(layer.shape)
        layers[layer_name] = layer_
    for ii, (ww_id, model_state) in enumerate(models_state_dict.items()):
        for layer_no, (layer_name, layer_data) in enumerate(model_state.items()):
            layers[layer_name] += weights[ii] * layer_data

    return layers


def save_model(model, name):
    parent_dir = "{}{}".format(args.log_dir, "models")
    if not os.path.isdir(parent_dir):
        logging.debug("Create a directory for model(s).")
        os.mkdir(parent_dir)
    full_path = "{}/{}".format(parent_dir, name)
    logging.debug("Saving the model into " + full_path)
    torch.save(model, full_path)
    
def train_workers(federated_dataloader, workers_model, round_no, args):
    workers_opt = {}
    for ww_id, ww_model in workers_model.items():
        if ww_model.location is None \
                or ww_model.location.id != ww_id:
            ww_model.send(workers[ww_id])
        workers_opt[ww_id] = torch.optim.SGD(
            params=ww_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch_no in range(args.epochs):
        for ww_id, fed_dataloader in federated_dataloader.items():
            if ww_id in workers_model.keys():
                for batch_idx, (data, target) in enumerate(fed_dataloader):
                    worker_id = data.location.id
                    worker_opt = workers_opt[worker_id]
                    workers_model[worker_id].train()
                    data, target = data.to(args.device), target.to(args.device)
                    worker_opt.zero_grad()
                    output = workers_model[worker_id](data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    worker_opt.step()

                    if batch_idx % args.log_interval == 0:
                        loss = loss.get()
                        if args.neptune_log:
                            neptune.log_metric("train_loss_" + str(worker_id), loss)
                        if args.local_log:
                            file = open(args.log_dir + str(worker_id) + "_train", "a")
                            TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch_no, batch_idx, worker_id, loss)
                            file.write(TO_FILE)
                            file.close()
                        logging.info('Train Round: {}, Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            round_no, epoch_no, worker_id, batch_idx, 
                            batch_idx * fed_dataloader.batch_size, 
                            len(fed_dataloader) * fed_dataloader.batch_size,
                            100. * batch_idx / len(fed_dataloader), loss.item()))
    print()
    
if __name__ == '__main__':
    # Initialization
    configs = utils.load_config(CONFIG_PATH)
    args = Arguments(
        configs['runtime']['batch_size'],
        configs['runtime']['test_batch_size'],
        configs['runtime']['rounds'],
        configs['runtime']['epochs'],
        configs['runtime']['lr'],
        configs['runtime']['momentum'],
        configs['runtime']['weight_decay'],
        configs['mnist']['shards_num'],
        configs['runtime']['use_cuda'],
        torch.device("cuda" if configs['runtime']['use_cuda'] else "cpu"),
        configs['runtime']['random_seed'],
        configs['log']['interval'],
        configs['log']['level'],
        configs['log']['format'],
        utils.make_output_dir(
            configs['log']['root_output_dir'], arguments['--output-prefix']
            ) if arguments['--log'] else "",
        True if arguments['--nep-log'] else False,
        True if arguments['--log'] else False
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # syft initialization
    hook = sy.TorchHook(torch)

    # Logging initialization
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=args.log_level, fmt=args.log_format)
    output_dir = None
    if args.local_log:
        utils.save_configs(args.log_dir, configs)

    # Neptune logging initialization
    if args.neptune_log:
        neptune.init(configs['log']['neptune_init'])
        neptune.create_experiment(name = configs['log']['neptune_exp'])
    
    total_num_workers = configs['mnist']['total_number_users']
    logging.info("Total number of users: {}".format(total_num_workers))

    workers_idx = ["worker_" + str(i) for i in range(total_num_workers)]
    workers = create_workers(hook, workers_idx)
    if args.local_log:
        utils.write_to_file(args.log_dir, "all_users", workers_idx)

    train_raw_data = utils.preprocess_mnist(
        utils.load_mnist_data_train(
            configs['mnist']['path'], 
            configs['mnist']['import_fraction']))

    # Let's create the dataset and normalize the data globally
    train_dataset = utils.get_mnist_dataset(train_raw_data)
    
    # Now sort the dataset and distribute among users
    sorted_train_data = utils.sort_mnist_dataset(train_dataset)
    splitted_train_data = utils.split_raw_data(sorted_train_data, args.shards_num)
    federated_train_data = federate_data(splitted_train_data, workers)
    
    fed_train_datasets = None
    if arguments["--no-attack"]:
        logging.info("No Attack will be performed.")
        fed_train_datasets = create_mnist_federated_datasets(federated_train_data, workers)

    # # elif arguments["--attack"] == "99": # Combines
    # #     logging.info("Perform combined attacks 1, 2, 3")
    # #     dataset = utils.perfrom_attack_femnist(
    # #             raw_train_data, 1, workers_idx_to_be_used, eavesdroppers_idx)
    # #     # dataset = utils.perfrom_attack_femnist(
    # #     #         dataset, 2, workers_idx_to_be_used, eavesdroppers_idx)
    # #     dataset = utils.perfrom_attack_femnist(
    # #             dataset, 3, workers_idx_to_be_used, eavesdroppers_idx)
    # #     fed_train_datasets = fl.create_femnist_fed_datasets(dataset, workers_idx_to_be_used)
    # # else:
    # #     logging.info("Perform attack type: {}".format(arguments["--attack"]))
    # #     fed_train_datasets = fl.create_femnist_fed_datasets(
    # #         utils.perfrom_attack_femnist(
    # #             raw_train_data, 
    # #             int(arguments["--attack"]),
    # #             workers_idx_to_be_used,
    # #             eavesdroppers_idx
    # #         ), workers_idx_to_be_used)

    fed_train_dataloaders = dict()
    for ww_id, fed_dataset in fed_train_datasets.items():
        dataloader = sy.FederatedDataLoader(
            fed_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        fed_train_dataloaders[ww_id] = dataloader

    test_raw_data = utils.load_mnist_data_test(configs['mnist']['path'])
    test_dataset = utils.get_mnist_dataset(test_raw_data)
    test_dataloader = utils.get_dataloader(
        test_dataset, args.test_batch_size, shuffle=True, drop_last=False)

    server_model = FLNet().to(args.device)
    selected_users_num = configs['mnist']['selected_users_num']
    
    for round_no in range(args.rounds):
        # select selected_users_num users randomly
        workers_to_be_used = random.sample(workers_idx, selected_users_num)
        logging.info("Some of selected users for this round: {}".format(workers_to_be_used[:3]))
        if args.local_log:
            utils.write_to_file(args.log_dir, "selected_workers", "R{}: {}".format(
                round_no, workers_to_be_used))

        logging.info("Update workers model in this round...")
        workers_model = dict()
        for worker_id in workers_to_be_used:
            workers_model[worker_id] = deepcopy(server_model)
        
        print()
        train_workers(fed_train_dataloaders, workers_model, round_no, args)
        
        # Find the best weights and update the server model
        weights = None
        if arguments['--avg']:
            # Each worker takes two shards of 300 random.samples. Total of 600 random.samples
            # per worker. Total number of random.samples is 60000.
            weights = [600.0 / 60000] * selected_users_num
        # elif arguments['--opt']:
        #     # weights = fl.find_best_weights(trained_server_model, workers_idx)
        #     weights = fl.find_best_weights(trained_w0_model, workers_idx)

        models_state = dict()
        for worker_id, worker_model in workers_model.items():
            if worker_model.location is not None:
                worker_model.get()
            models_state[worker_id] = worker_model.state_dict()
        weighted_avg_state = wieghted_avg_model(weights, models_state)
        logging.info("Update server model in this round...")
        server_model.load_state_dict(weighted_avg_state)

        # Apply the server model to the test dataset
        logging.info("Starting model evaluation on the test dataset...")
        test(server_model, test_dataloader, round_no, args)

        if args.local_log:
            save_model(
                server_model, 
                "R{}_{}".format(round_no, "server_model")
            )

        print("")

