"""
Usage: 
    run-study-niid-mnist.py\n\t\t--epoch=NUM --batch=NUM --output-prefix=NAME [--log] [--nep-log]
"""
# run-study-niid-mnist.py\n\t\t(--avg | --opt)\n\t\t--no-attack --output-prefix=NAME [--log] [--nep-log]
# run-study-niid-mnist.py\n\t\t(--avg | --opt)\n\t\t--attack=ATTACK-TYPE --output-prefix=NAME [--log] [--nep-log]
from docopt import docopt
import os
import torch
import random
import neptune
import syft as sy
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torchvision
import coloredlogs, logging
from torchvision import transforms
from collections import defaultdict
from torch.nn import functional as F
from federated_learning.FLNet import FLNet
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.FederatedLearning import FederatedLearning
from federated_learning.Arguments import Arguments
from federated_learning.helper import utils
from syft.frameworks.torch.fl.utils import federated_avg
CONFIG_PATH = 'configs/defaults.yml'

############ TEMPORARILY ################
arguments = docopt(__doc__)
############ TEMPORARILY ################
# arguments = dict()
# arguments['--output-prefix'] = "tmp"
# arguments["--no-attack"] = True
# arguments['--epoch'] = 1
# arguments['--batch'] = 20
# arguments['--log'] = False
# arguments['--nep-log'] = False
# arguments['--avg'] = True
############ TEMPORARILY ################


def print_model(name, model):
    print()
    for ii, jj in model.named_parameters():
        if ii == "conv1.bias":
            print("{}: {}".format(name, jj.data[:7]))
    print()


def create_workers(hook, workers_idx):
    logging.info("Creating {} workers...".format(len(workers_idx)))
    workers = dict()
    for worker_id in workers_idx:
        logging.debug("Creating the worker: {}".format(worker_id))
        workers[worker_id] = sy.VirtualWorker(hook, id=worker_id)
    logging.info("Creating {} workers..... OK".format(len(workers_idx)))
    return workers


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
    torch.save(model.state_dict(), full_path)
    

def train_workers(federated_train_loader, models, workers_id, round_no, args):
    workers_opt = dict()
    for ww_id in workers_id:
        workers_opt[ww_id] = torch.optim.SGD(
            params=models[ww_id].parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        for ww_id, fed_dataloader in federated_train_loader.items():
            if ww_id in workers_id:
                for batch_idx, (data, target) in enumerate(fed_dataloader): 
                    ww_id = data.location.id
                    model = models[ww_id]
                    model.train()
                    model.send(data.location)
                    data, target = data.to("cpu"), target.to("cpu")
                    workers_opt[ww_id].zero_grad()
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    workers_opt[ww_id].step()
                    model.get() # <-- NEW: get the model back
                    if batch_idx % args.log_interval == 0:
                        loss = loss.get() # <-- NEW: get the loss back
                        if args.neptune_log:
                            neptune.log_metric("train_loss_" + str(ww_id), loss)
                        if args.local_log:
                            file = open(args.log_dir + str(ww_id) + "_train", "a")
                            TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch, batch_idx, ww_id, loss)
                            file.write(TO_FILE)
                            file.close()
                        logging.info('Train Round: {}, Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            round_no, epoch, ww_id, batch_idx, 
                            batch_idx * fed_dataloader.batch_size, 
                            len(fed_dataloader) * fed_dataloader.batch_size,
                            100. * batch_idx / len(fed_dataloader), loss.item()))

    
def train_workers_1(fed_dataloader, models, workers_id, round_no, args):
    workers_opt = dict()
    for ww_id in workers_id:
        workers_opt[ww_id] = torch.optim.SGD(
            params=models[ww_id].parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(fed_dataloader): 
            ww_id = data.location.id
            if ww_id in workers_id:
                model = models[ww_id]
                model.train()
                model.send(data.location)
                data, target = data.to("cpu"), target.to("cpu")
                workers_opt[ww_id].zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                workers_opt[ww_id].step()
                model.get() # <-- NEW: get the model back
                if batch_idx % args.log_interval == 0:
                    loss = loss.get() # <-- NEW: get the loss back
                    if args.neptune_log:
                        neptune.log_metric("train_loss_" + str(ww_id), loss)
                    if args.local_log:
                        file = open(args.log_dir + str(ww_id) + "_train", "a")
                        TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch, batch_idx, ww_id, loss)
                        file.write(TO_FILE)
                        file.close()
                    logging.info('Train Round: {}, Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        round_no, epoch, ww_id, batch_idx, 
                        batch_idx * fed_dataloader.batch_size, 
                        len(fed_dataloader) * fed_dataloader.batch_size,
                        100. * batch_idx / len(fed_dataloader), loss.item()))


if __name__ == '__main__':
    # Initialization
    configs = utils.load_config(CONFIG_PATH)
    # configs['runtime']['epochs'] = int(arguments['--epoch'])
    # configs['mnist']['selected_users_num'] = int(arguments['--workers'])
    configs['runtime']['batch_size'] = int(arguments['--batch'])
    configs['runtime']['test_batch_size'] = int(arguments['--batch'])
    configs['runtime']['epochs'] = int(arguments['--epoch'])
    # configs['log']['interval'] = int(arguments['--interval'])
    args = Arguments(
        configs['runtime']['batch_size'],
        configs['runtime']['test_batch_size'],
        configs['runtime']['rounds'],
        configs['runtime']['epochs'],
        configs['runtime']['lr'],
        configs['runtime']['momentum'],
        configs['runtime']['weight_decay'],
        configs['mnist']['shards_num'],
        configs['mnist']['shards_per_worker_num'],
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
        neptune.init(project_qualified_name=configs['log']['neptune_init'])
        neptune.create_experiment(name=configs['log']['neptune_exp'])
    
    total_num_workers = configs['mnist']['total_users_num']
    logging.info("Total number of users: {}".format(total_num_workers))

    workers_idx = ["worker_" + str(i) for i in range(total_num_workers)]
    workers = create_workers(hook, workers_idx)
    # if args.local_log:
    #     utils.write_to_file(args.log_dir, "all_users", workers_idx)
    server = create_workers(hook, ["server"])['server']
    train_dataset = utils.load_mnist_data_train()
    
    # Now sort the dataset and distribute among users
    sorted_train_dataset = utils.sort_mnist_dataset(train_dataset)
    splitted_train_dataset = utils.split_dataset(
        sorted_train_dataset, int(len(sorted_train_dataset) / args.shards_num))
    mapped_train_datasets = utils.map_shards_to_worker(splitted_train_dataset, workers, args.shards_per_worker_num)

    server_data_train = utils.fraction_of_datasets(
        mapped_train_datasets, configs['server']['data_fraction'])

    # federated_train_loader = dict()
    # for ww_id, fed_dataset in federated_train_datasets.items():
    federated_train_loader = sy.FederatedDataLoader(
        server_data_train.federate([server]), batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    test_dataset = utils.load_mnist_data_test(configs['mnist']['path'])
    test_loader = utils.get_dataloader(
        test_dataset, args.test_batch_size, shuffle=True, drop_last=False)

    server_model = FLNet().to(args.device)
    print_model('Initial Server Model', server_model)
    selected_users_num = 1 # only server
    # selected_users_num = configs['mnist']['selected_users_num']

    for round_no in range(args.rounds):
        # workers_to_be_used = random.sample(workers_idx, selected_users_num)
        workers_to_be_used = ["server"]
        workers_model = dict()
        # for ww_id in workers_to_be_used:
        #     workers_model[ww_id] = deepcopy(server_model)

        # logging.info("Some of selected users for this round: {}".format(workers_to_be_used))
        # print_model("worker before training", list(workers_model.values())[0])
        # if args.local_log:
        #     utils.write_to_file(args.log_dir, "selected_workers", "R{}: {}".format(
        #         round_no, workers_to_be_used))
        train_workers_1(federated_train_loader, {"server" : server_model}, workers_to_be_used, round_no, args)

        # # Find the best weights and update the server model
        # weights = None
        # if arguments['--avg']:
        #     # Each worker takes two shards of 300 random.samples. Total of 600 random.samples
        #     # per worker. Total number of random.samples is 60000.
        #     # weights = [600.0 / 60000] * selected_users_num
        #     weights = [1.0 / selected_users_num] * selected_users_num
        # # elif arguments['--opt']:
        # #     # weights = fl.find_best_weights(trained_server_model, workers_idx)
        # #     weights = fl.find_best_weights(trained_w0_model, workers_idx)
        # print_model("server before update", server_model)
        # models_state = dict()
        # for worker_id, worker_model in workers_model.items():
        #     if worker_model.location is not None:
        #         worker_model.get()
        #     models_state[worker_id] = worker_model.state_dict()

        # weighted_avg_state = wieghted_avg_model(weights, models_state)
        # logging.info("Update server model in this round...")
        # server_model.load_state_dict(weighted_avg_state)
        # print_model("server after update", server_model)

        # Apply the server model to the test dataset
        logging.info("Starting model evaluation on the test dataset...")
        test(server_model, test_loader, round_no, args)

        if args.local_log:
            save_model(
                server_model, 
                "R{}_{}".format(round_no, "server_model")
            )

        print("")
    
    