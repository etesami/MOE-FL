"""
Usage: 
    run-study-niid-mnist.py (--avg | --opt) [--attack-type=ID] [--epochs=NUM] [--rounds=NUM] [--selected-workers=NUM] [--attackers-num=num] [--output-prefix=NAME] [--log] [--nep-log] 
"""
from docopt import docopt
import os
import torch
import random
import neptune
import syft as sy
import numpy as np
from tqdm import tqdm
import cvxpy as cp
from time import sleep
from copy import deepcopy
import torchattacks
import torchvision
import coloredlogs, logging
from torchvision import transforms
from collections import defaultdict
from torch.nn import functional as F
from federated_learning.FLNet import FLNet
from federated_learning.FLCustomDataset import FLCustomDataset
# from federated_learning.FederatedLearning import FederatedLearning
from federated_learning.Arguments import Arguments
from federated_learning.helper import utils
CONFIG_PATH = 'configs/defaults.yml'
TQDM_R_BAR = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}] '

############ TEMPORARILY ################
arguments = docopt(__doc__)
############ TEMPORARILY ################
# arguments = dict()
# arguments['--output-prefix'] = "tmp"
# arguments['--epoch'] = 1
# arguments['--log'] = False
# arguments['--nep-log'] = False
# arguments['--avg'] = False
# arguments['--opt'] = True
# arguments['--workers'] = 2
# arguments['--rounds'] = 1
# arguments['--attackers-num'] = 70
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
    with tqdm(total=len(test_loader), ncols=80, leave=False, desc="Test\t", bar_format=TQDM_R_BAR) as t1:
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(args.device), target.to(args.device, dtype=torch.int64)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                t1.update()

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
    
    logging.debug('Test Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    return test_loss, test_acc


def save_model(model, name):
    parent_dir = "{}{}".format(args.log_dir, "models")
    if not os.path.isdir(parent_dir):
        logging.debug("Create a directory for model(s).")
        os.mkdir(parent_dir)
    full_path = "{}/{}".format(parent_dir, name)
    logging.debug("Saving the model into " + full_path)
    torch.save(model.state_dict(), full_path)
    

# def train_workers(federated_train_loader, models, workers_id, attackers_idx, round_no, args):
#     workers_opt = dict()
#     for ww_id in workers_id:
#         workers_opt[ww_id] = torch.optim.SGD(
#             params=models[ww_id].parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     with tqdm(
#         total=args.epochs, leave=False, colour="yellow", ncols=80, desc="Epoch\t", bar_format=TQDM_R_BAR) as t2:
#         for epoch in range(args.epochs):
#             t2.set_postfix(Rounds=round_no, Epochs=epoch)
#             with tqdm(total=len(workers_id), ncols=80, desc="Workers\t", leave=False, bar_format=TQDM_R_BAR) as t3:
#                 for ww_id, fed_dataloader in federated_train_loader.items():
#                     if ww_id in workers_id:
#                         with tqdm(total=len(fed_dataloader), ncols=80, colour='red', desc="Batch\t", leave=False, bar_format=TQDM_R_BAR) as t4:
#                             for batch_idx, (data, target) in enumerate(fed_dataloader): 
#                                 ww_id = data.location.id
#                                 model = models[ww_id]
#                                 model.train()
#                                 model.send(data.location)
#                                 data, target = data.to("cpu"), target.to("cpu")
#                                 workers_opt[ww_id].zero_grad()
#                                 output = model(data)
#                                 loss = F.nll_loss(output, target)
#                                 loss.backward()
#                                 workers_opt[ww_id].step()
#                                 model.get() # <-- NEW: get the model back
#                                 loss = loss.get() # <-- NEW: get the loss back
#                                 if batch_idx % args.log_interval == 0:
#                                     if args.local_log:
#                                         file = open(args.log_dir + str(ww_id) + "_train", "a")
#                                         TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch, batch_idx, ww_id, loss)
#                                         file.write(TO_FILE)
#                                         file.close()
#                                 #     logging.info('Train Round: {}, Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                                 #         round_no, epoch, ww_id, batch_idx, 
#                                 #         batch_idx * fed_dataloader.batch_size, 
#                                 #         len(fed_dataloader) * fed_dataloader.batch_size,
#                                 #         100. * batch_idx / len(fed_dataloader), loss.item()))
#                                 t4.set_postfix(ordered_dict={'Worker':ww_id, 'BatchID':batch_idx, 'Loss':loss.item()})
#                                 t4.update()
#                         t3.update()
#             t2.update()
        

def train_workers_with_attack(federated_train_loader, models, workers_id, attackers_idx, round_no, args):
    workers_opt = dict()
    attackers_here = [ii for ii in workers_id if ii in attackers_idx]
    for ww_id in workers_id:
        workers_opt[ww_id] = torch.optim.SGD(
            params=models[ww_id].parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    with tqdm(
        total=args.epochs, leave=False, colour="yellow", ncols=80, desc="Epoch\t", bar_format=TQDM_R_BAR) as t2:
        for epoch in range(args.epochs):
            t2.set_postfix(Rounds=round_no, Epochs=epoch)
            with tqdm(total=len(workers_id), ncols=80, desc="Workers\t", leave=False, bar_format=TQDM_R_BAR) as t3:
                t3.set_postfix(ordered_dict={ 
                                        'ATK':"{}/{}".format(len(attackers_here), len(workers_id))})
                for ww_id, fed_dataloader in federated_train_loader.items():
                    if ww_id in workers_id:
                        with tqdm(total=len(fed_dataloader), ncols=80, colour='red', desc="Batch\t", leave=False, bar_format=TQDM_R_BAR) as t4:
                            for batch_idx, (data, target) in enumerate(fed_dataloader): 
                                ww = data.location
                                model = models[ww.id]
                                data, target = data.to("cpu"), target.to("cpu")
                                if ww.id in attackers_idx:
                                    if args.attack_type == 1:
                                        models[ww.id] = FLNet().to(args.device)
                                    elif args.attack_type == 2:
                                        ss = utils.negative_parameters(models[ww.id].state_dict())
                                        models[ww.id].load_state_dict(ss)
                                    t4.set_postfix(ordered_dict={
                                        'Worker':ww.id, 
                                        'ATK':"[T]" if ww.id in attackers_idx else "[F]" , 
                                        'BatchID':batch_idx, 'Loss':'-'})
                                    #TODO: Be careful about the break
                                    break    
                                    # atk = torchattacks.FGSM(model, eps=0.25)
                                    # data, target = data.get(), target.get()
                                    # data = atk(data, target)
                                    # data, target = data.send(ww.id), target.send(ww.id)
                                else:
                                    model.train()
                                    model.send(ww.id)
                                    workers_opt[ww.id].zero_grad()
                                    output = model(data)
                                    loss = F.nll_loss(output, target)
                                    loss.backward()
                                    workers_opt[ww.id].step()
                                    model.get() # <-- NEW: get the model back
                                    loss = loss.get() # <-- NEW: get the loss back
                                    if batch_idx % args.log_interval == 0:
                                    #     # if args.neptune_log:
                                    #     #     neptune.log_metric("train_loss_" + str(ww.id), loss)
                                        if args.local_log:
                                            file = open(args.log_dir + str(ww.id) + "_train", "a")
                                            TO_FILE = '{} {} {} {} {}\n'.format(round_no, epoch, batch_idx, ww.id, loss)
                                            file.write(TO_FILE)
                                            file.close()
                                    #     logging.info('Train Round: {}, Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    #         round_no, epoch, ww.id, batch_idx, 
                                    #         batch_idx * fed_dataloader.batch_size, 
                                    #         len(fed_dataloader) * fed_dataloader.batch_size,
                                    #         100. * batch_idx / len(fed_dataloader), loss.item()))
                                    t4.set_postfix(ordered_dict={
                                        'Worker':ww.id, 
                                        'ATK':"[T]" if ww.id in attackers_idx else "[F]" , 
                                        'BatchID':batch_idx, 'Loss':loss.item()})
                                t4.update()
                        t3.update()
            t2.update()
    # return models
    

if __name__ == '__main__':
    # Initialization
    configs = utils.load_config(CONFIG_PATH)
    if arguments['--rounds']:
        configs['runtime']['rounds'] = int(arguments['--rounds'])
    if arguments['--epochs']:
        configs['runtime']['epochs'] = int(arguments['--epochs'])
    if arguments['--selected-workers']:
        configs['mnist']['selected_users_num'] = int(arguments['--selected-workers'])
    if arguments['--attackers-num']:
        configs['attack']['attackers_num'] = int(arguments['--attackers-num'])

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
        configs['mnist']['total_users_num'],
        configs['mnist']['selected_users_num'],
        configs['server']['data_fraction'],
        "avg" if arguments['--avg'] else "opt",
        int(arguments['--attack-type']) if arguments['--attack-type'] else configs['attack']['attack_type'],
        configs['attack']['attackers_num'],
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

    logging.info(
        "Configs:\n\
        Epoch:\t{}\n\
        Rounds:\t{}\n\
        Total Number of Users:\t{}\n\
        Selected Users:\t{}\n\
        Mode:\t{}\n\
        Attack:\t{}\n\
        Attackers:\t{}".format(
            args.epochs, args.rounds, args.total_users_num, args.selected_users_num, 
            args.mode, args.attack_type, args.attackers_num, args.attackers_num
        ))

    # Neptune logging initialization
    if args.neptune_log:
        neptune.init(project_qualified_name=configs['log']['neptune_init'])
        neptune.create_experiment(name=configs['log']['neptune_exp'])
    
    logging.info("Total number of users: {}".format(args.total_users_num))
    workers_idx = ["worker_" + str(i) for i in range(args.total_users_num)]
    workers = create_workers(hook, workers_idx)
    server = create_workers(hook, ['server'])
    server = server['server']
    if args.local_log:
        utils.write_to_file(args.log_dir, "all_users", workers_idx)
    
    train_dataset = utils.load_mnist_dataset(
        train=True, transform=transforms.Compose([
                                transforms.ToTensor(),]))
    
    # Now sort the dataset and distribute among users
    sorted_train_dataset = utils.sort_mnist_dataset(train_dataset)
    splitted_train_dataset = utils.split_dataset(
        sorted_train_dataset, int(len(sorted_train_dataset) / args.shards_num))
    mapped_train_datasets = utils.map_shards_to_worker(splitted_train_dataset, workers_idx, args.shards_per_worker_num)

    # server_pub_dataset = dict()
    server_pub_dataset = utils.fraction_of_datasets(mapped_train_datasets, args.server_data_fraction)
    federated_server_loader = dict()
    federated_server_loader['server'] = sy.FederatedDataLoader(
            server_pub_dataset.federate([server]), batch_size=args.batch_size, shuffle=True, drop_last=False)

    attackers_idx = utils.get_workers_idx(workers_idx, args.attackers_num, [])
    if args.local_log:
        utils.write_to_file(args.log_dir, "attackers", attackers_idx)

    federated_train_loader = dict()
    for ww_id, fed_dataset in mapped_train_datasets.items():
        federated_train_loader[ww_id] = sy.FederatedDataLoader(
            fed_dataset.federate([workers[ww_id]]), batch_size=args.batch_size, shuffle=False, drop_last=False)

    test_dataset = utils.load_mnist_dataset(
        train=False, transform=transforms.Compose([
                                transforms.ToTensor(),]))
    test_loader = utils.get_dataloader(
        test_dataset, args.test_batch_size, shuffle=True, drop_last=False)

    server_model = FLNet().to(args.device)
    test_loss, test_acc = 0.0, 0.0
    
    with tqdm(
        total=args.rounds, leave=True, colour="green", ncols=80, desc="Round\t", bar_format=TQDM_R_BAR) as t1:
        for round_no in range(args.rounds):
            
            workers_to_be_used = random.sample(workers_idx, args.selected_users_num)
            workers_model = dict()
            for ww_id in workers_to_be_used:
                workers_model[ww_id] = deepcopy(server_model)

            # logging.info("Workers for this round: {}".format(workers_to_be_used))
            if args.local_log:
                utils.write_to_file(args.log_dir, "selected_workers", "R{}: {}".format(
                    round_no, workers_to_be_used))
            train_workers_with_attack(federated_train_loader, workers_model, workers_to_be_used, attackers_idx, round_no, args)
            # Find the best weights and update the server model


            weights = dict()
            if args.mode == "avg":
                # Each worker takes two shards of 300 random.samples. Total of 600 random.samples
                # per worker. Total number of random.samples is 60000.
                # weights = [600.0 / 60000] * args.selected_users_num
                for ww_id in workers_to_be_used:
                    weights[ww_id] = 1.0 / args.selected_users_num
                    # weights = [1.0 / args.selected_users_num] * args.selected_users_num
            elif args.mode == "opt":
                # models should be returned from the workers before calling the following functions:
                # Train server
                server_model_dict = dict()
                server_model_dict['server'] = server_model
                train_workers_with_attack(federated_server_loader, server_model_dict, ['server'], [], round_no, args)

                weights = utils.find_best_weights_opt(server_model, workers_model)
                if args.local_log:
                    utils.write_to_file(args.log_dir, "opt_weights", weights)
                    
            weighted_avg_state = utils.wieghted_avg_model(weights, workers_model)
            # logging.info("Update server model in this round...")
            server_model.load_state_dict(weighted_avg_state)

            # Apply the server model to the test dataset
            # logging.info("Starting model evaluation on the test dataset...")
            test_loss, test_acc = test(server_model, test_loader, round_no, args)
            

            if args.local_log:
                save_model(
                    server_model, 
                    "R{}_{}".format(round_no, "server_model")
                )
            print()
            logging.info('Test Average loss: {:.4f}, Accuracy: {:.0f}%'.format(test_loss, test_acc))
            print()
            t1.set_postfix(test_acc=test_acc, test_loss=test_loss)
            t1.update()
    