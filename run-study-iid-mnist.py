"""
Usage: 
    run-study-iid-mnist.py 
        (--avg | --opt) [--epochs=NUM] [--rounds=NUM]
        [--attack-type=ID] [--attackers-num=num] [--selected-workers=NUM]
        [--log] [--nep-log] [--output-prefix=NAME] 
"""
from docopt import docopt
import os
import torch
import random
import neptune
import _pickle as pickle
import syft as sy
import numpy as np
from tqdm import tqdm
import cvxpy as cp
import time
from copy import deepcopy
import torchvision
import coloredlogs, logging
from torchvision import transforms
from collections import defaultdict
from torch.nn import functional as F
from federated_learning.FLNet import FLNet
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.Arguments import Arguments
from federated_learning.helper import utils

CONFIG_PATH = 'configs/defaults.yml'
TQDM_R_BAR = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}] '
ROUNDS_BREAKDOWN = 25

arguments = docopt(__doc__)
############ TEMPORARILY ################
# arguments = dict()
############ TEMPORARILY ################

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
        TO_FILE = '{} "{{/*Accuracy:}}\\n{}%" {}'.format(test_loss, test_acc, test_acc)
        utils.write_to_file(args.log_dir, "accuracy", TO_FILE, round_no=round_no)
        
    logging.debug('Test Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    return test_loss, test_acc
        

def train_workers_with_attack(federated_train_loader, models, workers_idx, attackers_idx, round_no, args):
    attackers_here = [ii for ii in workers_idx if ii in attackers_idx]
    workers_opt = dict()
    workers_loss = data = defaultdict(lambda : [])
    for ii in workers_idx:
        workers_opt[ii] = torch.optim.SGD(
                            params=models[ii].parameters(), 
                            lr=args.lr, weight_decay=args.weight_decay)
    with tqdm(
        total=args.epochs, leave=False, colour="yellow", ncols=80, desc="Epoch\t", bar_format=TQDM_R_BAR) as t2:
        for epoch in range(args.epochs):
            t2.set_postfix(Rounds=round_no, Epochs=epoch)
            with tqdm(total=len(workers_idx), ncols=80, desc="Workers\t", leave=False, bar_format=TQDM_R_BAR) as t3:
                t3.set_postfix(
                    ordered_dict={'ATK':"{}/{}".format(len(attackers_here), len(workers_idx))})
                for ww_id, fed_dataloader in federated_train_loader.items():
                    if ww_id in workers_idx:
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
                                else:
                                    model.train()
                                    model.send(ww.id)
                                    opt = workers_opt[ww.id]
                                    opt.zero_grad()
                                    output = model(data)
                                    loss = F.nll_loss(output, target)
                                    loss.backward()
                                    opt.step()
                                    model.get() # <-- NEW: get the model back
                                    loss = loss.get() # <-- NEW: get the loss back
                                    workers_loss[ww.id].append(loss.item())
                                    t4.set_postfix(ordered_dict={
                                        'Worker':ww.id, 
                                        'ATK':"[T]" if ww.id in attackers_idx else "[F]" , 
                                        'BatchID':batch_idx, 'Loss':loss.item()})
                                t4.update()
                        t3.update()
            t2.update()
    
    # Mean per worker
    for ii in workers_loss:
        workers_loss[ii] = sum(workers_loss[ii]) / len(workers_loss[ii])
    return workers_loss
    

def main():
    logging.info("Total number of users: {}".format(args.total_users_num))
    workers_idx = ["worker_" + str(i) for i in range(args.total_users_num)]
    workers = create_workers(hook, workers_idx)
    server = create_workers(hook, ['server'])
    server = server['server']
    if args.local_log:
        utils.check_write_to_file(args.log_dir, "all_users", workers_idx)
    
    mapped_datasets = dict()
    if utils.find_file(args.log_dir, "mapped_datasets"):
        logging.info("mapped_datasets was found. Loading from file...")
        mapped_datasets = utils.load_object(args.log_dir, "mapped_datasets")
    else:
        # Now sort the dataset and distribute among users
        mapped_ds_itr = utils.map_shards_to_worker(
            utils.split_randomly_dataset(
                utils.load_mnist_dataset(
                    train=True, 
                    transform=transforms.Compose([transforms.ToTensor(),])),
                args.shards_num),
            workers_idx, 
            args.shards_per_worker_num)

        for mapped_ds in mapped_ds_itr:
            mapped_datasets.update(mapped_ds)

        if args.local_log:
            utils.save_object(args.log_dir, "mapped_datasets", mapped_datasets)

    server_pub_dataset = None
    if utils.find_file(args.log_dir, "server_pub_dataset"):
        logging.info("server_pub_dataset was found. Loading from file...")
        server_pub_dataset = utils.load_object(args.log_dir, "server_pub_dataset")
    else:
        server_pub_dataset = utils.fraction_of_datasets(mapped_datasets, args.server_data_fraction)
        if args.local_log:
            utils.save_object(args.log_dir, "server_pub_dataset", server_pub_dataset)

    federated_server_loader = dict()
    federated_server_loader['server'] = sy.FederatedDataLoader(
            server_pub_dataset.federate([server]), batch_size=args.batch_size, shuffle=True, drop_last=False)

    
    attackers_idx = None
    if utils.find_file(args.log_dir, "attackers"):
        logging.info("attackers list was found. Loading from file...")
        attackers_idx = utils.load_object(args.log_dir, "attackers")
    else:
        attackers_idx = utils.get_workers_idx(workers_idx, args.attackers_num, [])
        if args.local_log:
            utils.save_object(args.log_dir, "attackers", attackers_idx)

    federated_train_loader = dict()
    logging.info("Creating federated dataloaders for workers...")
    for ww_id, fed_dataset in mapped_datasets.items():
        federated_train_loader[ww_id] = sy.FederatedDataLoader(
            fed_dataset.federate([workers[ww_id]]), batch_size=args.batch_size, shuffle=False, drop_last=False)

    test_loader = utils.get_dataloader(
        utils.load_mnist_dataset(
            train=False, 
            transform=transforms.Compose([transforms.ToTensor(),])), 
        args.test_batch_size, shuffle=True, drop_last=False)

    previous_round = 0
    if utils.find_file(args.log_dir, "accuracy"):
        previous_round = int(utils.get_last_round_num(args.log_dir, "accuracy"))
        logging.info("Previous complete execution was found. Last run is: {}".format(previous_round))
    
    round_start = previous_round if previous_round == 0 else previous_round + 1
    round_end = round_start + ROUNDS_BREAKDOWN \
        if round_start + ROUNDS_BREAKDOWN < args.rounds else args.rounds

    server_model, server_model_name = FLNet().to(args.device), "R{}_server_model".format(previous_round)
    server_model_path = args.log_dir + "models"
    if utils.find_file(server_model_path, server_model_name):
        logging.info("server_model was found. Loading {} from the file...".format(server_model_name))
        server_model.load_state_dict(torch.load(server_model_path + "/" + server_model_name))
    server_opt = dict()
    server_opt['server'] = torch.optim.SGD(
                        params=server_model.parameters(), 
                        lr=args.lr, weight_decay=args.weight_decay)
    test_loss, test_acc = 0.0, 0.0

    with tqdm(
        total=min(ROUNDS_BREAKDOWN, args.rounds - round_start), leave=True, colour="green", ncols=80, desc="Round\t", bar_format=TQDM_R_BAR) as t1:
        for round_no in range(round_start, round_end):
            workers_to_be_used = random.sample(workers_idx, args.selected_users_num)
            workers_model = dict()
            for ww_id in workers_to_be_used:
                workers_model[ww_id] = deepcopy(server_model)

            # logging.info("Workers for this round: {}".format(workers_to_be_used))
            if args.local_log:
                utils.write_to_file(args.log_dir, "selected_workers", workers_to_be_used, round_no=round_no)
            train_loss = train_workers_with_attack(federated_train_loader, workers_model, workers_to_be_used, attackers_idx, round_no, args)

            # Find the best weights and update the server model
            weights = dict()
            if args.mode == "avg":
                # Each worker takes two shards of 300 random.samples. Total of 600 random.samples
                # per worker. Total number of random.samples is 60000.
                # weights = [600.0 / 60000] * args.selected_users_num
                for ww_id in workers_to_be_used:
                    weights[ww_id] = 1.0 / args.selected_users_num
                train_loss = sum(train_loss.values()) / len(train_loss)
            elif args.mode == "opt":
                # models should be returned from the workers before calling the following functions:
                # Train server
                train_workers_with_attack(federated_server_loader, {'server': server_model}, ['server'], [], round_no, args)
                pass

                weights = utils.find_best_weights_opt(server_model, workers_model)

                loss = 0
                for ww in train_loss:
                    loss += weights[ww] * train_loss[ww]
                train_loss = loss
                if args.local_log:
                    utils.write_to_file(args.log_dir, "opt_weights", weights, round_no=round_no)
                    
            # logging.info("Update server model in this round...")
            server_model.load_state_dict(
                utils.wieghted_avg_model(weights, workers_model, workers_to_be_used)
            )

            # Apply the server model to the test dataset
            # logging.info("Starting model evaluation on the test dataset...")
            test_loss, test_acc = test(server_model, test_loader, round_no, args)

            if args.local_log:
                utils.write_to_file(args.log_dir, "train_loss", train_loss, round_no=round_no)
                utils.save_model(
                    server_model.state_dict(),
                    args.log_dir, 
                    "R{}_{}".format(round_no, "server_model")
                )
            if args.neptune_log:
                neptune.log_metric("train_loss", train_loss)

            print()
            logging.info('Test Average loss: {:.4f}, Accuracy: {:.0f}%'.format(test_loss, test_acc))
            print()

            t1.set_postfix(test_acc=test_acc, test_loss=test_loss)
            t1.update()
    return round_end


if __name__ == '__main__':
    # Initialization
    configs = utils.load_config(CONFIG_PATH)

    # Logging initialization
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=configs['log']['level'], fmt=configs['log']['format'])

    if arguments['--rounds']:
        configs['runtime']['rounds'] = int(arguments['--rounds'])
    if arguments['--epochs']:
        configs['runtime']['epochs'] = int(arguments['--epochs'])
    if arguments['--selected-workers']:
        configs['mnist']['selected_users_num'] = int(arguments['--selected-workers'])
    if arguments['--attackers-num']:
        configs['attack']['attackers_num'] = int(arguments['--attackers-num'])

    log_dir_path = utils.check_create_output_dir(
        configs['log']['root_output_dir'], arguments['--output-prefix']) \
            if arguments['--log'] else ""

    args = Arguments(
        batch_size=configs['runtime']['batch_size'],
        test_batch_size=configs['runtime']['test_batch_size'],
        rounds=configs['runtime']['rounds'],
        epochs=configs['runtime']['epochs'],
        lr=configs['runtime']['lr'],
        momentum=configs['runtime']['momentum'],
        weight_decay=configs['runtime']['weight_decay'],
        shards_num=configs['mnist']['shards_num'],
        shards_per_worker_num=configs['mnist']['shards_per_worker_num'],
        total_users_num=configs['mnist']['total_users_num'],
        selected_users_num=configs['mnist']['selected_users_num'],
        server_data_fraction=configs['server']['data_fraction'],
        mode="avg" if arguments['--avg'] else "opt",
        attack_type=int(arguments['--attack-type']) if arguments['--attack-type'] else configs['attack']['attack_type'],
        attackers_num=configs['attack']['attackers_num'],
        use_cuda=configs['runtime']['use_cuda'],
        device=torch.device("cuda" if configs['runtime']['use_cuda'] else "cpu"),
        seed=configs['runtime']['random_seed'],
        log_interval=configs['log']['interval'],
        log_level=configs['log']['level'],
        log_format=configs['log']['format'],
        log_dir=log_dir_path,
        neptune_log=True if arguments['--nep-log'] else False,
        local_log=True if arguments['--log'] else False
    )

    # Since we have to run the app multiple times,
    # we have to generate multiple seeds to prevent
    # giving the same numbers to the app.
    seed = time.time()
    random.seed(seed)
    if args.local_log:
        logging.info("Saving the seed for this round: {}".format(seed))
        utils.write_to_file(args.log_dir, "seeds", seed)
    torch.manual_seed(seed)
    
    # syft initialization
    hook = sy.TorchHook(torch)

    output_dir = None
    if args.local_log:
        logging.info("Saving the configuration file...")
        utils.check_save_configs(args.log_dir, configs)

    logging.info(
        "Configs:\n\
        Epoch:\t{}\n\
        Rounds:\t{}\n\
        Total Number of Users:\t{}\n\
        Selected Users:\t{}\n\
        Mode:\t{}\n\
        Attack:\t{}\n\
        Attackers:\t{}\n\
        Output folder:\t{}".format(
            args.epochs, args.rounds, args.total_users_num, args.selected_users_num, 
            args.mode, args.attack_type, args.attackers_num, args.log_dir
        ))

    # Neptune logging initialization
    if args.neptune_log:
        neptune.init(project_qualified_name=configs['log']['neptune_init'], api_token=utils.get_neptune_token())
        neptune.create_experiment(name=configs['log']['neptune_exp'], upload_stdout=False, upload_stderr=False)
        neptune.append_tag(args.log_dir.split("/")[1])
    last_round = main()
    if args.neptune_log:
        neptune.stop()

    if last_round < args.rounds:
        exit(1)
    else:
        exit(0)

    