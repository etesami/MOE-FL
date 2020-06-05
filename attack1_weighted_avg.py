import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
import cvxpy as cp
import numpy as np
from mnist import MNIST
from FLCustomDataset import FLCustomDataset
from FLNet import FLNet

workers_num = 10
epochs_num = 5
# prefix_output = "/home/savi/output_attack_1_"
prefix_output = "/home/savi/out_6_10_OPT_"

hook = sy.TorchHook(torch)

class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = epochs_num
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False


args = Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
device = torch.device("cuda" if use_cuda else "cpu")

workers = {}
for num in range(0, workers_num):
    worker_id = "worker" + str(num)
    workers[worker_id] = sy.VirtualWorker(hook, id=worker_id)
server = sy.VirtualWorker(hook, id="server")

mndata = MNIST('/home/savi/data/MNIST/raw')

train_images, train_labels = mndata.load_training()
train_images = np.asarray(train_images, dtype=np.uint8).reshape(-1, 28, 28)
train_labels = np.asarray(train_labels)

# '''
# Attack 1
# Permute all 0 - 6000 labels (for worker0)
# 0 1 2 3 4 5 (6) 7 (8) 9
# 1 2 3 4 5 6  7  8  9 10
# Worker (6), (7) (8), (9) (10) Bad
# '''
for i in range(5, 10): # Zero-based A[0]- A[1]
    # if i != 8 or i != 6:
    train_labels[i * int(len(train_labels) / len(workers)): (i + 1) * int(len(train_labels) / len(workers))] = \
        np.random.permutation(
            train_labels[i * int(len(train_labels) / len(workers)): (i + 1) * int(len(train_labels) / len(workers))])

custom_dataset = FLCustomDataset(train_images, train_labels,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))]))

custom_dataset_loader = sy.FederatedDataLoader(
    custom_dataset.federate(tuple([ww for id, ww in workers.items()])),
    batch_size=args.batch_size, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size)

aggregated_data = None
aggregated_label = None

for id, worker in workers.items():
    print("Send out {} records from {} out of {}".format(worker.id,
                                                         int(0.2 * len(train_images) / len(workers)),
                                                         len(train_images) / len(workers)))
    if aggregated_data is None:
        aggregated_data = train_images[:int(0.2 * len(train_images) / len(workers))]
        aggregated_label = train_labels[:int(0.2 * len(train_images) / len(workers))]
    else:
        aggregated_data = np.concatenate((aggregated_data,
                                          train_images[:int(0.2 * len(train_images) / len(workers))]), 0)
        aggregated_label = np.concatenate((aggregated_label,
                                           train_labels[:int(0.2 * len(train_images) / len(workers))]), 0)

print("\nAggregated data contains {} records\n".format(len(aggregated_label)))

dataset_server = FLCustomDataset(aggregated_data, aggregated_label,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))]))

federated_server_loader = sy.FederatedDataLoader(
    dataset_server.federate(tuple([server])),
    batch_size=args.batch_size, shuffle=False, **kwargs)


# n_batches = int(len(aggregated_label) / args.batch_size)
# aggregated_train = [None] * n_batches
# for b in range(0, n_batches):
#     if b % 10 == 0:
#         print("Creating aggregated data, sending to server {}%".format(round(b * 100 / n_batches, 2)))
#     aggregated_train[b] = [torch.from_numpy(
#         aggregated_data[b * args.batch_size:(b + 1) * args.batch_size].reshape(-1, 1, 28, 28)).send(server),
#                            torch.from_numpy(
#                                aggregated_label[b * args.batch_size:(b + 1) * args.batch_size]).send(server)]
# print("Total Batches: {}".format(n_batches))
# print()

def send_model(model, location, location_id):
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


def getback_model(model):
    if isinstance(model, dict):
        for ww_id, ww in model.items():
            if ww.location is not None:
                ww.get()
    elif model.location is not None:
        model.get()


def train(args, workers_model, federated_train_loader, epoch):
    device = "cpu"
    workers_opt = {}
    file = open(prefix_output + "train", "a")
    for ww_id, ww in workers.items():
        if workers_model[ww_id].location is None \
                or workers_model[ww_id].location.id != ww_id:
            workers_model[ww_id].send(ww)
        workers_opt[ww_id] = optim.SGD(params=workers_model[ww_id].parameters(), lr=args.lr)

    for batch_idx, (data, target) in enumerate(federated_train_loader):
        worker_id = data.location.id
        worker_model = workers_model[worker_id]
        worker_opt = workers_opt[worker_id]
        worker_model.train()
        data, target = data.to(device), target.to(device)
        worker_opt.zero_grad()
        output = worker_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        worker_opt.step()

        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            TO_FILE = '{} {} {} {}\n'.format(epoch, batch_idx, data.location.id, loss)
            file.write(TO_FILE)
            print('Train Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, worker_id, batch_idx, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                                             100. * batch_idx / len(federated_train_loader), loss.item()))
    file.close()
    print()


def train_server(args, server, server_model, federated_train_server_loader, epoch):
    file = open(prefix_output + "train_server", "a")
    send_model(server_model, server, "server")
    device = "cpu"
    server_opt = optim.SGD(server_model.parameters(), lr=args.lr)
    for batch_idx, (data, target) in enumerate(federated_train_server_loader):
        server_model.train()
        data, target = data.to(device), target.to(device)
        server_opt.zero_grad()
        output = server_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        server_opt.step()

        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            TO_FILE = '{} {} {} {}\n'.format(epoch, batch_idx, data.location.id, loss)
            file.write(TO_FILE)
            print('Train Epoch: {} [server] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * args.batch_size, len(federated_train_server_loader) * args.batch_size,
                                  100. * batch_idx / len(federated_train_server_loader), loss.item()))
    file.close()
    print()


def test(args, model, test_loader, epoch):
    device = "cpu"
    file = open(prefix_output + "test", "a")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    TO_FILE = '{} {} {}\n'.format(epoch, test_loss, 100. * correct / len(test_loader.dataset))
    file.write(TO_FILE)
    file.close()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#
def find_best_weights(server, base_model, workers_model, epoch):
    file = open(prefix_output + "weights", "a")
    getback_model(base_model)
    with torch.no_grad():
        server_layers = [None] * 8
        server_layers[0] = base_model.conv1.weight.data.numpy().copy().reshape(-1, 1).ravel()
        server_layers[1] = base_model.conv1.bias.data.numpy().copy().reshape(-1, 1).ravel()
        server_layers[2] = base_model.conv2.weight.data.numpy().copy().reshape(-1, 1).ravel()
        server_layers[3] = base_model.conv2.bias.data.numpy().copy().reshape(-1, 1).ravel()
        server_layers[4] = base_model.fc1.weight.data.numpy().copy().reshape(-1, 1).ravel()
        server_layers[5] = base_model.fc1.bias.data.numpy().copy().reshape(-1, 1).ravel()
        server_layers[6] = base_model.fc2.weight.data.numpy().copy().reshape(-1, 1).ravel()
        server_layers[7] = base_model.fc2.bias.data.numpy().copy().reshape(-1, 1).ravel()

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
        getback_model(workers_model)
        for worker_id, worker_model in workers_model.items():
            workers_params[worker_id] = [[] for i in range(8)]
            workers_params[worker_id][0] = worker_model.conv1.weight.data.numpy().copy().reshape(-1, 1)
            workers_params[worker_id][1] = worker_model.conv1.bias.data.numpy().copy().reshape(-1, 1)
            workers_params[worker_id][2] = worker_model.conv2.weight.data.numpy().copy().reshape(-1, 1)
            workers_params[worker_id][3] = worker_model.conv2.bias.data.numpy().copy().reshape(-1, 1)
            workers_params[worker_id][4] = worker_model.fc1.weight.data.numpy().copy().reshape(-1, 1)
            workers_params[worker_id][5] = worker_model.fc1.bias.data.numpy().copy().reshape(-1, 1)
            workers_params[worker_id][6] = worker_model.fc2.weight.data.numpy().copy().reshape(-1, 1)
            workers_params[worker_id][7] = worker_model.fc2.bias.data.numpy().copy().reshape(-1, 1)
        """
        --> conv1.weight
        workers_all_params[0] =
            [workers_param[worker0][0], workers_param[worker1][0], workers_param[worker2][0]]
        --> conv1.bias
        workers_all_params[1] =
            [workers_param[worker0][1], workers_param[worker1][1], workers_param[worker2][1]]
        """

        workers_all_params = []
        print()
        print("Start the optimization....")
        workers_all_params.append(np.array([]).reshape(workers_params["worker0"][0].shape[0], 0))
        workers_all_params.append(np.array([]).reshape(workers_params["worker0"][1].shape[0], 0))
        workers_all_params.append(np.array([]).reshape(workers_params["worker0"][2].shape[0], 0))
        workers_all_params.append(np.array([]).reshape(workers_params["worker0"][3].shape[0], 0))
        workers_all_params.append(np.array([]).reshape(workers_params["worker0"][4].shape[0], 0))
        workers_all_params.append(np.array([]).reshape(workers_params["worker0"][5].shape[0], 0))
        workers_all_params.append(np.array([]).reshape(workers_params["worker0"][6].shape[0], 0))
        workers_all_params.append(np.array([]).reshape(workers_params["worker0"][7].shape[0], 0))

        for worker_id, worker_model in workers_params.items():
            workers_all_params[0] = np.concatenate((workers_all_params[0], workers_params[worker_id][0]), 1)
            workers_all_params[1] = np.concatenate((workers_all_params[1], workers_params[worker_id][1]), 1)
            workers_all_params[2] = np.concatenate((workers_all_params[2], workers_params[worker_id][2]), 1)
            workers_all_params[3] = np.concatenate((workers_all_params[3], workers_params[worker_id][3]), 1)
            workers_all_params[4] = np.concatenate((workers_all_params[4], workers_params[worker_id][4]), 1)
            workers_all_params[5] = np.concatenate((workers_all_params[5], workers_params[worker_id][5]), 1)
            workers_all_params[6] = np.concatenate((workers_all_params[6], workers_params[worker_id][6]), 1)
            workers_all_params[7] = np.concatenate((workers_all_params[7], workers_params[worker_id][7]), 1)

        W = cp.Variable(len(workers_model))

        objective = cp.Minimize(cp.norm2(cp.matmul(workers_all_params[0], W) - server_layers[0]) +
                                cp.norm2(cp.matmul(workers_all_params[1], W) - server_layers[1]) +
                                cp.norm2(cp.matmul(workers_all_params[2], W) - server_layers[2]) +
                                cp.norm2(cp.matmul(workers_all_params[3], W) - server_layers[3]) +
                                cp.norm2(cp.matmul(workers_all_params[4], W) - server_layers[4]) +
                                cp.norm2(cp.matmul(workers_all_params[5], W) - server_layers[5]) +
                                cp.norm2(cp.matmul(workers_all_params[6], W) - server_layers[6]) +
                                cp.norm2(cp.matmul(workers_all_params[7], W) - server_layers[7]))

        constraints = [0 <= W, W <= 1, sum(W) == 1]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.MOSEK)
        print(W.value)
        print()
        TO_FILE = '{} {}\n'.format(epoch, np.array2string(W.value).replace('\n',''))
        file.write(TO_FILE)
        # file.close()
        return W.value


def update_models(W, base_model, server_model, workers_model):
    getback_model(workers_model)
    getback_model(server_model)
    getback_model(base_model)
    tmp_model = FLNet().to(device)

    with torch.no_grad():
        tmp_model.conv1.weight.data.fill_(0)
        tmp_model.conv1.bias.data.fill_(0)
        tmp_model.conv2.weight.data.fill_(0)
        tmp_model.conv2.bias.data.fill_(0)
        tmp_model.fc1.weight.data.fill_(0)
        tmp_model.fc1.bias.data.fill_(0)
        tmp_model.fc2.weight.data.fill_(0)
        tmp_model.fc2.bias.data.fill_(0)

        counter = 0
        for worker_id, worker_model in workers_model.items():
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
            counter = counter + 1

        # base_model.conv1.weight.data = tmp_model.conv1.weight.data
        # base_model.conv1.bias.data = tmp_model.conv1.bias.data
        # base_model.conv2.weight.data = tmp_model.conv2.weight.data
        # base_model.conv2.bias.data = tmp_model.conv2.bias.data
        # base_model.fc1.weight.data = tmp_model.fc1.weight.data
        # base_model.fc1.bias.data = tmp_model.fc1.bias.data
        # base_model.fc2.weight.data = tmp_model.fc2.weight.data
        # base_model.fc2.bias.data = tmp_model.fc2.bias.data

        server_model.conv1.weight.data = tmp_model.conv1.weight.data
        server_model.conv1.bias.data = tmp_model.conv1.bias.data
        server_model.conv2.weight.data = tmp_model.conv2.weight.data
        server_model.conv2.bias.data = tmp_model.conv2.bias.data
        server_model.fc1.weight.data = tmp_model.fc1.weight.data
        server_model.fc1.bias.data = tmp_model.fc1.bias.data
        server_model.fc2.weight.data = tmp_model.fc2.weight.data
        server_model.fc2.bias.data = tmp_model.fc2.bias.data

        for worker_id in workers_model.keys():
            workers_model[worker_id].conv1.weight.data = tmp_model.conv1.weight.data
            workers_model[worker_id].conv1.bias.data = tmp_model.conv1.bias.data
            workers_model[worker_id].conv2.weight.data = tmp_model.conv2.weight.data
            workers_model[worker_id].conv2.bias.data = tmp_model.conv2.bias.data
            workers_model[worker_id].fc1.weight.data = tmp_model.fc1.weight.data
            workers_model[worker_id].fc1.bias.data = tmp_model.fc1.bias.data
            workers_model[worker_id].fc2.weight.data = tmp_model.fc2.weight.data
            workers_model[worker_id].fc2.bias.data = tmp_model.fc2.bias.data


'''
Clients' models are being trained locally.
Server model is being updated only
'''
server_model = FLNet().to(device)
workers_model = {}

for worker_id, worker in workers.items():
    workers_model[worker_id] = server_model.copy()

for epoch in range(1, args.epochs + 1):
    train_server(args, server, server_model, federated_server_loader, epoch)
    print()
    train(args, workers_model, custom_dataset_loader, epoch)
    print()
    W = find_best_weights(server, server_model, workers_model, epoch)
    # W = [0.1] * 10

    # base model is meant nothing in this scenario
    update_models(W, server_model, server_model, workers_model)
    print()

    # Apply the server model to the test dataset
    test(args, server_model, test_loader, epoch)
