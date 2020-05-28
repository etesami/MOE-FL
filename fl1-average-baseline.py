import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy

workers_num = 10
epochs_num = 5

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

federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    .federate(tuple([ww for id, ww in workers.items()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# aggregated_data = torch.Tensor()
# aggregated_label = torch.Tensor()
# for id, worker in workers.items():
#     ids_in_worker = list(worker._objects.keys()) # data & label
#     worker_data = worker.get_obj(ids_in_worker[0])
#     worker_label = worker.get_obj(ids_in_worker[1])
#     print("{}: get {} out of {}".format(worker.id, int(0.2 * len(worker_data)), len(worker_data)))
#     if len(aggregated_data) == 0:
#         aggregated_data = worker_data[:int(0.2 * len(worker_data))]
#         aggregated_label = worker_label[:int(0.2 * len(worker_label))]
#     else:
#         aggregated_data = torch.cat((aggregated_data, worker_data[:int(0.2 * len(worker_data))]), 0)
#         aggregated_label = torch.cat((aggregated_label, worker_label[:int(0.2 * len(worker_label))]), 0)
#
# print("Aggregated data contains {} records".format(len(aggregated_label)))
# # add collected data to the last worker
# aggregated_train = [aggregated_data.send(server), aggregated_label.send(server)]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, workers_model, workers_optimizer, federated_train_loader, epoch):
    device="cpu"
    file = open("/home/savi/output3-train.txt", "a")
    for ww_id, ww in workers.items():
        if workers_model[ww_id].location is None \
                or workers_model[ww_id].location.id != ww_id:
            workers_model[ww_id].send(ww)
            print("Sending worker {} to the {}".format(ww_id, ww.id))
        workers_optimizer[ww_id] = optim.SGD(params=workers_model[ww_id].parameters(), lr=args.lr)

    for batch_idx, (data, target) in enumerate(federated_train_loader):
        worker_id = data.location.id
        worker_model = workers_model[worker_id]
        worker_opt = workers_optimizer[worker_id]
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


def train_server(args, server_model, server_opt, aggregated_train):
    device = "cpu"
    data = aggregated_train[0]
    target = aggregated_train[1]
    # print("data loc: {}, len: {}\ntarget: {}".format(data.location.id, len(data), target))
    iteration = 1
    for iter in range(0,iteration):
        server_model.train()
        data, target = data.to(device), target.to(device)
        server_opt.zero_grad()
        output = server_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        server_opt.step()
        loss = loss.get()
        # TO_FILE = '{} {} {} {}\n'.format(epoch, batch_idx, data.location.id, loss)
        # file.write(TO_FILE)
        print('Train Server Iter: {} \tLoss: {:.6f}'.format(iter, loss.item()))

def test(args, model, test_loader, epoch):
    device="cpu"
    file = open("/home/savi/output3-test.txt", "a")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    TO_FILE = '{} {} {}\n'.format(epoch, test_loss, 100. * correct / len(test_loader.dataset))
    file.write(TO_FILE)
    file.close()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def update_models(server, base_model, workers_model):
    if base_model.location is None \
            or base_model.location.id != "server":
        base_model.send(server)
        print("Sending base_model to server")

    tmp_model = Net().to(device).send(server)

    with torch.no_grad():
        tmp_model.conv1.weight.data.fill_(0)
        tmp_model.conv1.bias.data.fill_(0)
        tmp_model.conv2.weight.data.fill_(0)
        tmp_model.conv2.bias.data.fill_(0)
        tmp_model.fc1.weight.data.fill_(0)
        tmp_model.fc1.bias.data.fill_(0)
        tmp_model.fc2.weight.data.fill_(0)
        tmp_model.fc2.bias.data.fill_(0)

        for worker_id, worker_model in workers_model.items():
            if worker_model.location is None:
                worker_model.send(server)
            elif worker_model.location.id != "server":
                worker_model.move(server)

            tmp_model.conv1.weight.data = (
                tmp_model.conv1.weight.data + worker_model.conv1.weight.data)
            tmp_model.conv1.bias.data = (
                tmp_model.conv1.bias.data + worker_model.conv1.bias.data)
            tmp_model.conv2.weight.data = (
                tmp_model.conv2.weight.data + worker_model.conv2.weight.data)
            tmp_model.conv2.bias.data = (
                tmp_model.conv2.bias.data + worker_model.conv2.bias.data)
            tmp_model.fc1.weight.data = (
                tmp_model.fc1.weight.data + worker_model.fc1.weight.data)
            tmp_model.fc1.bias.data = (
                tmp_model.fc1.bias.data + worker_model.fc1.bias.data)
            tmp_model.fc2.weight.data = (
                tmp_model.fc2.weight.data + worker_model.fc2.weight.data)
            tmp_model.fc2.bias.data = (
                tmp_model.fc2.bias.data + worker_model.fc2.bias.data)

        base_model.conv1.weight.data = (tmp_model.conv1.weight.data / len(workers_model))
        base_model.conv1.bias.data = (tmp_model.conv1.bias.data / len(workers_model))
        base_model.conv2.weight.data = (tmp_model.conv2.weight.data / len(workers_model))
        base_model.conv2.bias.data = (tmp_model.conv2.bias.data / len(workers_model))
        base_model.fc1.weight.data = (tmp_model.fc1.weight.data / len(workers_model))
        base_model.fc1.bias.data = (tmp_model.fc1.bias.data / len(workers_model))
        base_model.fc2.weight.data = (tmp_model.fc2.weight.data / len(workers_model))
        base_model.fc2.bias.data = (tmp_model.fc2.bias.data / len(workers_model))

        for models_id in workers_model:
            workers_model[models_id].get()
        base_model.get()

        for worker_id in workers_model.keys():
            workers_model[worker_id] = base_model.copy()

base_model = Net().to(device)
server_optimizer = optim.SGD(base_model.parameters(), lr=args.lr)
workers_model = {}
workers_optimizer = {}

for worker_id, worker in workers.items():
    print("Create a copy base_model and an optimizer for the {} ...".format(worker_id))
    workers_model[worker_id] = base_model.copy()

# One-time training at server
# In avg mode, the server does not perform training itself.
# train_server(args, server_model, server_optimizer, aggregated_train)

for epoch in range(1, args.epochs + 1):
    train(args, workers_model, workers_optimizer, federated_train_loader, epoch)
    update_models(server, base_model, workers_model)
    test(args, base_model, test_loader, epoch)

