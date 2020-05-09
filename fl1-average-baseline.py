import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy  # <-- NEW: import the Pysyft library
import cvxpy


workers_num = 10
epochs_num = 5

hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning

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
secure_worker = sy.VirtualWorker(hook, id="secure_worker")

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

def train(args, base_model, workers_model, workers_optimizer, federated_train_loader, secure_worker, epoch):
    device="cpu"
    file = open("/home/savi/output1-train.txt", "a")
    for ww_id, ww in workers.items():
        workers_model[ww_id].send(ww)
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

    avg_model = Net().to(device).send(secure_worker)

    with torch.no_grad():
        avg_model.conv1.weight.data.fill_(0)
        avg_model.conv1.bias.data.fill_(0)
        avg_model.conv2.weight.data.fill_(0)
        avg_model.conv2.bias.data.fill_(0)
        avg_model.fc1.weight.data.fill_(0)
        avg_model.fc1.bias.data.fill_(0)
        avg_model.fc2.weight.data.fill_(0)
        avg_model.fc2.bias.data.fill_(0)

        for worker_id, worker_model in workers_model.items():
            worker_model.move(secure_worker)
            avg_model.conv1.weight.data = (
                avg_model.conv1.weight.data + worker_model.conv1.weight.data)
            avg_model.conv1.bias.data = (
                avg_model.conv1.bias.data + worker_model.conv1.bias.data)
            avg_model.conv2.weight.data = (
                avg_model.conv2.weight.data + worker_model.conv2.weight.data)
            avg_model.conv2.bias.data = (
                avg_model.conv2.bias.data + worker_model.conv2.bias.data)
            avg_model.fc1.weight.data = (
                avg_model.fc1.weight.data + worker_model.fc1.weight.data)
            avg_model.fc1.bias.data = (
                avg_model.fc1.bias.data + worker_model.fc1.bias.data)
            avg_model.fc2.weight.data = (
                avg_model.fc2.weight.data + worker_model.fc2.weight.data)
            avg_model.fc2.bias.data = (
                avg_model.fc2.bias.data + worker_model.fc2.bias.data)

        base_model.conv1.weight.set_((avg_model.conv1.weight.data / len(workers_model)).get())
        base_model.conv1.bias.set_((avg_model.conv1.bias.data / len(workers_model)).get())
        base_model.conv2.weight.set_((avg_model.conv2.weight.data / len(workers_model)).get())
        base_model.conv2.bias.set_((avg_model.conv2.bias.data / len(workers_model)).get())
        base_model.fc1.weight.set_((avg_model.fc1.weight.data / len(workers_model)).get())
        base_model.fc1.bias.set_((avg_model.fc1.bias.data / len(workers_model)).get())
        base_model.fc2.weight.set_((avg_model.fc2.weight.data / len(workers_model)).get())
        base_model.fc2.bias.set_((avg_model.fc2.bias.data / len(workers_model)).get())

        for models_id in workers_model:
            workers_model[models_id].get()

        for worker_id in workers_model.keys():
            workers_model[worker_id] = base_model.copy()

def test(args, model, test_loader, epoch):
    device="cpu"
    file = open("/home/savi/output1-test.txt", "a")
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

base_model = Net().to(device)

workers_model = {}
workers_optimizer = {}

for worker_id, worker in workers.items():
    print("Create a copy base_model and an optimizer for the {} ...".format(worker_id))
    workers_model[worker_id] = base_model.copy()
    workers_optimizer[worker_id] = optim.SGD(workers_model[worker_id].parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    train(args, base_model, workers_model, workers_optimizer, federated_train_loader, secure_worker, epoch)
    test(args, base_model, test_loader, epoch)

