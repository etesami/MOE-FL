import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
import cvxpy as cp
import numpy as np
import logging
from mnist import MNIST
from federated_learning.FLCustomDataset import FLCustomDataset
from federated_learning.FLNet import FLNet

class Arguments():
    def __init__(self, epochs_num, workers_num, use_cuda = False):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = epochs_num
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False
        self.workers_num = workers_num


class FederatedLearning():

    # Initializing variables
    def __init__(self, workers_num = 10, epochs_num = 5, output_prefix = None, mnist_path = None, use_cuda = False, log_level = logging.INFO):
        logging.basicConfig(format='%(asctime)s %(message)s', level=log_level)
        logging.info("Creating a new test...")

        self.workers = {}
        self.server = None

        self.server_model = None
        self.workers_model = dict()

        self.train_images = None
        self.train_labels = None
        self.hook = sy.TorchHook(torch)
        
        # use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.args = Arguments(epochs_num = epochs_num, workers_num = workers_num, use_cuda = use_cuda)
        
        torch.manual_seed(self.args.seed)

        if output_prefix is None or mnist_path is None:
            raise Exception("Sorry, you should specify the output/mnist path!")
        self.output_prefix = output_prefix
        self.mnist_path = mnist_path


    def create_workers(self):
        logging.info("Creating {} workers".format(self.args.workers_num))
        for num in range(0, self.args.workers_num):
            worker_id = "worker" + str(num)
            logging.debug("Creating the worker: {}".format(worker_id))
            self.workers[worker_id] = sy.VirtualWorker(self.hook, id=worker_id)

    def create_server(self):
        logging.info("Creating the server")
        self.server = sy.VirtualWorker(self.hook, id="server")


    def load_data(self):
        mndata = MNIST(self.mnist_path + '/raw')
        logging.info("Loading the dataset")
        train_images, train_labels = mndata.load_training()
        self.train_images = np.asarray(train_images, dtype=np.uint8).reshape(-1, 28, 28)
        self.train_labels = np.asarray(train_labels)

        indices = np.arange(self.train_images.shape[0])
        np.random.shuffle(indices)
        self.train_images = self.train_images[indices]
        self.train_labels = self.train_labels[indices]

    def count_digits(self):
        count = {}
        for i in range(0,10):
            count[i] = 0
        for d in self.train_labels:
            count[d] = count[d] + 1

        logging.info("Percentage of digits in whole training dataset: {}".format(
            [round(d*100.0/len(self.train_labels),2) for _, d in count.items()]))
        with open(self.output_prefix + "_digits", "w") as f:
            f.write(' '.join([str(round(d*100.0/len(self.train_labels),2)) for _, d in count.items()]))
            f.close()


    def create_datasets(self):
        logging.info("Creating train_dataset_loader and test_dataset_loader")
        custom_dataset = FLCustomDataset(self.train_images, self.train_labels, 
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
      
        train_dataset_loader = sy.FederatedDataLoader(
            custom_dataset.federate(tuple([ww for id, ww in self.workers.items()])),
            batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

        test_dataset_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=self.args.test_batch_size)
        
        return train_dataset_loader, test_dataset_loader


    def create_aggregated_data(self):
        fraction = 0.2
        logging.info("Creating the aggregated data for the server")
        aggregated_label = self.train_labels.copy()
        aggregated_data = self.train_images.copy()
        indices = np.array([], dtype = np.int64)
        for i in range(0, len(self.workers)):
            step = int(len(self.train_labels) / len(self.workers))
            logging.debug("Selecting {} records from index {} to {} from [worker{}]".format(
                int(fraction * step), i * step, i * step + int((i + fraction) * step), i))
            new_indices_to_extract = np.arange(i * step, int((i + fraction) * step), dtype = np.int64)
            indices = np.concatenate((indices, new_indices_to_extract))

        aggregated_data = aggregated_data[indices]
        aggregated_label = aggregated_label[indices]
        
        dataset_server = FLCustomDataset(aggregated_data, aggregated_label,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))]))
        
        federated_server_loader = sy.FederatedDataLoader(
            dataset_server.federate(tuple([self.server])),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        
        return federated_server_loader


    def send_model(self, model, location, location_id):
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


    def getback_model(self, model):
        if isinstance(model, dict):
            for ww_id, ww in model.items():
                if ww.location is not None:
                    ww.get()
        elif model.location is not None:
            model.get()

    # '''
    # Attack 1
    # Permute all labels for given workers' id
    # workers_id_list: the list of workers' id (zero-based)
    # '''
    def attack_permute_labels_randomly(self, workers_percentage):
        logging.info("ATTACK 1: Permute labels of {} percentage of workers".format(workers_percentage))
        
        # Find workers which are counted as malicious users
        workers_id_list = None
        if 20 <= workers_percentage and workers_percentage < 40:
            workers_id_list = np.array([0, 1])
        elif 40 <= workers_percentage and workers_percentage < 50:
            workers_id_list = np.array([0, 1, 2, 3])
        elif 50 <= workers_percentage and workers_percentage < 60:
            workers_id_list = np.array([0, 1, 2, 3, 4])
        elif 60 <= workers_percentage and workers_percentage < 80:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5])
        elif 80 <= workers_percentage and workers_percentage < 100:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        elif workers_percentage == 100:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        logging.debug("Affected workers: {}".format(workers_id_list))  
        
        step = int(len(self.train_labels) / len(self.workers))
        for i in workers_id_list: 
            logging.debug("-- Permute all labels from index {} to {}".format(
                i * step, (i + 1) * step))
            self.train_labels[i * step:(i + 1) * step] = \
                        np.random.permutation(self.train_labels[i * step:(i + 1) * step])

    def attack_permute_labels_collaborative(self, workers_percentage, data_percentage):
        logging.info("ATTACK 2: Permute {} percentage of labels of the {} percentage of workers".format(data_percentage, workers_percentage))
        
        # Find workers which are counted as malicious users
        workers_id_list = None
        if 20 <= workers_percentage and workers_percentage < 40:
            workers_id_list = np.array([0, 1])
        elif 40 <= workers_percentage and workers_percentage < 50:
            workers_id_list = np.array([0, 1, 2, 3])
        elif 50 <= workers_percentage and workers_percentage < 60:
            workers_id_list = np.array([0, 1, 2, 3, 4])
        elif 60 <= workers_percentage and workers_percentage < 80:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5])
        elif 80 <= workers_percentage and workers_percentage < 100:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        elif workers_percentage == 100:
            workers_id_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        logging.debug("Affected workers: {}".format(workers_id_list))        

        # Find labels which are going to be permuted base on the value of the percentage
        labels_to_be_changed = None
        if 20 <= data_percentage and data_percentage < 40:
            labels_to_be_changed = np.array([0, 1])
        elif 40 <= data_percentage and data_percentage < 60:
            labels_to_be_changed = np.array([0, 1, 2, 3])
        # I tried to add an option for 50% changes in data but it is not 
        # rasy with the current implementation.
        elif 60 <= data_percentage and data_percentage < 80:
            labels_to_be_changed = np.array([0, 1, 2, 3, 4, 5])
        elif 80 <= data_percentage and data_percentage < 100:
            labels_to_be_changed = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        elif data_percentage == 100:
            labels_to_be_changed = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        logging.debug("Affected labels: {}".format(labels_to_be_changed))

        # Find index of each number in the train_label and store them
        # into a dic named labels_indexes
        labels_indexes = {}
        for i in range(0, 10):
            labels_indexes[i] = np.array([], dtype = np.int64)
        
        # Initialization of indexes
        index = 0
        for n in self.train_labels:
            labels_indexes[n] = np.concatenate((labels_indexes[n], [index]))
            index = index + 1

        step = int(len(self.train_labels) / len(self.workers))
        # Start permutation
        for i in workers_id_list:
            for l in range(0, len(labels_to_be_changed), 2):
                
                # ex.
                # labels_to_be_changed = [0, 1]
                # labels_to_be_changed[l] = 0
                # labels_to_be_changed[l + 1] = 1 
                # labels_indexes[0] = list if indexes of 0
                logging.debug("-- Permute {} with {} from index {} to {}".format(
                    labels_to_be_changed[l], labels_to_be_changed[l+1],
                    i * step, (i + 1) * step
                    ))

                indexes_first_digit = np.where(
                    (i * step <= labels_indexes[labels_to_be_changed[l]]) &
                    (labels_indexes[labels_to_be_changed[l]] < (i + 1) * step)
                )[0]
                logging.debug("-- To be verified: Some indexes of {}: {}".format(
                    labels_to_be_changed[l], 
                    labels_indexes[labels_to_be_changed[l]][indexes_first_digit][0:10])
                )
                
                indexes_sec_digit = np.where(
                    (i * step <= labels_indexes[labels_to_be_changed[l + 1]]) &
                    (labels_indexes[labels_to_be_changed[l + 1]] < (i + 1) * step)
                )[0]
                logging.debug("-- To be verified: Some indexes of {}: {}".format(
                    labels_to_be_changed[l + 1], 
                    labels_indexes[labels_to_be_changed[l + 1]][indexes_sec_digit][0:10])
                )

                self.train_labels[labels_indexes[labels_to_be_changed[l]][indexes_first_digit]] = labels_to_be_changed[l + 1]
                self.train_labels[labels_indexes[labels_to_be_changed[l + 1]][indexes_sec_digit]]= labels_to_be_changed[l]


    def train_workers(self, federated_train_loader, epoch):
        workers_opt = {}
        file = open(self.output_prefix + "_train", "a")
        for ww_id, ww in self.workers.items():
            if self.workers_model[ww_id].location is None \
                    or self.workers_model[ww_id].location.id != ww_id:
                self.workers_model[ww_id].send(ww)
            workers_opt[ww_id] = optim.SGD(params=self.workers_model[ww_id].parameters(), lr=self.args.lr)

        for batch_idx, (data, target) in enumerate(federated_train_loader):
            worker_id = data.location.id
            worker_model = self.workers_model[worker_id]
            worker_opt = workers_opt[worker_id]
            worker_model.train()
            data, target = data.to(self.device), target.to(self.device)
            worker_opt.zero_grad()
            output = worker_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            worker_opt.step()

            if batch_idx % self.args.log_interval == 0:
                loss = loss.get()
                TO_FILE = '{} {} {} {}\n'.format(epoch, batch_idx, data.location.id, loss)
                file.write(TO_FILE)
                logging.info('Train Epoch: {} [{}] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, worker_id, batch_idx, batch_idx * self.args.batch_size, len(federated_train_loader) * self.args.batch_size,
                                                100. * batch_idx / len(federated_train_loader), loss.item()))
        # Need to getback the self.workers_model
        file.close()
        print()


    def train_server(self, federated_train_server_loader, epoch):
        file = open(self.output_prefix + "_train_server", "a")
        self.send_model(self.server_model, self.server, "server")
        server_opt = optim.SGD(self.server_model.parameters(), lr=self.args.lr)
        for batch_idx, (data, target) in enumerate(federated_train_server_loader):
            self.server_model.train()
            data, target = data.to(self.device), target.to(self.device)
            server_opt.zero_grad()
            output = self.server_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            server_opt.step()

            if batch_idx % self.args.log_interval == 0:
                loss = loss.get()
                TO_FILE = '{} {} {} {}\n'.format(epoch, batch_idx, data.location.id, loss)
                file.write(TO_FILE)
                logging.info('Train Epoch: {} [server] [{}: {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, batch_idx * self.args.batch_size, len(federated_train_server_loader) * self.args.batch_size,
                                    100. * batch_idx / len(federated_train_server_loader), loss.item()))
        file.close()
        # Always need to get back the model
        self.getback_model(self.server_model)
        print()


    def test(self, model, test_loader, epoch):
        file = open(self.output_prefix + "_test", "a")
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        TO_FILE = '{} {} "{{/*0.80 Accuracy:}}\\n{}%" {}\n'.format(
            epoch, test_loss, 
            100. * correct / len(test_loader.dataset),
            100. * correct / len(test_loader.dataset))
        file.write(TO_FILE)
        file.close()
        logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    def find_best_weights(self, epoch):
        # reference_model = self.server_model
        # workers_model = self.workers_model
        file = open(self.output_prefix + "_weights", "a")
        self.getback_model(self.server_model)
        with torch.no_grad():
            reference_layers = [None] * 8
            reference_layers[0] = self.server_model.conv1.weight.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[1] = self.server_model.conv1.bias.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[2] = self.server_model.conv2.weight.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[3] = self.server_model.conv2.bias.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[4] = self.server_model.fc1.weight.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[5] = self.server_model.fc1.bias.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[6] = self.server_model.fc2.weight.data.numpy().copy().reshape(-1, 1).ravel()
            reference_layers[7] = self.server_model.fc2.bias.data.numpy().copy().reshape(-1, 1).ravel()

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
            self.getback_model(self.workers_model)
            for worker_id, worker_model in self.workers_model.items():
                workers_params[worker_id] = [[] for i in range(8)]
                workers_params[worker_id][0] = worker_model.conv1.weight.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][1] = worker_model.conv1.bias.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][2] = worker_model.conv2.weight.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][3] = worker_model.conv2.bias.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][4] = worker_model.fc1.weight.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][5] = worker_model.fc1.bias.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][6] = worker_model.fc2.weight.data.numpy().copy().reshape(-1, 1)
                workers_params[worker_id][7] = worker_model.fc2.bias.data.numpy().copy().reshape(-1, 1)

            # logging.debug("workers_param shape: {}".format(len(workers_params)))
            # for key in workers_params:
            #     logging.debug("workers_param[{}]: {}".format(key, len(workers_params[key])))
            #     if key == "worker0":
            #         for i in range(0, len(workers_params[key])):
            #             logging.debug("workers_param[{}][{}]: {}".format(key, i, len(workers_params[key][i])))
            """
            --> conv1.weight
            workers_all_params[0] =
                [workers_param[worker0][0], workers_param[worker1][0], workers_param[worker2][0]]
            --> conv1.bias
            workers_all_params[1] =
                [workers_param[worker0][1], workers_param[worker1][1], workers_param[worker2][1]]
            """

            workers_all_params = []
            logging.info("Start the optimization....")
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

            # logging.debug("workers_all_param: {}".format(len(workers_all_params)))
            # for i in range(len(workers_all_params)):
            #     logging.debug("workers_all_params[{}]: {}".format(i, workers_all_params[i].shape))

            W = cp.Variable(len(self.workers_model))

            objective = cp.Minimize(cp.norm2(cp.matmul(workers_all_params[0], W) - reference_layers[0]) +
                                    cp.norm2(cp.matmul(workers_all_params[1], W) - reference_layers[1]) +
                                    cp.norm2(cp.matmul(workers_all_params[2], W) - reference_layers[2]) +
                                    cp.norm2(cp.matmul(workers_all_params[3], W) - reference_layers[3]) +
                                    cp.norm2(cp.matmul(workers_all_params[4], W) - reference_layers[4]) +
                                    cp.norm2(cp.matmul(workers_all_params[5], W) - reference_layers[5]) +
                                    cp.norm2(cp.matmul(workers_all_params[6], W) - reference_layers[6]) +
                                    cp.norm2(cp.matmul(workers_all_params[7], W) - reference_layers[7]))

            for i in range(len(workers_all_params)):
                logging.debug("Mean [{}]: {}".format(i, np.round(np.mean(workers_all_params[i],0) - np.mean(reference_layers[i],0),6)))
                logging.debug("")

            constraints = [0 <= W, W <= 1, sum(W) == 1]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.MOSEK)
            logging.info(W.value)
            logging.info("")
            TO_FILE = '{} {}\n'.format(epoch, np.array2string(W.value).replace('\n',''))
            file.write(TO_FILE)
            # file.close()
            return W.value


    def update_models(self, W, server_model, workers_model):
        self.getback_model(workers_model)
        self.getback_model(server_model)
        tmp_model = FLNet().to(self.device)

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

    def create_server_model(self):
        self.server_model = FLNet().to(self.device)

    def create_workers_model(self):
        for worker_id, worker in self.workers.items():
            self.workers_model[worker_id] = self.server_model.copy()
