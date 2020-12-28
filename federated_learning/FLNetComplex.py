import torch.nn as nn
import torch.nn.functional as F

# class FLNet(nn.Module):
#      def __init__(self):
#          super(FLNet, self).__init__()
#          self.conv1 = nn.Conv2d(1, 20, 5, 1)
#          self.conv2 = nn.Conv2d(20, 50, 5, 1)
#          self.fc1 = nn.Linear(4 * 4 * 50, 500)
#          self.fc2 = nn.Linear(500, 10)

#      def forward(self, x):
#          x = F.relu(self.conv1(x))
#          x = F.max_pool2d(x, 2, 2)
#          x = F.relu(self.conv2(x))
#          x = F.max_pool2d(x, 2, 2)
#          x = x.view(-1, 4 * 4 * 50)
#          x = F.relu(self.fc1(x))
#          x = self.fc2(x)
#          return F.log_softmax(x, dim=1) 

class FLNetComplex(nn.Module):
    def __init__(self):
        super(FLNetComplex, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1) # in channel: 1, out channel (filters): 32, kernel: 5
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 64, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print("Conv1: {}".format(x.shape))

        x = F.max_pool2d(x, 2, 2) # Kernel size: 2, stride: 2
        # print("mac Pool: {}".format(x.shape))

        x = F.relu(self.conv2(x))
        # print("Conv2: {}".format(x.shape))

        x = F.max_pool2d(x, 2, 2)
        # print("Max Pool: {}".format(x.shape))

        x = x.view(-1, 4 * 4 * 64) # Reshape
        # print("Reshape: {}".format(x.shape))

        x = F.relu(self.fc1(x))
        # print("fc1: {}".format(x.shape))

        x = self.fc2(x)
        # print("fc2: {}".format(x.shape))
        return F.log_softmax(x, dim=1)