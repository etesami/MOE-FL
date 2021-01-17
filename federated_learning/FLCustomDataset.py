from torch.utils.data import Dataset
import torch
from PIL import Image

class FLCustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # <--Data must be initialized as self.data,
        # self.train_data or self.test_data
        self.data = images
        # <--Targets must be initialized as self.targets,
        # self.test_labels or self.train_labels
        self.targets = labels
        # <--The data and target must be converted to torch
        # tensors before it is returned by __getitem__ method
        # self.to_torchtensor()
        # <--If any transforms have to be performed on the dataset
        self.transform = transform

    # def to_torchtensor(self):
    #     self.data = torch.from_numpy(self.data)
    #     self.labels = torch.from_numpy(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
