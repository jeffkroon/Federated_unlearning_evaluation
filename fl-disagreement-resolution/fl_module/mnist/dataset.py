"""MNIST dataset class for image classification."""

import torch
from fl_module.base import BaseDataset

class MNISTDataset(BaseDataset):
    def __init__(self, images, labels):
        super(MNISTDataset, self).__init__()
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
