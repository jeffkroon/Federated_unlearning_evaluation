"""Tabular dataset class."""

import torch
from fl_module.base import BaseDataset

class TabularDataset(BaseDataset):
    def __init__(self, features, labels, is_classification=True):
        super(TabularDataset, self).__init__()
        self.features = torch.tensor(features, dtype=torch.float32)
        # classification labels must be long, regression labels float
        if is_classification:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
