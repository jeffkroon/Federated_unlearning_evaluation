"""N-CMAPSS dataset class for remaining useful life (RUL) prediction."""

import torch
from fl_module.base import BaseDataset

class NCMAPSSDataset(BaseDataset):
    def __init__(self, samples, labels):
        super(NCMAPSSDataset, self).__init__()
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
