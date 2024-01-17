import torch
from torch.utils.data import Dataset
import random

class TripletDataset(Dataset):
    def __init__(self, data_dict, verbose=False):
        self.data_dict = data_dict
        self.classes = list(data_dict.keys())
        self.verbose = verbose

    def __getitem__(self, index):
        class_anchor = random.choice(list(self.data_dict.keys()))
        landmarks_anchor = random.choice(self.data_dict[class_anchor])

        # Select a positive sample from the same class
        landmarks_p = landmarks_anchor
        while landmarks_p == landmarks_anchor:
            landmarks_p = random.choice(self.data_dict[class_anchor])

        # Select a negative sample from a different class
        class_n = class_anchor

        while class_n == class_anchor:
            class_n = random.choice(list(self.data_dict.keys()))
        landmarks_n = random.choice(self.data_dict[class_n])

        # Return the triplets: anchor, positive, negative
        return torch.Tensor(landmarks_anchor), torch.Tensor(landmarks_p), torch.Tensor(landmarks_n)

    def __len__(self):
        # The length should be the same as the original dataset's length
        return sum(len(self.data_dict[c]) for c in self.classes)