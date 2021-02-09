import itertools
import random

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader

from utils import collate_fn


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y=None, device="cpu"):
        self.X = X
        self.Y = Y
        self.device = device
    
    def __len__(self):
        return len(self.X["input_ids"])

    def __getitem__(self, index):
        to_return = {key: torch.LongTensor(value[index]).to(self.device) for key, value in self.X.items()}
        to_return["index"] = index
        if self.Y is not None:
            to_return["labels"] = self.Y[index].to(self.device)
        return to_return

def make_dataset(tokenizer, data, has_labels=True, device="cpu", 
                 first_key="sentence1", second_key="sentence2", 
                 answer_field="answer", pos_label=True):
    questions = [elem[first_key] for elem in data]
    passages = [elem[second_key] for elem in data]
    X = tokenizer(text=questions,  text_pair=passages, truncation=True)
    if has_labels:
        Y = torch.FloatTensor([int(elem[answer_field]==pos_label) for elem in data])
    else:
        Y = None
    return SimpleDataset(X, Y, device=device)


class OrderedBatchSampler(Sampler):
    
    def __init__(self, data, batch_size, length_func=None, shuffle=True, random_state=187):
        if length_func is None:
            length_func = lambda x: 0
        self.order = sorted(range(len(data)), key=lambda i: length_func(data[i]), reverse=True)
        self.batch_size = batch_size
        self.batches = [self.order[start:start+batch_size] for start in range(0, len(self.order), batch_size)]
        random.seed(random_state)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def make_dataloader(dataset, batch_size=16, shuffle=True, key="input_ids"):
    length_func = lambda x: len(x[key]) if key else None
    sampler = OrderedBatchSampler(dataset, batch_size=batch_size, 
                                  length_func=length_func, shuffle=shuffle)
    return DataLoader(dataset, collate_fn=collate_fn, batch_sampler=sampler)