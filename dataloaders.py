import ast
import linecache
import random
from pathlib import Path

from utils import gen_batch_line
import torch
from torch.utils.data import Dataset, DataLoader


class TSPDataset(Dataset):

    def __init__(self, tokenizer, len_data, len_dataset, datafile_path: str):
        assert len_data >= len_dataset, f'{len_data=}, {len_dataset=}'
        self.tokenizer = tokenizer
        self.len_data = len_data  # the length of the actual data
        self.len_dataset = len_dataset  # the amount of samples per epoch
        if isinstance(datafile_path, Path):
            datafile_path = str(datafile_path.resolve())
        self.datafile_path = datafile_path
        self.batch_rows = 10000
        self.gen_row = gen_batch_line(self.datafile_path, random.randint(0, self.len_data - self.batch_rows), self.batch_rows)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        try:
            points, expected, length = next(self.gen_row)
        except:
            idx = random.randint(0, self.len_data - self.batch_rows)
            self.gen_row = gen_batch_line(self.datafile_path, idx, self.batch_rows)
            points, expected, length = next(self.gen_row)

        t_points = self.tokenizer.tokenize_points(points)
        t_expected = self.tokenizer.tokenize_path(expected)

        samples = []
        targets = []

        for i in range(len(t_expected)):
            targets.append(t_expected[i])
            samples.append(torch.tensor(t_points + t_expected[:i]))

        return samples, targets


class Dataloaders:

    def __init__(self, tokenizer, len_dataset_train, len_dataset_test, train_len_data, test_len_data, train_datafile_path, test_datafile_path, batch_size):
        self.train_dataset = TSPDataset(tokenizer=tokenizer, len_dataset=len_dataset_train,
                                        len_data=train_len_data, datafile_path=train_datafile_path)
        self.test_dataset = TSPDataset(tokenizer=tokenizer, len_dataset=len_dataset_test,
                                       len_data=test_len_data, datafile_path=test_datafile_path)

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size,
                                           shuffle=False)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size,
                                          shuffle=False)

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_test_dataloader(self):
        return self.test_dataloader