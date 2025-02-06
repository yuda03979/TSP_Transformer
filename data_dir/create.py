from data_dir.sample_points import SamplePoints
from data_dir.solver import Solver

import time
import random
from tqdm import tqdm


class CreateData:

    def __init__(self, train_datafile_path, test_datafile_path, train_len_data, test_len_data, min_n_points: int,
                 max_n_points: int, board_size: tuple, batch: int, mult: int):
        # the batch doesn't relate to the batch_size of the model!
        self.train_datafile_path = train_datafile_path
        self.test_datafile_path = test_datafile_path
        self.train_len_data = train_len_data
        self.test_len_data = test_len_data
        self.min_n_points = min_n_points
        self.max_n_points = max_n_points
        self.board_size = board_size
        self.batch = batch

        self.mult = mult
        assert ((self.train_len_data // self.batch) // self.mult), 'please increase the train data length'
        assert ((self.test_len_data // self.batch) // self.mult), 'please increase the test data length'

        self.sampler = SamplePoints(board_size=self.board_size)
        self.solver = Solver()

    @staticmethod
    def shuffle_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line for line in f.readlines() if line.count('|') == 2 and line.strip()]  # Keep only valid lines

        random.shuffle(lines)  # Shuffle the list in place

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)  # Write back the shuffled lines

    def write_batch_to_file(self, batch_samples, filename):
        with open(filename, 'a') as file:
            for _ in range(2):
                for points, route, distance in batch_samples:
                    route = route[::-1]
                    file.write(f"{sorted(points, key=lambda p: (p[0], p[1]))}|{route}|{distance}\n")

    def create_train(self):
        train_tqdm = tqdm(range((self.train_len_data // self.batch) // 2), f"generating train data")
        for i in train_tqdm:
            n_points = random.randint(self.min_n_points, self.max_n_points)
            batch_points = self.sampler.sample_train(n_points, self.batch)
            batch_samples = self.solver.solve_parallel(batch_points)
            samples = []
            for board, solution in zip(batch_points, batch_samples):
                samples.append([board, solution[0], solution[1]])
            self.write_batch_to_file(samples, self.train_datafile_path)
        self.shuffle_file(self.train_datafile_path)

    def create_test(self):
        test_tqdm = tqdm(range((self.test_len_data // self.batch) // 2), desc="generating test data")
        for i in test_tqdm:
            n_points = random.randint(self.min_n_points, self.max_n_points)
            batch_points = self.sampler.sample_test(n_points, self.batch)
            batch_samples = self.solver.solve_parallel(batch_points)
            samples = []
            for board, solution in zip(batch_points, batch_samples):
                samples.append([board, solution[0], solution[1]])
            self.write_batch_to_file(samples, self.test_datafile_path)
        self.shuffle_file(self.test_datafile_path)


if __name__ == '__main__':

    board_size: tuple | list = [10, 10]
    max_n_points: int = 6
    min_n_points: int = 6
    train_datafile_path = 'train_data.txt'
    test_datafile_path = 'test_data.txt'
    train_len_data = 2_000_000
    test_len_data = 100000
    batch = 4000
    mult = 20

    creator = CreateData(train_datafile_path, test_datafile_path, train_len_data, test_len_data, min_n_points,
                 max_n_points, board_size, batch, mult)
    s = time.time()
    creator.create_train()
    creator.create_test()
    print(time.time() - s)
