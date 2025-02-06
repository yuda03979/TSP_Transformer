import random

class SamplePoints:

    def __init__(self, board_size):
        self.board_size = board_size
        self.available_points = list((i, j) for i in range(self.board_size[0]) for j in range(self.board_size[1]))
        self.available_test_points = list((i, j) for i in range(0, self.board_size[0], 2) for j in range(self.board_size[1]))

    def sample_train(self, n_points: int, batch: int):
        # in the train we cannot assign all point[0] % 2 == 0
        assert n_points < len(self.available_points)

        def train_points(points) -> bool:
            for p in points:
                if p[0] % 2 == 1:
                    return True
            return False

        batch_points = []
        for _ in range(batch):
            points = random.sample(self.available_points, k=n_points)
            while not train_points(points):
                points = random.sample(self.available_points, k=n_points)
            batch_points.append(points)
        return batch_points

    def sample_test(self, n_points: int, batch: int):
        assert n_points < len(self.available_test_points)

        batch_points = []
        for _ in range(batch):
            points = random.sample(self.available_test_points, k=n_points)
            batch_points.append(points)
        return batch_points


