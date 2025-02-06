import random
import torch
from utils import load_middle_row_from_file


class Infer:

    def __init__(self, model, tokenizer, seq_len, device, datafile_path, data_len):
        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = device
        self.datafile_path = datafile_path
        self.data_len = data_len

    def sampler(self):
        idx = random.randint(0, self.data_len)
        return load_middle_row_from_file(datafile_path=self.datafile_path, idx=idx)

    def infer(self) -> tuple[list, list, list]:

        points, expected, length = self.sampler()
        t_points = self.tokenizer.tokenize_points(points)

        predicted = []
        self.model.eval()
        with torch.no_grad():

            for step in range(len(t_points), self.seq_len):
                tokens = torch.tensor(t_points).unsqueeze(0).to(self.device)
                logits = self.model(tokens)
                predicted_index = logits.argmax(dim=-1).item()
                predicted.append(self.tokenizer.detokenize_token(predicted_index))
                t_points.append(self.tokenizer.tokenize_token(predicted[-1]))

        return points, expected, predicted


