import torch


class Infer:

    def __init__(self, model, tokenizer, seq_len, device):
        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = device


    def infer(self, points: list) -> list:
        points = sorted(points, key=lambda p: (p[0], p[1]))
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
        return predicted


