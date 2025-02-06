from config import Config, API, load_pretrained_model
from model import TSPTransformer
from utils import save_graph

from pathlib import Path
import time
from tqdm import tqdm
import subprocess
import torch
from torch.utils.tensorboard import SummaryWriter


class Train:

    def __init__(self, preload=False, preload_epoch=0):
        self.config = Config()
        self.model = TSPTransformer(self.config.config_model).to(self.config.device)
        self.tokenizer = API.tokenizer(self.config)
        self.infer = API.infer(self.config, self.model, self.tokenizer)
        self.dataloaders = API.dataloaders(self.config, self.tokenizer)

        config_exp = self.config.config_exp
        self.tensorboard_folder = config_exp.tensorboard_folder
        self.logs_txt_file = config_exp.logs_txt_file
        self.weights_folder = config_exp.weights_folder
        self.exp_name = config_exp.exp_name
        self.images_folder = config_exp.images_folder

        self.device = self.config.device
        self.num_epochs = self.config.config_train.num_epochs
        self.lr = self.config.config_train.lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(lr=self.lr, params=self.model.parameters())
        self.writer = SummaryWriter(log_dir=self.tensorboard_folder)
        self.tensorboard_process = self.start_tensorboard()

        self.preload = preload
        self.preload_epoch = preload_epoch
        self.global_step = 0
        self.start_epoch = 0
        self.train_loss = []
        self.test_loss = []

        if self.preload:
            self.load()

    def load(self):
        self.start_epoch, self.global_step, self.train_loss, self.test_loss = load_pretrained_model(
            self.config,
            epoch=self.preload_epoch,
            model=self.model,
            optimizer=self.optimizer)

    def start_tensorboard(self):
        self.tensorboard_process = subprocess.Popen(
            ['tensorboard',
             f'--logdir={self.tensorboard_folder}',
             '--reload_interval=5']
        )
        return self.tensorboard_process

    def log_epoch_file(self, epoch, infer_example):
        with open(self.logs_txt_file, "a") as f:
            f.write(f"\n{'-' * 50}\n")
            f.write(f"Epoch {epoch}/{self.num_epochs + self.start_epoch}\n")
            f.write(f"Train Loss: {self.train_loss[-1]:.4f}\n")
            f.write(f"Test Loss: {self.test_loss[-1]:.4f}\n")
            f.write(f"{infer_example=}\n")

    def train(self):

        for epoch in range(self.start_epoch, self.num_epochs + self.start_epoch):

            self.train_loss.append(self.train_epoch(epoch))
            self.test_loss.append(self.evaluate(epoch))
            infer_example = self.infer_func(epoch)

            print(f"\n{'-' * 50}\n")
            print(f"Epoch {epoch}/{self.num_epochs + self.start_epoch}")
            print(f"Train Loss: {self.train_loss[-1]:.4f}")
            print(f"Test Loss: {self.test_loss[-1]:.4f}")
            print(f"{infer_example=}")
            print(f"\n{'-' * 50}\n")

            self.log_epoch_file(epoch, infer_example)
            # write the graph of learning
            self.writer.add_scalar("Loss/train", self.train_loss[epoch], epoch)
            self.writer.add_scalar("Loss/test", self.test_loss[epoch], epoch)

            # save the weights of the epoch
            weights_filename = Path(self.weights_folder) / f"{self.exp_name}_{epoch}.pt"
            checkpoint = dict(
                epoch=epoch,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                global_step=self.global_step,
                train_loss=self.train_loss,
                test_loss=self.test_loss
            )
            torch.save(checkpoint, weights_filename)

        self.writer.close()
        self.tensorboard_process.terminate()

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        num_iterations = 0

        batch_iterator = tqdm(self.dataloaders.get_train_dataloader(), f'epoch: {epoch}; training... ')
        start_time = time.time()
        for seq_batch_tokens, seq_batch_targets in batch_iterator:
            for batch_tokens, batch_targets in zip(seq_batch_tokens, seq_batch_targets):

                batch_tokens = batch_tokens.to(self.device)
                batch_targets = batch_targets.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(batch_tokens)
                loss = self.criterion(logits, batch_targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_iterations += 1

                self.global_step += 1

            if int(time.time() - start_time) >= 60:
                time.sleep(10)
                start_time = time.time()

        return total_loss / num_iterations

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():

            total_loss = 0
            num_iterations = 0

            batch_iterator = tqdm(self.dataloaders.get_test_dataloader(), f'epoch: {epoch}; evaluating... ')
            for seq_batch_tokens, seq_batch_targets in batch_iterator:
                for batch_tokens, batch_targets in zip(seq_batch_tokens, seq_batch_targets):

                    batch_tokens = batch_tokens.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    logits = self.model(batch_tokens)
                    loss = self.criterion(logits, batch_targets)

                    total_loss += loss.item()
                    num_iterations += 1

        return total_loss / num_iterations

    def infer_func(self, epoch):
        self.model.eval()
        points, expected, predicted = self.infer.infer()
        save_graph(self.tokenizer.board_size, points, expected, predicted, f'{self.images_folder}/{self.exp_name}_{epoch}.png')
        return points, predicted


if __name__ == '__main__':
    train = Train(preload=True, preload_epoch=12)
    train.train()

