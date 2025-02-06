import json
from pathlib import Path
import torch
from dataclasses import asdict, dataclass

from tokenizer import Tokenizer
from infer import Infer

from dataclasses import dataclass, field
from pathlib import Path
import torch

metadata_file = 'weights/metadata.json'
model_file = 'weights/try14_12.pt'

@dataclass
class ConfigDatasets:
    len_dataset_train: int = 100000
    len_dataset_test: int = 10000
    board_size: tuple[int, int] = (10, 10)
    min_n_points: int = 6
    max_n_points: int = 6
    train_datafile_path: str = ''
    test_datafile_path: str = ''
    train_len_data: int = 0
    test_len_data: int = 0



@dataclass
class ConfigTokenizer:
    board_size: tuple[int, int]
    special_tokens: list[str] = field(default_factory=lambda: ['mask', 'sop', 'eop'])
    vocab_size: int = 0
    sop_token: str = 'sop'
    eop_token: str = 'eop'
    mask_token: str = 'mask'

    def __post_init__(self):
        self.vocab_size = (self.board_size[0] * self.board_size[1]) + len(self.special_tokens)


@dataclass
class ConfigModel:
    vocab_size: int
    board_size: tuple[int, int]
    d_model: int = 512
    n_layers: int = 2
    output_dim: int = 0
    num_heads: int = 4
    head_dim: int = 0
    mask: bool = False
    mlp_dim: int = 4096

    def __post_init__(self):
        self.output_dim = (self.board_size[0] * self.board_size[1]) # + 1  # for mask_token
        self.head_dim = self.d_model // self.num_heads
        assert self.d_model % self.num_heads == 0


@dataclass
class Config:
    config_datasets: ConfigDatasets = field(default_factory=ConfigDatasets)
    config_tokenizer: ConfigTokenizer = field(init=False)
    config_model: ConfigModel = field(init=False)
    device: str = field(init=False)
    seq_len: int = field(init=False)

    def __post_init__(self):
        self.config_tokenizer = ConfigTokenizer(self.config_datasets.board_size)
        self.config_model = ConfigModel(self.config_tokenizer.vocab_size, self.config_datasets.board_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.seq_len = (self.config_datasets.max_n_points * 2) + 2  # Sequence length



##############



def load_config_from_json(config) -> Config:
    """Loads a Config object from a JSON file."""
    with open(metadata_file, 'r') as f:
        config_dict = json.load(f)

    config = Config()

    # Restore dataset config
    config.config_datasets.__dict__.update(config_dict["config_datasets"])

    # Restore tokenizer config
    board_size = tuple(config_dict["config_datasets"]["board_size"])
    config.config_tokenizer = ConfigTokenizer(board_size)
    config.config_tokenizer.__dict__.update(config_dict["config_tokenizer"])


    # Restore model config
    vocab_size = config.config_tokenizer.vocab_size
    config.config_model = ConfigModel(vocab_size, board_size)
    config.config_model.__dict__.update(config_dict["config_model"])

    # Restore device and seq_len
    config.seq_len = config_dict["seq_len"]

    return config



def load_pretrained_model(config: Config, model):
    config = load_config_from_json(config)
    state = torch.load(model_file, map_location=torch.device(config.device))
    model.load_state_dict(state['model_state_dict'])


##############


class API:

    @staticmethod
    def tokenizer(config: Config):
        special_tokens: list = config.config_tokenizer.special_tokens
        vocab_size: int = config.config_tokenizer.vocab_size
        board_size: tuple[int, int] = config.config_datasets.board_size
        seq_len: int = config.seq_len
        return Tokenizer(special_tokens, vocab_size, board_size, seq_len)

    @staticmethod
    def infer(config: Config, model, tokenizer):
        seq_len = config.seq_len
        device = config.device
        return Infer(model, tokenizer, seq_len, device)
