import json
from pathlib import Path
import torch
from dataclasses import asdict, dataclass

from utils import get_len_data
from tokenizer import Tokenizer
from dataloaders import Dataloaders
from infer import Infer

from dataclasses import dataclass, field
from pathlib import Path
import torch


def exp_name_gen():
    exp_name = input("Enter the experiment name:\n")
    yield exp_name


exp_name = exp_name_gen()


@dataclass
class ConfigDatasets:
    len_dataset_train: int = 100000
    len_dataset_test: int = 50000
    board_size: tuple[int, int] = (10, 10)
    min_n_points: int = 6
    max_n_points: int = 6
    train_datafile_path: Path = Path('data_dir') / 'train_data.txt'
    test_datafile_path: Path = Path('data_dir') / 'test_data.txt'
    train_len_data: int = 0
    test_len_data: int = 0

    def __post_init__(self):
        self.train_len_data = get_len_data(self.train_datafile_path)
        self.test_len_data = get_len_data(self.test_datafile_path)
        print(f"{self.train_len_data=}, {self.test_len_data=}")


@dataclass
class ConfigTokenizer:
    board_size: tuple[int, int]
    special_tokens: list[str] = field(default_factory=lambda: ['mask', 'sop', 'eop'])
    vocab_size: int = 0
    mask_token: str = 'mask'
    sop_token: str = 'sop'
    eop_token: str = 'eop'

    def __post_init__(self):
        self.vocab_size = (self.board_size[0] * self.board_size[1]) + len(self.special_tokens)


@dataclass
class ConfigExp:
    model_base_name: str = 'TSP_transformers'
    global_folder: str = field(init=False)
    exp_name: str = next(exp_name)
    exp_folder: Path = field(init=False)
    weights_folder: Path = field(init=False)
    images_folder: Path = field(init=False)
    trainer_folder: Path = field(init=False)
    metadata_file: Path = field(init=False)
    tensorboard_folder: Path = field(init=False)
    logs_txt_file: Path = field(init=False)

    def __post_init__(self):
        self.global_folder = f'pretrained_{self.model_base_name}'
        self.exp_folder = Path(self.global_folder) / self.exp_name
        self.weights_folder = self.exp_folder / '_weights'
        self.images_folder = self.exp_folder / '_images'
        self.trainer_folder = self.exp_folder / '_trainer'
        self.metadata_file = self.exp_folder / 'metadata.json'
        self.tensorboard_folder = self.trainer_folder / 'tensorboard_logs'
        self.logs_txt_file = self.trainer_folder / 'logs.txt'

        for folder in [self.weights_folder, self.images_folder, self.tensorboard_folder]:
            folder.mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigTrain:
    batch_size: int = 1024
    lr: float = 1e-4
    num_epochs: int = 50


@dataclass
class ConfigModel:
    vocab_size: int
    board_size: tuple[int, int]
    d_model: int = 512  # 1024
    n_layers: int = 2
    output_dim: int = 0
    num_heads: int = 4
    head_dim: int = 0
    mask: bool = False
    mlp_dim: int = 4096

    def __post_init__(self):
        self.output_dim = (self.board_size[0] * self.board_size[1])  # + 1  # for mask_token
        self.head_dim = self.d_model // self.num_heads
        assert self.d_model % self.num_heads == 0


@dataclass
class Config:
    config_datasets: ConfigDatasets = field(default_factory=ConfigDatasets)
    config_tokenizer: ConfigTokenizer = field(init=False)
    config_exp: ConfigExp = field(default_factory=ConfigExp)
    config_train: ConfigTrain = field(default_factory=ConfigTrain)
    config_model: ConfigModel = field(init=False)
    device: str = field(init=False)
    seq_len: int = field(init=False)

    def __post_init__(self):
        self.config_tokenizer = ConfigTokenizer(self.config_datasets.board_size)
        self.config_model = ConfigModel(self.config_tokenizer.vocab_size, self.config_datasets.board_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.seq_len = (self.config_datasets.max_n_points * 2) + 2  # Sequence length

        save_config_to_json(self)


##############


def save_config_to_json(config: Config):
    """Saves the Config object as a JSON file."""
    filename = config.config_exp.metadata_file

    def serialize(obj):
        if isinstance(obj, Path):
            return obj.name  # Convert Path to string
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    config_dict = {
        "config_datasets": asdict(config.config_datasets),
        "config_tokenizer": asdict(config.config_tokenizer),
        "config_exp": asdict(config.config_exp),
        "config_train": asdict(config.config_train),
        "config_model": asdict(config.config_model),
        "device": config.device,
        "seq_len": config.seq_len,
    }

    with open(filename, "w") as f:
        json.dump(config_dict, f, indent=4, default=serialize)


def load_config_from_json(config) -> Config:
    """Loads a Config object from a JSON file."""
    metadata_file = config.config_exp.metadata_file
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            config_dict = json.load(f)

        config = Config()

        # Restore dataset config
        config.config_datasets.__dict__.update(config_dict["config_datasets"])

        # Restore tokenizer config
        board_size = tuple(config_dict["config_datasets"]["board_size"])
        config.config_tokenizer = ConfigTokenizer(board_size)
        config.config_tokenizer.__dict__.update(config_dict["config_tokenizer"])

        # Restore experiment config
        config.config_exp.__dict__.update(config_dict["config_exp"])

        config.config_exp.exp_folder = Path(config.config_exp.global_folder) / config.config_exp.exp_name
        config.config_exp.weights_folder = config.config_exp.exp_folder / '_weights'
        config.config_exp.images_folder = config.config_exp.exp_folder / '_images'
        config.config_exp.trainer_folder = config.config_exp.exp_folder / '_trainer'
        config.config_exp.metadata_file = config.config_exp.exp_folder / 'metadata.json'
        config.config_exp.tensorboard_folder = config.config_exp.trainer_folder / 'tensorboard_logs'
        config.config_exp.logs_txt_file = config.config_exp.trainer_folder / 'logs.txt'

        # Restore training config
        config.config_train.__dict__.update(config_dict["config_train"])

        # Restore model config
        vocab_size = config.config_tokenizer.vocab_size
        config.config_model = ConfigModel(vocab_size, board_size)
        config.config_model.__dict__.update(config_dict["config_model"])

        # Restore device and seq_len
        config.seq_len = config_dict["seq_len"]

        return config
    else:
        print(f"File {metadata_file} does not exist!")
        raise FileNotFoundError(f"{metadata_file} not found!")


def load_pretrained_model(config: Config, epoch, model, optimizer):
    config = load_config_from_json(config)
    model_filename = config.config_exp.weights_folder / f"{config.config_exp.exp_name}_{epoch}.pt"

    print(f"loading_model: {model_filename}")

    state = torch.load(model_filename, map_location=torch.device(config.device))
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = config.config_train.lr

    initial_epoch = state['epoch'] + 1
    global_step = state['global_step']
    train_loss = state['train_loss']
    test_loss = state['test_loss']

    return initial_epoch, global_step, train_loss, test_loss


##############


class API:

    @staticmethod
    def dataloaders(config: Config, tokenizer):
        len_dataset_train = config.config_datasets.len_dataset_train
        len_dataset_test = config.config_datasets.len_dataset_test
        train_len_data = config.config_datasets.train_len_data
        test_len_data = config.config_datasets.test_len_data
        train_datafile_path = config.config_datasets.train_datafile_path
        test_datafile_path = config.config_datasets.test_datafile_path
        batch_size = config.config_train.batch_size
        return Dataloaders(tokenizer, len_dataset_train, len_dataset_test, train_len_data, test_len_data,
                           train_datafile_path, test_datafile_path, batch_size)

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
        datafile_path = config.config_datasets.test_datafile_path
        data_len = config.config_datasets.test_len_data
        return Infer(model, tokenizer, seq_len, device, datafile_path, data_len)
