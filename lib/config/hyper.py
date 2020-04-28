import json

from dataclasses import dataclass

@dataclass
class Hyper(object):
    def __init__(self, path: str):
        self.dataset: str
        self.model: str
        self.data_root: str
        self.raw_data_root: str
        self.train: str
        self.dev: str
        self.print_epoch: int
        self.activation: str
        self.optimizer: str
        self.epoch_num: int
        self.gpu: int
        self.train_batch: int
        self.eval_batch: int
        self.criterion: str

        self.__dict__ = json.load(open(path, 'r'))

    def __post_init__(self):
        pass

