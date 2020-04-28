import os
import time 
import json
import argparse
import torchvision.models as models
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from lib.preprocessings import Weapon_resnet_preprocessing
from lib.dataloaders import Selection_Dataset
from torch.utils.data import DataLoader
from lib.metrics import F1_resnet
from lib.config import Hyper
import numpy as np 
from lib.models.resnet import resnet101_normal

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='weapon_resnet',
                    help='experiments/exp_name.json')
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='preprocessing',
                    help='preprocessing|train|evaluation')
args = parser.parse_args()
classes = ['no', 'stick', 'gun','dao']





class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.model_dir = 'save_models'
        self.hyper = Hyper(os.path.join('experiments',
                                        self.exp_name + '.json'))

        self.gpu = self.hyper.gpu
        self.criterion = None
        self.preprocessor = None
        self.optimizer = None
        self.model = None
        self.metrics = [F1_resnet(i) for i in range(4)]


    def _optimizer(self, name, model):
        m = {
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.5)
        }
        return m[name]
    

    def _init_model(self):
        if self.hyper.criterion == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        model = resnet101_normal()
        self.model = model.cuda(self.gpu)

    def preprocessing(self):
        if self.exp_name == 'weapon_resnet':
            self.preprocessor = Weapon_resnet_preprocessing(self.hyper)
        self.preprocessor.split_data()
        
    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.hyper.evaluation_epoch)
            self.evaluation()
        else:
            raise ValueError('invalid mode')
    def load_model(self, epoch: int):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,
                            self.exp_name + '_' + str(epoch))))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))


    def evaluation(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = DataLoader(dev_set, batch_size=self.hyper.train_batch, shuffle=True)
        for m in self.metrics:
            m.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                inputs, labels = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda(self.gpu)
                    labels = labels.cuda(self.gpu)
                output = self.model(inputs)
                pred1 = output.data.max(1, keepdim=True)
                pred = pred1[1].squeeze(1)
                for metric in self.metrics:
                    metric(np.array(labels.cpu().numpy(),dtype=int).tolist(), np.array(pred.cpu().numpy(),dtype=int).tolist())

            results = [mtc.get_metric() for mtc in self.metrics]
            for result in results:
                print('result-> ' +  ', '.join([
                    "%s: %.4f" % (name[0], value)
                    for name, value in result.items() if not name.startswith("_")
                ]))


    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = DataLoader(train_set, batch_size=self.hyper.train_batch, shuffle=True)
        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:

                self.optimizer.zero_grad()
                inputs, labels = sample
                if torch.cuda.is_available():
                    inputs = inputs.cuda(self.gpu)
                    labels = labels.cuda(self.gpu)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                loss_item = loss.item()
                pbar.set_description(' epoch'+ str(epoch)+': Loss=' + str(loss_item))

            self.save_model(epoch)

            if epoch % self.hyper.print_epoch == 0:
                self.evaluation()




if __name__ == "__main__":
    config = Runner(exp_name=args.exp_name)
    config.run(mode=args.mode)

