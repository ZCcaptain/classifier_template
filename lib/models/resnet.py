import torch
import torchvision
import torch.nn as nn 
import torchvision.models as models



class resnet101_normal(nn.Module):
    def __init__(self, p=0.5, is_pretrain=True):
        super(resnet101_normal, self).__init__()
        self.feature = models.resnet101(pretrained=is_pretrain)
        num_ftrs = self.feature.fc.in_features
        self.feature.fc = nn.Linear(num_ftrs, num_ftrs)
        self.fc_dropout = nn.Dropout(p)
        self.fc = nn.Linear(num_ftrs, 4)
    def forward(self, x):
        x = self.fc_dropout(self.feature(x))
        out =  self.fc(x)
        return out

if __name__ == "__main__":
    model = models.resnet101(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    print(model)