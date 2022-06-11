from matplotlib import transforms
import numpy as np
import torch 
from torchvision import transforms, datasets
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F 


class Shift_Net(nn.Module):
    def __init__(self, pars):
        super(Shift_Net, self).__init__()
        ks=(5,5)
        ps=np.int32(5)
        self.mid_layer=256
        # Two successive convolutional layers.
        # Two pooling layers that come after convolutional layers.
        # Two dropout layers.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=ks[1],padding='same')
        self.pool1= nn.MaxPool2d(kernel_size=10,stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=ks[1],padding='same')
        self.drop2 = nn.Dropout2d(p=0.2)
        self.pool2=nn.MaxPool2d(kernel_size=5,stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=ks[1],padding='same')
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=3)
        self.drop_final=nn.Dropout(p=0.2)
        self.total_pars = 0

        self.first=True
        if self.first:
            self.forward(torch.zeros((1,)+pars.inp_dim))
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.001)
        self.criterion=nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.conv1(x)
         
        # Apply relu to a pooled conv1 layer.
        x = F.relu(self.pool1(x))
        # Apply relu to a pooled conv2 layer with a drop layer inbetween.
        x = self.drop2(F.relu(self.pool2(self.conv2(x))))
        x = F.relu(self.pool3(self.conv3(x)))
        if self.first:
            self.first=False
            self.inp=x.shape[1]*x.shape[2]*x.shape[3]
            # Compute dimension of output of x and setup a fully connected layer with that input dim 
            # pars.mid_layer output dim. Then setup final 3 node output layer.
            print('input dimension to fc1',self.inp)
            if self.mid_layer is not None:
                self.fc1 = nn.Linear(self.inp, self.mid_layer)
                self.fc_final = nn.Linear(self.mid_layer, 3)
            else:
                self.fc1=nn.Identity()
                self.fc_final = nn.Linear(self.inp, 3)
        x = x.reshape(-1, self.inp)
        x = self.fc1(x)
        x = self.fc_final(x)
        return x
    
    # Run the network on the data, compute the loss, compute the predictions and compute classification rate/
    def get_acc_and_loss(self, data, targ):
        output = self.forward(data)
        loss = self.criterion(output, targ)
        pred = torch.max(output,1)[1]
        #print(f'Prediction is {pred}')
        correct = torch.eq(pred,targ).sum()
        
        return loss,correct
    
    # Compute classification and loss and then do a gradient step on the loss.
    def run_grad(self,data,targ):
        self.optimizer.zero_grad()
        loss, correct=self.get_acc_and_loss(data,targ)
        loss.backward()
        self.optimizer.step()
        
        return loss, correct


# An object containing the relevant parameters for running the experiment.
class par(object):
    def __init__(self):
        self.batch_size=1000
        self.step_size=.001
        self.num_epochs=20
        self.numtrain=55000
        self.minimizer="Adam"
        self.data_set="mnist"
        self.model_name="model"
        self.dropout=0.
        self.dim=32
        self.pool_size=2
        self.kernel_size=5
        self.mid_layer=256
        self.use_gpu=False
        