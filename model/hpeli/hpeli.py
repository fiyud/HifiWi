import torchvision
import torch.nn as nn
import torch
import cv2
from torchvision.transforms import Resize
from model.hpeli.SK_network import SKConv, SKUnit
import time
import torch.nn.functional as F

class hpelinet(nn.Module):
    def __init__(self, num_keypoints, num_coor, subcarrier_num, num_person=1, dataset='mmfi-csi'):
        super(hpelinet, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_coor = num_coor
        self.subcarrier_num = subcarrier_num
        self.num_person = num_person
        self.dataset = dataset
        
        k1 = 4 # number branches of DyConv1
        k2 = 4 # number branches of DyConv2
        num_lay = 64 # numer hidden dim of DyConv1
        D = 64 # number hidden dim of BiLSTM
        N = 1 # number hidden layers of BiLSTm 
        R = 32 # Reduction Ratios
        T = 64 # Temperature
        
        if dataset == 'person-in-wifi-3d':
            self.regression = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 1), stride=(2, 1), padding=0),  # (128, 22, 1) -> (64, 10, 1)
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=0),  # (64, 10, 1) -> (32, 4, 1)
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 1), stride=(1, 1), padding=0),  # (32, 4, 1) -> (16, 2, 1)
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(640, self.num_keypoints * self.num_coor * self.num_person)  # Fully connected layer to get (n, 36)
            )
        elif dataset == 'wipose':
            self.regression = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 1), stride=(2, 1), padding=0),  # (128, 22, 1) -> (64, 10, 1)
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=0),  # (64, 10, 1) -> (32, 4, 1)
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 1), stride=(1, 1), padding=0),  # (32, 4, 1) -> (16, 2, 1)
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16 * 2, self.num_keypoints * self.num_coor)  # Fully connected layer to get (n, 36)
            )
        else:
            self.regression = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 1), stride=(2, 1), padding=0),  # (128, 22, 1) -> (64, 10, 1)
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=0),  # (64, 10, 1) -> (32, 4, 1)
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 1), stride=(1, 1), padding=0),  # (32, 4, 1) -> (16, 2, 1)
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128, self.num_keypoints*self.num_coor)  # Fully connected layer to get (n, 36)
            )

        self.skunit1 = SKUnit(in_features=3, mid_features=num_lay, out_features=num_lay, dim1 =self.subcarrier_num,dim2 = 10,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)
        self.skunit2 = SKUnit(in_features=num_lay, mid_features=num_lay*2, out_features=num_lay*2, dim1 = self.subcarrier_num//2, dim2 = 8,pool_dim = 'freq-chan', M=1, G=64, r=4, stride=1, L=32)

    def forward(self,x): #16,2,3,114,32
        batch = x.shape[0]
        time_start = time.time()

        m = torch.nn.AvgPool2d((2, 2))
        x = self.skunit1(x)
        x = m(x) # [32, 64, 57, 5]
        
        
        out1 = self.skunit2(x)
        out1 = m(out1) 
        fea = out1.mean(3).mean(2)

        # print(out1.size())
        x = self.regression(out1)
        if self.dataset == 'person-in-wifi-3d':
            x = x.reshape(batch,self.num_person,self.num_keypoints, self.num_coor)
        else:
            x = x.reshape(batch,self.num_keypoints, self.num_coor)
        
        
        time_end = time.time()
        time_sum = time_end - time_start
        return x, fea

def hpeli_weights_init(m):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.xavier_normal_(m.weight.data)
    #     nn.init.xavier_normal_(m.bias.data)
    # elif classname.find('BatchNorm2d') != -1:
    #     nn.init.normal_(m.weight.data, 1.0)
    #     nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)