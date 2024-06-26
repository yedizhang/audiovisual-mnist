import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def display(X):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    img = [x.cpu().detach().numpy() for x in X]
    f = plt.figure()
    for i in range(len(X)):
        f.add_subplot(1, len(X), i+1)
        plt.imshow(img[i], cmap='gray')
    plt.show()


def bernoulli(prob, shape):
    a = torch.bernoulli(prob*torch.ones(shape[0])).to(device)
    return a[:, None, None, None].expand(shape)


def corrupt(xA, xB, feed_A=0.8, feed_AB=0.2):
    A, AB = bernoulli(feed_A, xA.shape), bernoulli(feed_AB, xA.shape)
    n = torch.normal(0, 0.15, size=xA.shape).to(device)
    A_only = A * (1-AB)
    B_only = 1 - A - AB
    xA = xA * (AB + A_only) + (0.1*xA+n)* (1 - AB - A_only)
    xB = xB * (AB + B_only) + (0.1*xB+n)* (1 - AB - B_only)
    # display([xA[0,0,:,:], xB[0,0,:,:]])
    return xA, xB


def set_data(xA, xB, unimodal=None, flatten=False):
    if unimodal == 'visual':  # Feed blank img if unimodal
        xA = xA
        xB = torch.zeros(xB.shape).to(device)
    elif unimodal == 'audio':
        xA = torch.zeros(xA.shape).to(device)
        xB = xB
    if flatten:
        return torch.flatten(xA, 1), torch.flatten(xB, 1)
    else:
        return xA, xB


class FCN(nn.Module):
    def __init__(self, depth, fuse_depth, hid_dim=500):
        super(FCN, self).__init__()
        self.depth = depth
        self.fuse_depth = fuse_depth
        self.layers = nn.ModuleDict()

        for i in range(1, fuse_depth):  # iterate 1, ..., fuse_depth-1
            if i == 1:
                self.layers['encodeA_'+str(i)] = torch.nn.Linear(784, hid_dim)
                self.layers['encodeB_'+str(i)] = torch.nn.Linear(784, hid_dim)
            else:
                self.layers['encodeA_'+str(i)] = torch.nn.Linear(hid_dim, hid_dim)
                self.layers['encodeB_'+str(i)] = torch.nn.Linear(hid_dim, hid_dim)

        if fuse_depth == 1:  # early fusion
            self.layers['fuse'] = torch.nn.Linear(1568, hid_dim)
        elif fuse_depth == depth:  # latest fusion
            self.layers['fuse'] = torch.nn.Linear(hid_dim*2, 10)
        else:
            self.layers['fuse'] = torch.nn.Linear(hid_dim*2, hid_dim)
        
        for i in range(fuse_depth, depth):  # iterate fuse_depth, ..., depth-1
            if i != depth-1:
                self.layers['decode_'+str(i)] = torch.nn.Linear(hid_dim, hid_dim)
            else:
                self.layers['decode_'+str(i)] = torch.nn.Linear(hid_dim, 10)


    def forward(self, xA, xB, unimodal=None):
        xA, xB = set_data(xA, xB, unimodal, True)

        for i in range(1, self.fuse_depth):
            xA = self.layers['encodeA_'+str(i)](xA)
            xB = self.layers['encodeB_'+str(i)](xB)
            xA, xB = F.relu(xA), F.relu(xB)
        x = torch.cat((xA, xB), -1)
        x = self.layers['fuse'](x)
        for i in range(self.fuse_depth, self.depth):
            x = F.relu(x)
            x = self.layers['decode_'+str(i)](x)

        return F.log_softmax(x, dim=1)
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                torch.nn.init.normal_(m.bias, std=0.1)


class CNN(nn.Module):
    def __init__(self, depth=4, fuse_depth=3, hid=32):
        super(CNN, self).__init__()
        self.depth = depth
        self.fuse_depth = fuse_depth
        self.layers = nn.ModuleDict()

        for i in range(1, fuse_depth):  # iterate 1, ..., fuse_depth-1
            if i == 1:
                self.layers['encodeA_'+str(i)] = nn.Conv2d(1, hid, 3, 1)
                self.layers['encodeB_'+str(i)] = nn.Conv2d(1, hid, 3, 1)
            else:
                self.layers['encodeA_'+str(i)] = nn.Conv2d(hid, hid, 3, 1)
                self.layers['encodeB_'+str(i)] = nn.Conv2d(hid, hid, 3, 1)

        if fuse_depth == 1:  # early fusion
            self.layers['fuse'] = nn.Conv2d(2, hid, 3, 1)
        else:
            self.layers['fuse'] = nn.Conv2d(hid*2, hid, 3, 1)

        for i in range(fuse_depth, depth):
            self.layers['decode_'+str(i)] = nn.Conv2d(hid, hid, 3, 1)

        self.layers['fc'] = nn.Linear(3200, 10)


    def forward(self, xA, xB, unimodal=None):
        xA, xB = set_data(xA, xB, unimodal)

        for i in range(1, self.fuse_depth):
            xA = self.layers['encodeA_'+str(i)](xA)
            xB = self.layers['encodeB_'+str(i)](xB)
            xA, xB = F.relu(xA), F.relu(xB)
        x = torch.cat((xA, xB), 1)  # concat along channel
        x = F.relu(self.layers['fuse'](x))
        for i in range(self.fuse_depth, self.depth):
            x = self.layers['decode_'+str(i)](x)
            x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.layers['fc'](x)

        return F.log_softmax(x, dim=1)