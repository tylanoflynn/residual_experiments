import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Module
import numpy as np


class Identity(Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SkipTable(Module):

    def __init__(self, moduleList):
        super(SkipTable, self).__init__()
        assert(moduleList.__len__() == 2)
        self.op = moduleList[0]
        self.skip_op = moduleList[1]

    def forward(self, x):
        y = self.skip_op(x.clone())
        x = self.op(x)
        assert(y.size() == x.size())
        return x + y


class View(Module):

    def __init__(self, dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)

class BlockDrop(Module):

    def __init__(self, p):
        super(BlockDrop, self).__init__()
        self.prob = torch.Tensor([p]).cuda()

    def forward(self, x):
        if self.training:
            x = Variable(torch.bernoulli(
                1 - self.prob)) * x
            return x
        else:
            return x


def buildModel(opt):
    assert(opt and opt['depth'])
    assert(opt and opt['num_classes'])
    assert(opt and opt['widen_factor'])

    depth = opt['depth']

    blocks = []

    def blockBasic(nInputPlane, nOutputPlane, stride):
        convParams = [[3, stride, 1], [3, 1, 1]]
        nBottleneckPlane = nOutputPlane

        block = nn.Sequential()
        convs = nn.Sequential()

        for i,v in enumerate(convParams):
            if i == 0:
                module = (nInputPlane == nOutputPlane) and convs or block
                module.add_module('BatchNorm' + str(i), nn.BatchNorm2d(nInputPlane))
                module.add_module('ReLU' + str(i), nn.ReLU(True))
                if opt['dropout'] > 0 and opt['blockwise']:
                    convs.add_module('BlockDrop' + str(i), BlockDrop(opt['dropout']))
                convs.add_module('Conv' + str(i), nn.Conv2d(nInputPlane, nBottleneckPlane, *v))
            else:
                convs.add_module('BatchNorm' + str(i), nn.BatchNorm2d(nOutputPlane))
                convs.add_module('ReLU' + str(i), nn.ReLU(True))
                if opt['dropout'] > 0 and not opt['blockwise']:
                    convs.add_module('Dropout' + str(i), nn.Dropout(opt['dropout'], False))
                convs.add_module('Conv' + str(i),
                                 nn.Conv2d(nBottleneckPlane, nBottleneckPlane, *v))

        shortcut = (nInputPlane == nOutputPlane) and Identity() or nn.Conv2d(nInputPlane,
                                                                             nOutputPlane, 1,
                                                                             stride, 0)

        skipTable = SkipTable(nn.ModuleList([convs, shortcut]))
        block.add_module('SkipTable', skipTable)
        return block

    def layer(block, nInputPlane, nOutputPlane, count, stride):
        s = nn.Sequential()

        s.add_module('Block1', block(nInputPlane, nOutputPlane, stride))
        for i in range(count - 1):
            s.add_module('Block' + str(i + 2), block(nOutputPlane, nOutputPlane, 1))

        return s

    model = nn.Sequential()

    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    k = opt['widen_factor']
    nStages = torch.IntTensor([16, 16*k, 32*k, 64*k])

    model.add_module('Conv1', nn.Conv2d(3, nStages[0], 3, 1, 1))
    model.add_module('Layer1', layer(blockBasic, nStages[0], nStages[1], n, 1))
    model.add_module('Layer2', layer(blockBasic, nStages[1], nStages[2], n, 2))
    model.add_module('Layer3', layer(blockBasic, nStages[2], nStages[3], n, 2))
    model.add_module('LastBN', nn.BatchNorm2d(nStages[3]))
    model.add_module('lastReLU', nn.ReLU(True))
    model.add_module('AvgPool',nn.AvgPool2d(8))
    model.add_module('View', View([-1, nStages[3]]))
    model.add_module('FCLayer', nn.Linear(nStages[3], opt['num_classes']))

    return model
