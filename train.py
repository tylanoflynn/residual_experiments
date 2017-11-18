import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from models import resnet
from utils import get_train_valid_loader

opt = {
    'dataset': None,
    'num_classes': 10,
    'augment': True,
    'validation': False,
    'save': 'logs',
    'batchSize': 128,
    'learningRate': 0.1,
    'learningRateDecayRatio': 0.2,
    'weightDecay': 0.0005,
    'dampening': 0,
    'momentum': 0.9,
    'max_epoch': 200,
    'optimMethod': 'sgd',
    'depth': 28,
    'shortcutType': 'A',
    'nesterov': True,
    'dropout': 0.0,
    'blockwise': False,
    'hflip': True,
    'randomCrop': 4,
    'imageSize': 32,
    'randomcrop_type': 0,
    'widen_factor': 10,
    'nGPU': 1,
    'data_type': 'torch.CudaTensor',
    'seed': 444
    }

torch.manual_seed(opt['seed'])

trainloader, validloader = get_train_valid_loader(root='./data',
                            batch_size=128, augment=True, random_seed=opt['seed'],
                            num_workers=4, pin_memory=True)

Normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    Normalize])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if opt['validation']:
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    optim_dropout = {'rate': 0.0, 'accuracy': 0.0}
    for dropout in [0.02, 0.03, 0.04]:
        opt['dropout'] = dropout
        net = resnet.buildModel(opt = opt)
        if torch.cuda.is_available():
            net.cuda()
            print('New net moved to CUDA')
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay = 0.0005,
                              nesterov = True)
        for epoch in range(100):

            if epoch == 30:
                adjust_learning_rate(optimizer, 0.02)
            elif epoch == 60:
                adjust_learning_rate(optimizer, 0.004)
            elif epoch == 90:
                adjust_learning_rate(optimizer, 0.0008)
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in a variable
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 100 == 99:
                    grad_of_params = {}
                    print ('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

        correct = 0
        total = 0

        net.eval()

        for data in validloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        accuracy = 100 * correct / float(total)

        print('''Accuracy of the network with {} dropout rate {} on the {} validation images:
        {:2.2f}%'''.format(opt['blockwise'] and 'block' or '', opt['dropout'],
                         total, accuracy))

        if accuracy > optim_dropout['accuracy']:
            optim_dropout['rate'] = dropout
            optim_dropout['accuracy'] = accuracy

        del net
        del optimizer

    print('The optimal validation dropout was {}'.format(optim_dropout['rate']))


else:
    
    net = resnet.buildModel(opt = opt)
    if torch.cuda.is_available():
        net.cuda()
        print('CUDA IS AVAILABLE')
        print('TRAIN NET MOVED TO CUDA')
    else:
        print('CUDA NOT AVAILABLE')

    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay = 0.0005,
                          nesterov = True)

    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    net.train()

    for epoch in range(200):

        if epoch == 60:
            adjust_learning_rate(optimizer, 0.02)
        elif epoch == 120:
            adjust_learning_rate(optimizer, 0.004)
        elif epoch == 160:
            adjust_learning_rate(optimizer, 0.0008)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data

            # wrap them in a variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:
                grad_of_params = {}
                print ('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0

    net.eval()

    for data in testloader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the {} test images: {:2.2f}%'.format(
        total, 100 * correct / float(total)))
