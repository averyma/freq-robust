import random
import os
import operator as op
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
# from models import c11, ResNet18, ResNet8, c12, two_layer_conv, c2, linear_conv, c13, c14, c13_basic, ResNet8_basic, ResNet18_basic, two_layer_flatten, linear_flatten
from models import c2, two_layer_flatten, linear_flatten, PreActResNet18, Wide_ResNet, VGG, DenseNet121, ResNeXt29_4x24d, PreActResNet50
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import grad
    
def seed_everything(manual_seed):
    # set benchmark to False for EXACT reproducibility
    # when benchmark is true, cudnn will run some tests at
    # the beginning which determine which cudnn kernels are
    # optimal for opertions
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model(argu):

    if argu.dataset == 'cifar100':
        num_classes=100
    elif argu.dataset in ['cifar10', 'svhn', 'mnist', 'fashionmnist', 'imagenette','fmd']:
        num_classes=10
    elif argu.dataset == 'tiny':
        num_classes=200
    elif argu.dataset == 'dtd':
        num_classes=47
    elif argu.dataset in ['caltech', 'food']:
        num_classes=101
    elif argu.dataset == 'flowers':
        num_classes=102
    elif argu.dataset == 'cars':
        num_classes=196
    else:
        raise ValueError('dataset unspecified!')

    if argu.arch == "c2":
        model = c2()
    # elif argu.arch == "c11":
        # model = c11()
    # elif argu.arch == "resnet18":
        # model = ResNet18()
    # elif argu.arch == "resnet8":
        # model = ResNet8()
    # elif argu.arch == "resnet8_basic":
        # model = ResNet8_basic()
    # elif argu.arch == "resnet18_basic":
        # model = ResNet18_basic()
    # elif argu.arch == "c12":
        # model = c12()
    # elif argu.arch == "c13":
        # model = c13()
    # elif argu.arch == "c13_basic":
        # model = c13_basic()
    # elif argu.arch == "two_layer_conv":
        # model = two_layer_conv(argu.input_d, argu.hidden_d, argu.output_d, argu.activation, argu.weight_init)
    elif argu.arch == "two_layer_flatten":
        model = two_layer_flatten(argu.input_d, argu.hidden_d, argu.output_d, argu.activation, argu.bias)
    # elif argu.arch == "linear_conv":
        # model = linear_conv(argu.input_d, argu.output_d, argu.weight_init, argu.bias)
    elif argu.arch == "linear_flatten":
        model = linear_flatten()
    elif argu.arch == 'vgg19':
        model = VGG('VGG19', argu.dataset, num_classes, argu.input_normalization, argu.enable_batchnorm)
    elif argu.arch == 'preactresnet18':
        model = PreActResNet18(argu.dataset, num_classes, argu.input_normalization, argu.enable_batchnorm)
    elif argu.arch == 'preactresnet50':
        # model = PreActResNet18(argu.dataset, num_classes, argu.input_normalization, argu.enable_batchnorm)
        model = PreActResNet50(argu.dataset, num_classes, argu.input_normalization, argu.enable_batchnorm)
    # elif argu.arch == 'wrn16':
        # model = Wide_ResNet(16, 8, 0.3, num_classes, argu.input_normalization)
    # elif argu.arch == 'resnext29':
        # model = ResNeXt29_4x24d(num_classes, argu.input_normalization)
    # elif argu.arch == 'densenet121':
        # model = DenseNet121(num_classes, argu.input_normalization)
    else:
        raise NotImplementedError("model not included")
    
    # if argu.pretrain:
        
        # model.load_state_dict(torch.load(argu.pretrain, map_location=device))
        # model.to(device)
        # print("\n ***  pretrain model loaded: "+ argu.pretrain + " *** \n")

    return model

def get_optim(model, argu):
    """
    recommended setup:
    SGD_step: initial lr:0.1, momentum: 0.9, weight_decay: 0.0002, miliestones: [100, 150]
    Adam_step: initial lr:0.1, milestones: [80,120,160,180]
    others: constant lr at 0.001 should be sufficient
    """
    
    if "sgd" in argu.optim:
        opt = optim.SGD(model.parameters(), lr = argu.lr, momentum = 0, weight_decay = argu.weight_decay)
    elif "sgd-m" in argu.optim:
        opt = optim.SGD(model.parameters(), lr = argu.lr, momentum = argu.momentum, weight_decay = argu.weight_decay)
    elif "adam" in argu.optim:
        opt = optim.Adam(model.parameters(), lr = argu.lr, betas=(argu.adam_beta1, argu.adam_beta2))
    elif "rmsprop" in argu.optim:
        opt = optim.RMSprop(model.parameters(), lr=argu.lr, alpha=argu.rmsp_alpha, weight_decay=argu.weight_decay, momentum=argu.momentum, centered=False)
        # opt = optim.RMSprop(model.parameters(), lr=argu.lr, alpha=argu.rmsp_alpha, weight_decay=argu.weight_decay, momentum=argu.momentum, centered=False)
    elif "adagrad" in argu.optim:
        opt = optim.Adagrad(model.parameters(), lr=argu.lr, lr_decay=0, weight_decay=argu.weight_decay, initial_accumulator_value=0, eps=1e-10)

    # check if milestone is an empty array
    if argu.lr_update == "multistep":
        _milestones = [argu.epoch/ 2, argu.epoch * 3 / 4]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=_milestones, gamma=0.1)
    elif argu.lr_update == "fixed":
        lr_scheduler = False

    return opt, lr_scheduler

class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]

def ep2itr(epoch, loader):
    try:
        data_len = loader.dataset.data.shape[0]
    except AttributeError:
        data_len = loader.dataset.tensors[0].shape[0]
    batch_size = loader.batch_size
    iteration = epoch * np.ceil(data_len/batch_size)
    return iteration
