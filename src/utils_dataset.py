import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.transforms.functional import InterpolationMode
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct2, getDCTmatrix
import ipdb
from typing import Any, Callable, List, Optional, Union, Tuple
import os
from PIL import Image
# from data.Caltech101.caltech_dataset import Caltech
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Subset

data_dir = '/scratch/ssd001/home/ama/workspace/data/'
    
def load_dataset(dataset, _batch_size = 128, standard_DA = True, freq=False, test_shuffle=False):

    if dataset in ["mnist"]:
        train_loader, test_loader = load_MNIST(_batch_size, freq, test_shuffle)
    elif dataset in ["binarymnist"]:
        train_loader, test_loader = load_binaryMNIST(_batch_size)
    elif dataset in ["fashionmnist"]:
        train_loader, test_loader = load_FashionMNIST(_batch_size, freq, test_shuffle)
    elif dataset in ["binarycifar10"]:
        train_loader, test_loader = load_binaryCIFAR(_batch_size)
    elif dataset in ["cifar10"]:
        train_loader, test_loader = load_CIFAR10(_batch_size, standard_DA, test_shuffle)
    elif dataset in ["cifar100"]:
        train_loader, test_loader = load_CIFAR100(_batch_size, standard_DA, test_shuffle)
    elif dataset in ["svhn"]:
        train_loader, test_loader = load_SVHN(_batch_size, standard_DA, test_shuffle)
    elif dataset in ["graycifar10"]:
        train_loader, test_loader = load_grayCIFAR(_batch_size)
    elif dataset in ["imagenette"]:
        train_loader, test_loader = load_imagenette(_batch_size, standard_DA, test_shuffle)
    elif dataset in ["dtd"]:
        train_loader, test_loader = load_dtd(_batch_size, standard_DA)
    elif dataset in ["tiny"]:
        train_loader, test_loader = load_tiny(_batch_size, standard_DA)
    elif dataset in ["caltech"]:
        train_loader, test_loader = load_caltech(_batch_size, standard_DA, test_shuffle)
    elif dataset in ["fmd"]:
        train_loader, test_loader = load_fmd(_batch_size, standard_DA)
    elif dataset in ["flowers"]:
        train_loader, test_loader = load_flowers(_batch_size, standard_DA)
    elif dataset in ["food"]:
        train_loader, test_loader = load_food(_batch_size, standard_DA)
    elif dataset in ["cars"]:
        train_loader, test_loader = load_cars(_batch_size, standard_DA)
    elif dataset in ["country"]:
        train_loader, test_loader = load_country(_batch_size, standard_DA)
    else:
        raise NotImplementedError("Dataset not included")
        
    return train_loader, test_loader
    
def load_binaryMNIST(batch_size, target = [2,7]):
    # load MNIST data set into data loader
    assert len(target)==2 and isinstance(target[0], int) and isinstance(target[1], int) and target[0] != target[1]
    
    if target[1]<target[0]:
        temp = target[1]
        target[1] = target[0]
        target[0] = temp
        del temp
        assert target[1]>target[0]
    
    mnist_train = datasets.MNIST(data_dir, train=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(data_dir, train=False,  transform=transforms.ToTensor())

    idx_3, idx_7 = mnist_train.targets == target[0], mnist_train.targets == target[1]
    idx_train = idx_3 | idx_7

    idx_3, idx_7 = mnist_test.targets == target[0], mnist_test.targets == target[1]
    idx_test = idx_3 | idx_7
    
    mnist_train.data = mnist_train.data[idx_train]
    mnist_test.data = mnist_test.data[idx_test]
        
    
    mnist_train.targets = mnist_train.targets[idx_train]
    mnist_test.targets = mnist_test.targets[idx_test]

    mnist_train.targets = ((mnist_train.targets - target[0])/(target[1]-target[0])).float()
    mnist_test.targets = ((mnist_test.targets - target[0])/(target[1]-target[0])).float()

    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=True)
    
    class_holder = ['0 - zero',
                    '1 - one']
    train_loader.dataset.classes = class_holder
    test_loader.dataset.classes = class_holder
    
    return train_loader, test_loader

def load_MNIST(batch_size, freq= False, test_shuffle=False):
    # load MNIST data set into data loader

    if freq == True:
        mnist_train = datasets.MNIST(data_dir, train=True, transform=None)
        mnist_test = datasets.MNIST(data_dir, train=False,  transform=None)
        dct_matrix = getDCTmatrix(28)
        mnist_train.data = mnist_train.data.to(torch.float)/255
        mnist_train.data = batch_dct2(mnist_train.data, dct_matrix).unsqueeze(1)
        mnist_test.data = mnist_test.data.to(torch.float)/255
        mnist_test.data = batch_dct2(mnist_test.data, dct_matrix).unsqueeze(1)
        
        train_dataset = TensorDataset(mnist_train.data, mnist_train.targets)
        test_dataset = TensorDataset(mnist_test.data, mnist_test.targets)
        
    else:
        train_dataset = datasets.MNIST(data_dir, train=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(data_dir, train=False,  transform=transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=test_shuffle)
    
    
    return train_loader, test_loader

def load_FashionMNIST(batch_size, freq=False, test_shuffle=False):
    # load MNIST data set into data loader

    if freq == True:
        fmnist_train = datasets.FashionMNIST(data_dir, train=True, transform=None)
        fmnist_test = datasets.FashionMNIST(data_dir, train=False,  transform=None)
        dct_matrix = getDCTmatrix(28)
        fmnist_train.data = fmnist_train.data.to(torch.float)/255
        fmnist_train.data = batch_dct2(fmnist_train.data, dct_matrix).unsqueeze(1)
        fmnist_test.data = fmnist_test.data.to(torch.float)/255
        fmnist_test.data = batch_dct2(fmnist_test.data, dct_matrix).unsqueeze(1)
        
        train_dataset = TensorDataset(fmnist_train.data, fmnist_train.targets)
        test_dataset = TensorDataset(fmnist_test.data, fmnist_test.targets)
        
    else:
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download = True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(data_dir, train=False,  download = True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=test_shuffle)
    
    return train_loader, test_loader

def load_grayCIFAR(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])

    data_train = datasets.CIFAR10(data_dir, train=True, download = True, transform=transform_train)
    data_test = datasets.CIFAR10(data_dir, train=False, download = True, transform=transform_test)
    
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=True)
    
    class_holder = ['0 - zero',
                    '1 - one']
    train_loader.dataset.classes = class_holder
    test_loader.dataset.classes = class_holder

    return train_loader, test_loader

def load_binaryCIFAR(batch_size, target1=1, target2=5):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])
    
    # load CIFAR data set into data loader
    data_train = datasets.CIFAR10(data_dir, train=True, download = True, transform=transform_train)
    data_test = datasets.CIFAR10(data_dir, train=False, download = True, transform=transform_test)
    
     
    idx_3, idx_7 = torch.tensor(data_train.targets) == target1, torch.tensor(data_train.targets) == target2
    idx_train = (idx_3 | idx_7).tolist()

    idx_3, idx_7 = torch.tensor(data_test.targets) == target1, torch.tensor(data_test.targets) == target2
    idx_test = (idx_3 | idx_7).tolist()
    
    data_train.data = data_train.data[idx_train,:,:,:]
    data_test.data = data_test.data[idx_test,:,:,:]
        
    data_train.targets = torch.tensor(data_train.targets)[idx_train].tolist()
    data_test.targets = torch.tensor(data_test.targets)[idx_test].tolist()

    # label 0: 3, label 1: 7
    idx_train = (torch.tensor(data_train.targets) == target1)
    data_train.targets = idx_train.float().tolist()
    idx_test = (torch.tensor(data_test.targets) == target1)
    data_test.targets = idx_test.float().tolist()
    
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=True)
    
    class_holder = ['0 - zero',
                    '1 - one']
    train_loader.dataset.classes = class_holder
    test_loader.dataset.classes = class_holder
    
    return train_loader, test_loader

def load_CIFAR10(batch_size, standard_DA = False, test_shuffle=False):

    # mean = [0.49139968, 0.48215841, 0.44653091]
    # std = [0.24703223, 0.24348513, 0.26158784]

    if standard_DA:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding = 2),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.ToTensor()])

    data_train = datasets.CIFAR10(data_dir, train=True, download = True, transform=transform_train)
    data_test = datasets.CIFAR10(data_dir, train=False, download = True, transform=transform_test)

    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=test_shuffle)

    return train_loader, test_loader

def load_CIFAR100(batch_size, standard_DA = False, test_shuffle=False):

    # mean = [0.50707516, 0.48654887, 0.44091784]
    # std = [0.26733429, 0.25643846, 0.27615047]

    if standard_DA:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding = 2),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.ToTensor()])

    data_train = datasets.CIFAR100(data_dir, train=True, download = True, transform=transform_train)
    data_test = datasets.CIFAR100(data_dir, train=False, download = True, transform=transform_test)

    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=test_shuffle)

    return train_loader, test_loader

def load_SVHN(batch_size, standard_DA = False, test_shuffle=False):

    # mean = [0.4376821, 0.4437697, 0.47280442]
    # std = [0.19803012, 0.20101562, 0.19703614]

    if standard_DA:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 2),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.ToTensor()])

    data_train = datasets.SVHN(data_dir+"/SVHN", split='train', download = True, transform=transform_train)
    data_test = datasets.SVHN(data_dir+"/SVHN", split='test', download = True, transform=transform_test)
    
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=test_shuffle)

    return train_loader, test_loader

def load_imagenette(batch_size, standard_DA = False, test_shuffle=False):

    if standard_DA:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])

    train_dataset = datasets.ImageFolder(root=data_dir+'/imagenette2/train',
                                               transform=transform_train)
    test_dataset = datasets.ImageFolder(root=data_dir+'/imagenette2/val',
                                               transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size, shuffle=test_shuffle,
                                                num_workers=2)
    return train_loader, test_loader

def load_dtd(batch_size, standard_DA=False):

    full_dataset = datasets.ImageFolder(root=data_dir+'/dtd/dtd/images')
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [3947, 5640-3947], generator=torch.Generator().manual_seed(42))
    assert len(train_dataset)==3947 and len(test_dataset)==1693

    if standard_DA:
        #1
        train_dataset.dataset.transform = transforms.Compose([
            # transforms.Resize(72),
            # transforms.RandomResizedCrop(64),
#             transforms.Resize(36),
#             transforms.RandomResizedCrop(32),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        test_dataset.dataset.transform = transforms.Compose([
            # transforms.Resize(72),
            # transforms.CenterCrop(64),
#             transforms.Resize(36),
#             transforms.CenterCrop(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    else:
        train_dataset.dataset.transform = transforms.Compose([
            # transforms.Resize(72),
            # transforms.CenterCrop(64),
#             transforms.Resize(36),
#             transforms.CenterCrop(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        test_dataset.dataset.transform = transforms.Compose([
            # transforms.Resize(72),
            # transforms.CenterCrop(64),
#             transforms.Resize(36),
#             transforms.CenterCrop(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    # if standard_DA:
        # transform_train = transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor()
            # ])
        # transform_test = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # ])
    # else:
        # transform_train = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # ])
        # transform_test = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # ])    
    
    # _dir = '/scratch/ssd001/home/ama/workspace/ama-at-vector/freq-robust/data'
    # _part=2
    # print("partition: {}".format(_part))
    # train_dataset = datasets.DTD(_dir, split='train', transform = transform_train, download = False, partition=_part)
    # test_dataset = datasets.DTD(_dir, split='test', transform = transform_test, download = False, partition=_part)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)
    return train_loader, test_loader

def load_tiny(batch_size, standard_DA = False):
    if standard_DA:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([transforms.ToTensor()])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([transforms.ToTensor()])


    train_root = data_dir+'/tiny-imagenet-200/train'
    validation_root = data_dir+'/tiny-imagenet-200/val/images'
    train_data = datasets.ImageFolder(train_root, transform=train_transform)
    test_data = datasets.ImageFolder(validation_root, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True)
    return train_loader, test_loader


def load_caltech(batch_size, standard_DA = False, test_shuffle=False):
    if standard_DA:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])

    DATA_DIR = data_dir+'/Homework2-Caltech101/101_ObjectCategories'
    SPLIT_TRAIN = data_dir+'/Homework2-Caltech101/train.txt'
    SPLIT_TEST = data_dir+'/Homework2-Caltech101/test.txt'

    data_train = Caltech(DATA_DIR, split = SPLIT_TRAIN, transform=transform_train)
    data_test = Caltech(DATA_DIR, split = SPLIT_TEST, transform=transform_test)
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=test_shuffle)

    return train_loader, test_loader

def load_fmd(batch_size, standard_DA=False):
#     https://people.csail.mit.edu/lavanya/fmd.html

    full_dataset = datasets.ImageFolder(root=data_dir+'/fmd/image')
    train_dataset,test_dataset = torch.utils.data.random_split(full_dataset, [700, 300], generator=torch.Generator().manual_seed(42))

    if standard_DA:
        #1
        train_dataset.dataset.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        test_dataset.dataset.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    else:
        train_dataset.dataset.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        test_dataset.dataset.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=2)
    return train_loader, test_loader

def load_flowers(batch_size, standard_DA = False):
    if standard_DA:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])

    _data_train = datasets.Flowers102(data_dir, split = 'train', transform=transform_train, download = True)
    _data_val = datasets.Flowers102(data_dir, split = 'val', transform=transform_train, download = True)
    # data_train = torch.utils.data.ChainDataset([_data_train, _data_val])
    data_train = torch.utils.data.ConcatDataset([_data_train, _data_val])
    data_test = datasets.Flowers102(data_dir, split = 'test', transform = transform_test, download =True)
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader

def load_food(batch_size, standard_DA = False):
    if standard_DA:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])

    data_train = datasets.Food101(data_dir, split = 'train', transform=transform_train, download = True)
    data_test = datasets.Food101(data_dir, split = 'test', transform = transform_test, download =True)
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader

# def load_caltech256(batch_size, standard_DA = False):
    # full_dataset = datasets.Caltech256('./data', download = True)
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [21424,9183], generator=torch.Generator().manual_seed(42))

    # if standard_DA:
        # train_dataset.dataset.transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor()
            # ])
        # test_dataset.dataset.transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # ])
    # else:
        # train_dataset.dataset.transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # ])
        # test_dataset.dataset.transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # ])

    # train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    # return train_loader, test_loader

def load_country(batch_size, standard_DA = False):
    if standard_DA:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])

    data_train = datasets.Country211(data_dir, split = 'train', transform=transform_train, download = True)
    data_test = datasets.Country211(data_dir, split = 'test', transform = transform_test, download =True)
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader

def load_cars(batch_size, standard_DA = False):
    if standard_DA:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])

    data_train = datasets.StanfordCars(data_dir, split = 'train', transform=transform_train, download = True)
    data_test = datasets.StanfordCars(data_dir, split = 'test', transform = transform_test, download =True)
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader

##################################################################################################
##################################################################################################

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        # Open file in read only mode and read all lines
        file = open(self.split, "r")
        lines = file.readlines()

        # Filter out the lines which start with 'BACKGROUND_Google' as asked in the homework
        self.elements = [i for i in lines if not i.startswith('BACKGROUND_Google')]

        # Delete BACKGROUND_Google class from dataset labels
        self.classes = sorted(os.listdir(os.path.join(self.root, "")))
        self.classes.remove("BACKGROUND_Google")


    def __getitem__(self, index):
        ''' 
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

#         img = Image.open(os.path.join(self.root, self.elements[index].rstrip()))
        img = pil_loader(os.path.join(self.root, self.elements[index].rstrip()))

        target = self.classes.index(self.elements[index].rstrip().split('/')[0])

        image, label = img, target # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        # Provides a way to get the length (number of elements) of the dataset
        length =  len(self.elements)
        return length
