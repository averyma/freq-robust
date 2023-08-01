import torch
import numpy as np

from torchvision import datasets, transforms
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct2, batch_idct2, getDCTmatrix
from torch.utils.data import DataLoader, TensorDataset

import ipdb

def load_SyntheticDataset(case = 1, input_d = 784,
                          mu = 5, std = 1, 
                          lambbda = 0.005, batchsize = 128):

    train_x, train_y = synthesizeData(case, input_d, 50000, mu, std, lambbda)
    test_x, test_y = synthesizeData(case, input_d, 10000, mu, std, lambbda)
        
    if case in [7,8,9,10,11,12,13,14]:
        train_dataset = TensorDataset(train_x, train_y)
        test_dataset = TensorDataset(test_x, test_y)
    else:
        train_dataset = TensorDataset(train_x.t(), train_y.t())
        test_dataset = TensorDataset(test_x.t(), test_y.t())
        
    train_dataset.classes = lambda:None
    test_dataset.classes = lambda:None
        
    if case in [1,2,3,7,8,9,10,11,12]:
        class_holder = ['0 - zero',
                         '1 - one']
    elif case in [4,5,6,13]:
        class_holder = ['0 - zero',
                         '1 - one',
                         '2 - two',
                         '3 - three',
                         '4 - four',
                         '5 - five',
                         '6 - six',
                         '7 - seven',
                         '8 - eight',
                         '9 - nine']
    else:
        raise ValueError("Check class holder!")
    
    train_loader = DataLoader(train_dataset, batch_size = batchsize, pin_memory = True, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batchsize, pin_memory = True, shuffle = True)
    
    train_loader.dataset.classes = class_holder
    test_loader.dataset.classes = class_holder
    
#     ipdb.set_trace()
        
    return train_loader, test_loader
    
def synthesizeData(case = 1, d = 10, batchsize = 128, mu = 1, std = 0.5, lambbda = 1):
    
    if case in [7,8,9,10,11,12] and d >500:
        raise ValueError("d is TOO DAMN HIGH!!!")
    
    
    if case == 1:
        x_tilde = torch.zeros(d, batchsize)
        half_bs = int(batchsize/2)
        y = torch.zeros(batchsize)
        x_tilde[0,:half_bs] = torch.normal(mean = mu, std = std, size = (1,half_bs))
        y[:half_bs] = 1
        x_tilde[0,half_bs:] = torch.normal(mean = -mu, std = std, size = (1,half_bs))
        y[half_bs:] = 0
        
    elif case == 2:
        x_tilde = torch.zeros(d, batchsize)
        half_bs = int(batchsize/2)
        y = torch.zeros(batchsize)
        x_tilde[:,:half_bs] = torch.normal(mean = mu, std = std, size = (d,half_bs))
        y[:half_bs] = 1

        x_tilde[:,half_bs:] = torch.normal(mean = -mu, std = std, size = (d,half_bs))
        y[half_bs:] = 0
        
        random_sign = (2*torch.randint(0,2,size=(d,batchsize))-1)
        random_sign[0,:] = 1
        x_tilde = x_tilde*random_sign
        
    elif case == 3:        
        x_tilde = torch.zeros(d, batchsize)
        half_bs = int(batchsize/2)
        y = torch.zeros(batchsize)
        decay = torch.exp(-lambbda*(torch.range(1, d)-1).view(d,1).expand(-1, half_bs))
        x_tilde[:,:half_bs] = decay * torch.normal(mean = mu, std = std, size = (d,half_bs))
        y[:half_bs] = 1
        x_tilde[:,half_bs:] = decay * torch.normal(mean = -mu, std = std, size = (d,half_bs))
        y[half_bs:] = 0
        
        random_sign = (2*torch.randint(0,2,size=(d,batchsize))-1)
        random_sign[0,:] = 1
        x_tilde = x_tilde*random_sign
        
    elif case == 4:
        x_tilde = torch.zeros(d, batchsize)
        class_bs = int(batchsize/10)
        y = torch.zeros(batchsize)
        
        for i in range(10):
            x_tilde[0, i*class_bs : (i+1)*class_bs] = torch.normal(mean = mu * (i+1), std = std, size = (1, class_bs))
            y[i*class_bs : (i+1)*class_bs] = i
    
    elif case == 5:
        x_tilde = torch.zeros(d, batchsize)
        class_bs = int(batchsize/10)
        y = torch.zeros(batchsize)
        
        for i in range(10):
            x_tilde[:, i*class_bs : (i+1)*class_bs] = torch.normal(mean = mu * (i+1), std = std, size = (d, class_bs))
            y[i*class_bs : (i+1)*class_bs] = i
            
    elif case == 6:
        x_tilde = torch.zeros(d, batchsize)
        class_bs = int(batchsize/10)
        y = torch.zeros(batchsize)
        decay = torch.exp(-lambbda*(torch.range(1, d)-1).view(d,1).expand(-1, class_bs))
        
        for i in range(10):
            x_tilde[:, i*class_bs : (i+1)*class_bs] = decay * torch.normal(mean = mu * (i+1), std = std, size = (d, class_bs))
            y[i*class_bs : (i+1)*class_bs] = i
            
    elif case == 7:
        x_tilde = torch.zeros(batchsize,1,d,d)
        half_bs = int(batchsize/2)
        y = torch.zeros(batchsize)
        
        x_tilde[:half_bs,0,0,0] = torch.normal(mean = mu, std = std, size = (1, half_bs))
        y[:half_bs] = 1
        x_tilde[half_bs:,0,0,0] = torch.normal(mean = -mu, std = std, size = (1, half_bs))
        y[half_bs:] = 0
        
    elif case == 8:
        x_tilde = torch.zeros(batchsize,1,d,d)
        half_bs = int(batchsize/2)
        y = torch.zeros(batchsize)
#         ipdb.set_trace()
        x_tilde[:half_bs,0,:,:] = torch.normal(mean = mu, std = std, size = (half_bs,d,d))
        y[:half_bs] = 1
        x_tilde[half_bs:,0,:,:] = torch.normal(mean = -mu, std = std, size = (half_bs,d,d))
        y[half_bs:] = 0
        
        random_sign = (2*torch.randint(0,2,size=(batchsize,1,d,d))-1)
        random_sign[:,:,0,0] = 1
        x_tilde = x_tilde*random_sign
        
    elif case == 9:
        x_tilde = torch.zeros(batchsize,1,d,d)
        half_bs = int(batchsize/2)
        y = torch.zeros(batchsize)
        
        decay_ij = torch.zeros(batchsize,1,d,d)
        for i in range(d):
            for j in range(d):
                decay_ij[:,0,i,j] = i+j
                        
        decay = torch.exp(-lambbda*decay_ij)
        
        x_tilde[:half_bs,0,:,:] = torch.normal(mean = mu, std = std, size = (half_bs,d,d))
        y[:half_bs] = 1
        x_tilde[half_bs:,0,:,:] = torch.normal(mean = -mu, std = std, size = (half_bs,d,d))
        y[half_bs:] = 0
        
        random_sign = (2*torch.randint(0,2,size=(batchsize,1,d,d))-1)
        random_sign[:,:,0,0] = 1
        x_tilde = decay*x_tilde*random_sign
        
    elif case == 10:
        cor_region = 10
        x_tilde = torch.zeros(batchsize,1,d,d)
        half_bs = int(batchsize/2)
        y = torch.zeros(batchsize)
#         ipdb.set_trace()
        x_tilde[:half_bs,0,:cor_region,:cor_region] = torch.normal(mean = mu, std = std, size = (half_bs,cor_region,cor_region))
        y[:half_bs] = 1
        x_tilde[half_bs:,0,:cor_region,:cor_region] = torch.normal(mean = -mu, std = std, size = (half_bs,cor_region,cor_region))
        y[half_bs:] = 0
        
    elif case == 11:
        cor_region = 10
        x_tilde = torch.zeros(batchsize,1,d,d)
        half_bs = int(batchsize/2)
        y = torch.zeros(batchsize)
#         ipdb.set_trace()
        x_tilde[:half_bs,0,:,:] = torch.normal(mean = mu, std = std, size = (half_bs,d,d))
        y[:half_bs] = 1
        x_tilde[half_bs:,0,:,:] = torch.normal(mean = -mu, std = std, size = (half_bs,d,d))
        y[half_bs:] = 0
        
        random_sign = (2*torch.randint(0,2,size=(batchsize,1,d,d))-1)
        random_sign[:,:,:cor_region,:cor_region] = 1
        x_tilde = x_tilde*random_sign
        
    elif case == 12:
        cor_region = 10
        x_tilde = torch.zeros(batchsize,1,d,d)
        half_bs = int(batchsize/2)
        y = torch.zeros(batchsize)
        
        decay_ij = torch.zeros(batchsize,1,d,d)
        for i in range(d):
            for j in range(d):
                if i >(cor_region-1) or j >(cor_region-1):
                    decay_ij[:,0,i,j] = i+j
                        
        decay = torch.exp(-lambbda*decay_ij)
        
        x_tilde[:half_bs,0,:,:] = torch.normal(mean = mu, std = std, size = (half_bs,d,d))
        y[:half_bs] = 1
        x_tilde[half_bs:,0,:,:] = torch.normal(mean = -mu, std = std, size = (half_bs,d,d))
        y[half_bs:] = 0
        
        random_sign = (2*torch.randint(0,2,size=(batchsize,1,d,d))-1)
        random_sign[:,:,:cor_region,:cor_region] = 1
        x_tilde = decay*x_tilde*random_sign

        
    elif case ==13:        
        cor_region = 10
        x_tilde = torch.zeros(batchsize,1,d,d)
        class_bs = int(batchsize/10)
        y = torch.zeros(batchsize)
        
        for i in range(10):
            x_tilde[i*class_bs : (i+1)*class_bs,0,:cor_region,:cor_region] = torch.normal(mean = mu * (i+1), std = std, size = (class_bs,cor_region,cor_region))
            y[i*class_bs : (i+1)*class_bs] = i
    else:
        raise NotImplementedError("CASE NOT IMPLEMENTED")
    
    if case in [7,8,9,10,11,12,13]:
        dct_matrix = getDCTmatrix(d)
        x = batch_idct2(x_tilde,dct_matrix).view(batchsize,1,d,d)
    else:
        x = idct(x_tilde)
    
    return x, y
