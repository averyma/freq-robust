import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from src.utils_general import seed_everything
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct2, batch_idct2, getDCTmatrix,mask_radial,batch_dct2_3channel, batch_idct2_3channel, equal_dist_from_top_left

import scipy.fft
import numpy as np
import ipdb
from tqdm import trange

from src.evaluation import test_gaussian_LF_HF_v2
from src.utils_general import seed_everything

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def computeSensitivityMap(loader, model, eps, dim, numb_inputs, clip, device):
    sens_map = torch.zeros(dim,dim,device=device)

    with trange(dim*dim) as t:
        for i in range(dim):
            for j in range(dim):
                sens_map[i,j] = test_freq_sensitivity(loader, model, eps, i,j, dim, numb_inputs, clip, device)

                t.set_postfix(err = '{0:.2f}%'.format(sens_map[i,j]))
                t.update()
    return sens_map

def computeSensitivityMap_through_KLdiv(loader, model, eps, dim, numb_inputs, clip, device):
    sens_map = torch.zeros(dim,dim,device=device)

    with trange(dim*dim) as t:
        for i in range(dim):
            for j in range(dim):
                sens_map[i,j] = test_freq_sensitivity_through_KLdiv(loader, model, eps, i,j, dim, numb_inputs, clip, device)

                t.set_postfix(err = '{0:.2f}%'.format(sens_map[i,j]))
                t.update()
    return sens_map

def computeSensitivityMap_through_CEloss(loader, model, eps, dim, numb_inputs, clip, device):
    sens_map = torch.zeros(dim,dim,device=device)

    with trange(dim*dim) as t:
        for i in range(dim):
            for j in range(dim):
                sens_map[i,j] = test_freq_sensitivity_through_CEloss(loader, model, eps, i,j, dim, numb_inputs, clip, device)

                t.set_postfix(err = '{0:.2f}%'.format(sens_map[i,j]))
                t.update()
    return sens_map

def compute_Hessian_wrt_freq(loader, model, criteria, eps, dim, numb_inputs, clip, intermediate_point, device):
    sens_map = torch.zeros(dim,device=device)

    with trange(dim) as t:
        for i in range(dim):
            if intermediate_point:
                sens_map[i] = hessian_finite_difference_intermediate_point(loader, model, criteria, eps, i,i, dim, numb_inputs, clip, device)
            else:
                sens_map[i] = hessian_finite_difference(loader, model, criteria, eps, i,i, dim, numb_inputs, clip, device)
            t.set_postfix(err = '{0:.2f}%'.format(sens_map[i]))
            t.update()
    return sens_map

def hessian_finite_difference(loader, model, criteria, eps, x, y, size, numb_inputs, clip, device):
    total_correct = 0.
    total_correct_forward, total_correct_backward = 0.,0.
    
    total_loss = 0.
    total_loss_forward, total_loss_backward = 0.,0.
    
    dct_matrix = getDCTmatrix(size)
    dct_delta = torch.zeros(1,1,size,size, device = device)
    dct_delta[0,0,x,y] = eps
    delta = batch_idct2(dct_delta,dct_matrix).view(1,1,size,size)
    
    dct_matrix = getDCTmatrix(28)
    
    model.eval()
    total_tested_input = 0 
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            
            yp = model(X)
            
            if clip:
                yp_forward =  model((X+delta).clamp(min = 0, max = 1))
                yp_backward =  model((X-delta).clamp(min = 0, max = 1))
            else:
                yp_forward =  model(X+delta)
                yp_backward =  model(X-delta)
                
            loss = torch.nn.CrossEntropyLoss(reduction='sum')(yp,y)
            loss_forward = torch.nn.CrossEntropyLoss(reduction='sum')(yp_forward,y)
            loss_backward = torch.nn.CrossEntropyLoss(reduction='sum')(yp_backward,y)
            
            batch_correct = (yp.argmax(dim = 1) == y).sum().item()
            batch_correct_forward = (yp_forward.argmax(dim = 1) == y).sum().item()
            batch_correct_backward = (yp_backward.argmax(dim = 1) == y).sum().item()
                
        total_correct += batch_correct
        total_correct_forward += batch_correct_forward
        total_correct_backward += batch_correct_backward
        
        total_loss += loss
        total_loss_forward += loss_forward
        total_loss_backward += loss_backward
        
        total_tested_input += X.shape[0]
        if total_tested_input>numb_inputs:
            break
    
#     ipdb.set_trace()
    if criteria == 'loss':
        total_loss /= total_tested_input
        total_loss_forward /= total_tested_input
        total_loss_backward /= total_tested_input
        
        h = (total_loss_forward - 2*total_loss + total_loss_backward)/(eps**2)
        
    elif criteria == 'accuracy':
        total_acc = total_correct/total_tested_input
        total_acc_forward = total_correct_forward/total_tested_input
        total_acc_backward = total_correct_backward/total_tested_input
        h = (total_acc_forward - 2*total_acc + total_acc_backward)/(eps**2)
#     ipdb.set_trace()
    return h

def hessian_finite_difference_intermediate_point(loader, model, criteria, eps, x, y, size, numb_inputs, clip, device):
    total_correct = 0.
    total_correct_forward, total_correct_backward = 0.,0.
    
    total_loss = 0.
    total_loss_forward, total_loss_backward = 0.,0.
    
    dct_matrix = getDCTmatrix(size)
    dct_delta = torch.zeros(1,1,size,size, device = device)
    dct_delta[0,0,x,y] = eps/2
    delta = batch_idct2(dct_delta,dct_matrix).view(1,1,size,size)
    
    dct_matrix = getDCTmatrix(28)
    
    model.eval()
    total_tested_input = 0 
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            
            if clip:
                yp = model((X+delta).clamp(min = 0, max = 1))
                yp_forward =  model((X+delta+2*delta).clamp(min = 0, max = 1))
                yp_backward =  model((X+delta-2*delta).clamp(min = 0, max = 1))
            else:
                yp = model(X+delta)
                yp_forward =  model(X+delta+2*delta)
                yp_backward =  model(X+delta-2*delta)
                
            loss = torch.nn.CrossEntropyLoss(reduction='sum')(yp,y)
            loss_forward = torch.nn.CrossEntropyLoss(reduction='sum')(yp_forward,y)
            loss_backward = torch.nn.CrossEntropyLoss(reduction='sum')(yp_backward,y)
            
            batch_correct = (yp.argmax(dim = 1) == y).sum().item()
            batch_correct_forward = (yp_forward.argmax(dim = 1) == y).sum().item()
            batch_correct_backward = (yp_backward.argmax(dim = 1) == y).sum().item()
                
        total_correct += batch_correct
        total_correct_forward += batch_correct_forward
        total_correct_backward += batch_correct_backward
        
        total_loss += loss
        total_loss_forward += loss_forward
        total_loss_backward += loss_backward
        
        total_tested_input += X.shape[0]
        if total_tested_input>numb_inputs:
            break
    
#     ipdb.set_trace()
    if criteria == 'loss':
        total_loss /= total_tested_input
        total_loss_forward /= total_tested_input
        total_loss_backward /= total_tested_input
        
        h = (total_loss_forward - 2*total_loss + total_loss_backward)/(eps**2)
        
    elif criteria == 'accuracy':
        total_acc = total_correct/total_tested_input
        total_acc_forward = total_correct_forward/total_tested_input
        total_acc_backward = total_correct_backward/total_tested_input
        h = (total_acc_forward - 2*total_acc + total_acc_backward)/(eps**2)
#     ipdb.set_trace()
    return h


def test_freq_sensitivity_through_CEloss(loader, model, eps, x, y, size, numb_inputs, clip, device):
    total_loss = 0.
    dct_matrix = getDCTmatrix(size)
    dct_delta = torch.zeros(1,1,size,size, device = device)
    dct_delta[0,0,x,y] = eps
    delta_pos = batch_idct2(dct_delta,dct_matrix).view(1,1,size,size)
#     dct_delta[0,0,x,y] = -eps
#     delta_neg = batch_idct2(dct_delta,dct_matrix).view(1,1,size,size)
    
    dct_matrix = getDCTmatrix(28)
    
    kl_loss = nn.KLDivLoss(reduction="none", log_target= True)
    
    model.eval()
    total_tested_input = 0 
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            
            if clip:
                yp_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
#                 yp_hat_neg = F.log_softmax(model((X+delta_neg).clamp(min = 0, max = 1)))
            else:
                yp_hat_pos = model((X+delta_pos))
#                 yp_hat_neg = F.log_softmax(model((X+delta_neg)))
                                
            yp = model(X)
        
            loss_original = nn.CrossEntropyLoss(reduction='sum')(yp, y)
            loss_noise = nn.CrossEntropyLoss(reduction='sum')(yp_hat_pos,y)
            
#             loss_pos = (kl_loss(yp, yp_hat_pos).sum(dim=1) + kl_loss(yp_hat_pos, yp).sum(dim=1)).unsqueeze(1)
#             loss_neg = (kl_loss(yp, yp_hat_neg).sum(dim=1) + kl_loss(yp_hat_neg, yp).sum(dim=1)).unsqueeze(1)
            batch_loss = loss_noise-loss_original
        
        total_loss += batch_loss
        
        total_tested_input += X.shape[0]
        if total_tested_input>numb_inputs:
            break
        
    test_loss = total_loss / total_tested_input
    return test_loss

def test_freq_sensitivity(loader, model, eps, x, y, size, numb_inputs, clip, device):
    total_correct = 0.
    dct_matrix = getDCTmatrix(size)
    dct_delta = torch.zeros(1,1,size,size, device = device)
    dct_delta[0,0,x,y] = eps
    delta_pos = batch_idct2(dct_delta,dct_matrix).view(1,1,size,size)
    dct_delta[0,0,x,y] = -eps
    delta_neg = batch_idct2(dct_delta,dct_matrix).view(1,1,size,size)
    
    dct_matrix = getDCTmatrix(28)
    
    model.eval()
    total_tested_input = 0 
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            
            if clip:
                y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
                y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
            else:
                y_hat_pos = model((X+delta_pos))
                y_hat_neg = model((X+delta_neg))
                
            y = model(X).argmax(dim=1)
            batch_correct_pos = (y_hat_pos.argmax(dim = 1) == y)
            batch_correct_neg = (y_hat_neg.argmax(dim = 1) == y)
            batch_correct = (batch_correct_neg*batch_correct_pos).sum().item()
        
        total_correct += batch_correct
        
        total_tested_input += X.shape[0]
        if total_tested_input>numb_inputs:
            break
        
    test_err = (total_tested_input-total_correct) / total_tested_input * 100
    return test_err

def test_freq_sensitivity_through_KLdiv(loader, model, eps, x, y, size, numb_inputs, clip, device):
    total_loss = 0.
    dct_matrix = getDCTmatrix(size)
    dct_delta = torch.zeros(1,1,size,size, device = device)
    dct_delta[0,0,x,y] = eps
    delta_pos = batch_idct2(dct_delta,dct_matrix).view(1,1,size,size)
    dct_delta[0,0,x,y] = -eps
    delta_neg = batch_idct2(dct_delta,dct_matrix).view(1,1,size,size)
    
    dct_matrix = getDCTmatrix(28)
    
    kl_loss = nn.KLDivLoss(reduction="none", log_target= True)
    
    model.eval()
    total_tested_input = 0 
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            
            if clip:
                yp_hat_pos = F.log_softmax(model((X+delta_pos).clamp(min = 0, max = 1)))
                yp_hat_neg = F.log_softmax(model((X+delta_neg).clamp(min = 0, max = 1)))
            else:
                yp_hat_pos = F.log_softmax(model((X+delta_pos)))
                yp_hat_neg = F.log_softmax(model((X+delta_neg)))
                                
            yp = F.log_softmax(model(X))
            
            loss_pos = (kl_loss(yp, yp_hat_pos).sum(dim=1) + kl_loss(yp_hat_pos, yp).sum(dim=1)).unsqueeze(1)
            loss_neg = (kl_loss(yp, yp_hat_neg).sum(dim=1) + kl_loss(yp_hat_neg, yp).sum(dim=1)).unsqueeze(1)
            batch_loss = torch.cat([loss_pos,loss_neg],dim=1).amax(dim=1).sum()
        
        total_loss += batch_loss
        
        total_tested_input += X.shape[0]
        if total_tested_input>numb_inputs:
            break
        
    test_loss = total_loss / total_tested_input
    return test_loss


def visualize_freq_perturbation(loader, eps, sens_map, clip, size, device):
    argmax_row = (sens_map.argmax()//28).item()
    argmax_col = (sens_map.argmax()%28).item()
    
    dct_delta = torch.zeros(1,1,size,size, device = device)
    dct_delta[0,0,argmax_row,argmax_col] = eps
    delta = idct2(dct_delta)
    
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        break
    
    X_0 = X[y==0,:,:,:][0,:,:,:].squeeze()
    X_1 = X[y==1,:,:,:][0,:,:,:].squeeze()
    
    X0_delta = (X_0+delta).clamp(min = 0, max = 1)
    X1_delta = (X_1+delta).clamp(min = 0, max = 1)
        
    return X0_delta, X1_delta, delta
    
    

def data_l2_norm(loader, device):
    avg_norm = torch.norm(loader.dataset.tensors[0], p=2, dim = (2,3)).mean().item()
    std_norm = torch.norm(loader.dataset.tensors[0], p=2, dim = (2,3)).std().item()
    return avg_norm, std_norm

def computeSensitivityMap_score_diff(loader, model, eps, dim, numb_inputs, clip, device):
    sens_map = torch.zeros(dim,dim,device=device)

    with trange(dim*dim) as t:
        for i in range(dim):
            for j in range(dim):
                sens_map[i,j]  = test_freq_sensitivity_score_diff(loader, model, eps, i,j, dim, numb_inputs, clip, device)

                t.set_postfix(acc = '{0:.2f}%'.format(sens_map[i,j]))
                t.update()
    return sens_map



def computeSensitivityMap_analysis(loader, model, eps, dim, numb_inputs, clip, device):
    sens_map = torch.zeros(6,dim,dim,device=device)

    with trange(dim*dim) as t:
        for i in range(dim):
            for j in range(dim):
#                 ipdb.set_trace()
                result = test_freq_sensitivity_analysis(loader, model, eps, i,j, dim, numb_inputs, clip, device)
#                 ipdb.set_trace()
                sens_map[:,i,j] = result

#                 t.set_postfix(acc = '{0:.2f}%'.format(sens_map[i,j]))
                t.update()
    return sens_map

def data_l2_norm(loader, device):
    avg_norm = torch.norm(loader.dataset.tensors[0], p=2, dim = (2,3)).mean().item()
    std_norm = torch.norm(loader.dataset.tensors[0], p=2, dim = (2,3)).std().item()
    return avg_norm, std_norm
    
def avg_attack_DCT(loader, model, eps, dim, attack, clip, device):
    dct_delta = torch.zeros(dim,dim, device=device)
    avg_delta = torch.zeros(dim,dim, device=device)
    
    if attack == "pgd":
        param = {'ord': 2,
                 'epsilon': eps,
                 'alpha': 2.5*eps/100.,
                 'num_iter': 100,
                 'restarts': 1,
                 'loss_fn': nn.CrossEntropyLoss(),
                 'clip': clip}
    elif attack == "fgsm":
        param = {'ord': 2,
                 'epsilon': eps,
                 'loss_fn': nn.CrossEntropyLoss(),
                 'clip': clip}
        
    if len(loader.dataset.classes) == 2:
        param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        
    dct_matrix = getDCTmatrix(dim)
    
    with trange(len(loader)) as t:
        for X,y in loader:
            model.eval()
            X,y = X.to(device), y.to(device)
            
            if len(loader.dataset.classes) != 2:
                y = y.long()

            if attack == "pgd":
                delta = pgd_rand_nn(**param).generate(model, X, y)
            elif attack == "fgsm":
                delta = fgsm_nn(**param).generate(model, X, y)
                
#             ipdb.set_trace()
            dct_delta += batch_dct2(delta, dct_matrix).abs().sum(dim=0)
            avg_delta += delta.squeeze().sum(dim=0)
#             dct_delta += batch_dct2(delta, dct_matrix).sum(dim=0)
            
            t.update()
        
    dct_delta = dct_delta / len(loader.dataset)
    avg_delta = avg_delta / len(loader.dataset)
    return dct_delta, avg_delta

def single_attack_DCT(loader, model, eps, dim, attack, clip, device):
    num_class = len(loader.dataset.classes)
    dct_delta = torch.zeros(num_class, dim, dim, device=device)
    delta = torch.zeros(num_class, dim, dim, device=device)
    data = torch.zeros(num_class, dim, dim, device=device)
    dct_data = torch.zeros(num_class, dim, dim, device=device)
    
    if attack == "pgd":
        param = {'ord': 2,
                 'epsilon': eps,
                 'alpha': 2.5*eps/100.,
                 'num_iter': 100,
                 'restarts': 1,
                 'loss_fn': nn.CrossEntropyLoss(),
                 'clip': clip}
    elif attack == "fgsm":
        param = {'ord': 2,
                 'epsilon': eps,
                 'loss_fn': nn.CrossEntropyLoss(),
                 'clip': clip}
        
    if len(loader.dataset.classes) == 2:
        param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        
    dct_matrix = getDCTmatrix(dim)
    
    
    for X,y in loader:
        model.eval()
        X,y = X.to(device), y.to(device)
        if len(loader.dataset.classes) != 2:
            y = y.long()

        if attack == "pgd":
            _delta = pgd_rand_nn(**param).generate(model, X, y)
        elif attack == "fgsm":
            _delta = fgsm_nn(**param).generate(model, X, y)   
        break
    
    _dct_data = batch_dct2(X, dct_matrix)
    _dct_delta = batch_dct2(_delta, dct_matrix)
    for i in range(num_class):
        data[i,:,:] = X[y==i,:,:,:][0,:,:,:].squeeze()
        dct_data[i,:,:] = _dct_data[y==i,:,:][0,:,:]
        delta[i,:,:] = _delta[y==i,:,:,:][0,:,:,:].squeeze()
        dct_delta[i,:,:]= _dct_delta[y==i,:,:][0,:,:]
        
    return data, dct_data, delta, dct_delta

# def avg_attack(loader, model, eps, dim, attack, clip, device):
#     dct_delta = torch.zeros(dim,dim, device=device)
    
#     if attack == "pgd":
#         param = {'ord': 2,
#                  'epsilon': eps,
#                  'alpha': 2.5*eps/100.,
#                  'num_iter': 100,
#                  'restarts': 1,
#                  'loss_fn': nn.CrossEntropyLoss(),
#                  'clip': clip}
#     elif attack == "fgsm":
#         param = {'ord': 2,
#                  'epsilon': eps,
#                  'loss_fn': nn.CrossEntropyLoss(),
#                  'clip': clip}
        
#     if len(loader.dataset.classes) == 2:
#         param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
            
#     with trange(len(loader)) as t:
#         for X,y in loader:
#             model.eval()
#             X,y = X.to(device), y.to(device)
            
#             if len(loader.dataset.classes) != 2:
#                 y = y.long()

#             if attack == "pgd":
#                 delta = pgd_rand_nn(**param).generate(model, X, y)
#             elif attack == "fgsm":
#                 delta = fgsm_nn(**param).generate(model, X, y)
                
#             dct_delta += delta.sum(dim=0)
# #             dct_delta += batch_dct2(delta, dct_matrix).sum(dim=0)
            
#             t.update()
        
#     dct_delta = dct_delta / len(loader.dataset)
#     return dct_delta

def single_image_freq_exam(loader, model, eps, clip, device):
    
    
    for X,y in loader:
        model.eval()
        X,y = X.to(device), y.to(device)
        
        y_hat = model(X)
        if len(loader.dataset.classes) == 2:
            y = y.float().view(-1,1)
            correct = ((y_hat > 0) == (y==1))
        else:
            y = y.long()
            correct = (y_hat.argmax(dim = 1) == y)
        # makes sure that we are examine an image that can 
        # be classified correctly without perturbation
        X = X[correct.squeeze() ==True, :,:,:]
        y = y[correct.squeeze() ==True]
        break
    
    pgd_param = {'ord': 2,
                 'epsilon': eps,
                 'alpha': 2.5*eps/100.,
                 'num_iter': 100,
                 'restarts': 1,
                 'loss_fn': nn.CrossEntropyLoss(),
                 'clip': clip}
    
    fgsm_param = {'ord': 2,
                 'epsilon': eps,
                 'loss_fn': nn.CrossEntropyLoss(),
                 'clip': clip}
    
    if len(loader.dataset.classes) == 2:
        fgsm_param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        pgd_param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        
    
    pgd = pgd_rand_nn(**pgd_param).generate(model, X, y)[0,:,:,:]
    fgsm = fgsm_nn(**fgsm_param).generate(model, X, y)[0,:,:,:]
    
#     ipdb.set_trace()
                
    X = X[0,:,:,:].view(1,1,X.shape[2],X.shape[3])
    y = y[0].view(1)
    dct_matrix = getDCTmatrix(X.shape[2])
    dct_fgsm = batch_dct2(fgsm, dct_matrix).abs()
    dct_pgd = batch_dct2(pgd, dct_matrix).abs()
    dct_X = batch_dct2(X, dct_matrix).abs()
    
    sens_map = torch.zeros(X.shape[2],X.shape[2], device = device)
    for i in range(X.shape[2]):
        for j in range(X.shape[3]):
            
            dct_delta = torch.zeros(1,1,X.shape[2],X.shape[2], device = device)
    
            dct_delta[0,0,i,j] = eps
            delta_pos = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])
            dct_delta[0,0,i,j] = -eps
            delta_neg = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])
#             ipdb.set_trace()
            model.eval()
            if clip:
                y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
                y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
            else:
                y_hat_pos = model((X+delta_pos))
                y_hat_neg = model((X+delta_neg))
                        
            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                pos_true = ((y_hat_pos > 0) == (y==1))
                neg_true = ((y_hat_neg > 0) == (y==1))
            else:
                y = y.long()
                pos_true = (y_hat_pos.argmax(dim = 1) == y)
                neg_true = (y_hat_neg.argmax(dim = 1) == y)
#             ipdb.set_trace()
            sens_map[i,j] = pos_true*neg_true
    
#     eps_map = torch.zeros(X.shape[2],X.shape[2], device = device)
#     with trange(X.shape[2]**2) as t:
#         for i in range(X.shape[2]):
#             for j in range(X.shape[3]):
#                 k = 0
#                 neg_true = 1
#                 pos_true = 1
#                 while neg_true ==1 and pos_true==1: #both correct
#                     k += 1
#                     dct_delta = torch.zeros(1,1,X.shape[2],X.shape[2], device = device)

#                     dct_delta[0,0,i,j] = k
#                     delta_pos = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])
#                     dct_delta[0,0,i,j] = -k
#                     delta_neg = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])

#     #                 ipdb.set_trace()
#                     model.eval()
#                     if clip:
#                         y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
#                         y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
#                     else:
#                         y_hat_pos = model(X+delta_pos)
#                         y_hat_neg = model(X+delta_neg)

#                     if len(loader.dataset.classes) == 2:
#                         y = y.float().view(-1,1)
#                         pos_true = ((y_hat_pos > 0) == (y==1)).sum().item()
#                         neg_true = ((y_hat_neg > 0) == (y==1)).sum().item()
#                     else:
#                         y = y.long()
#                         pos_true = (y_hat_pos.argmax(dim = 1) == y).sum().item()
#                         neg_true = (y_hat_neg.argmax(dim = 1) == y).sum().item()
                        
#                     t.set_postfix(i = '{0:.2f}'.format(i),
#                                   j = '{0:.2f}'.format(j),
#                                   k = '{0:.2f}'.format(k))
                
                    
#                     if k ==100:
#                         break
#                 t.update()


#                 eps_map[i,j] = k
    
    return X, dct_X, fgsm, dct_fgsm, pgd, dct_pgd, sens_map

def avg_data_DCT(loader, dim, device, per_class=False, log=False):
    dct_matrix = getDCTmatrix(dim)
    if per_class:
        num_class = len(loader.dataset.classes)
        dct_X = torch.zeros(num_class,dim,dim, device=device)
    else:
        dct_X = torch.zeros(1,dim,dim, device=device)
        
    for X,y in loader:
        X = X.to(device)
        if per_class:
            for i in range(num_class):
                X_per_class = X[y==i,:,:,:]
                if log:
                    dct_X[i,:,:] += torch.log(batch_dct2(X_per_class, dct_matrix).abs()).sum(dim=0)
                else:
                    dct_X[i,:,:] += batch_dct2(X_per_class, dct_matrix).abs().sum(dim=0)
        else:
            if log:
                dct_X += torch.log(batch_dct2(X, dct_matrix).abs()).sum(dim=0)
            else:
                dct_X += batch_dct2(X, dct_matrix).abs().sum(dim=0)
                
    if per_class:
        for i in range(num_class):
            dct_X[i,:,:] /= (loader.dataset.targets == i).sum()
    else:
        dct_X /= len(loader.dataset)
        
    return dct_X.squeeze()

# def avg_data(loader, dim, device, per_class=False):
#     if per_class:
#         num_class = len(loader.dataset.classes)
#         data = torch.zeros(num_class,dim,dim, device=device)
#     else:
#         data = torch.zeros(1,dim,dim, device=device)
        
#     for X,y in loader:
#         X = X.to(device)
#         if per_class:
#             for i in range(num_class):
#                 data[i,:,:] += X[y==i,:,:,:].sum(dim=0)
#         else:
#             data += X.sum(dim=0)

#     if per_class:
#         ipdb.set_trace()
#         for i in range(num_class):
#             data[i,:,:] /= X[y==i,:,:,:].shape[0]
#     else:
#         X /= len(loader.dataset)
#     return X.squeeze()


def avg_x_FFT(loader, dim, device):
    fft_X = np.zeros((dim,dim),dtype=np.float)
    for X,y in loader:
        X = X.cpu().numpy()
#         fft_X += scipy.fft.fft2(X).sum(axis = 0).squeeze()
        fft_X += np.abs(scipy.fft.fft2(X)).sum(axis = 0).squeeze()
#         break
    fft_X = np.fft.fftshift(fft_X / len(loader.dataset))
    return fft_X.squeeze()

def test_freq_sensitivity_score_diff(loader, model, eps, x, y, size, numb_inputs, clip, device):
    total_change = 0.

    dct_delta = torch.zeros(1,1,size,size, device = device)
    
    dct_delta[0,0,x,y] = eps
    delta_pos = idct2(dct_delta).view(1,1,size,size)
    dct_delta[0,0,x,y] = -eps
    delta_neg = idct2(dct_delta).view(1,1,size,size)
    
    model.eval()
    total_tested_input = 0 
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            
            y_hat = model(X)
            
            if clip:
                y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
                y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
            else:
                y_hat_pos = model((X+delta_pos))
                y_hat_neg = model((X+delta_neg))
                
                
                
            
            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
#                 ipdb.set_trace()
                max_change = torch.max(torch.abs(y_hat - y_hat_pos), torch.abs(y_hat - y_hat_neg)).sum().item()
            else:
                pass
#                 y = y.long()
#                 batch_correct_pos = (y_hat_pos.argmax(dim = 1) == y)
#                 batch_correct_neg = (y_hat_neg.argmax(dim = 1) == y)
#                 batch_correct = (batch_correct_neg*batch_correct_pos).sum().item()
        
        total_change += max_change
        
        total_tested_input += X.shape[0]
        if total_tested_input>numb_inputs:
            break
        
    change = total_change / total_tested_input * 100
    return change


def test_freq_sensitivity_analysis(loader, model, eps, x, y, size, numb_inputs, clip, device):
    total_change = 0.

    dct_delta = torch.zeros(1,1,size,size, device = device)
    
    dct_delta[0,0,x,y] = eps
    delta_pos = idct2(dct_delta).view(1,1,size,size)
    dct_delta[0,0,x,y] = -eps
    delta_neg = idct2(dct_delta).view(1,1,size,size)
    
    model.eval()
    theta = model.state_dict()['conv1.weight'].squeeze().clone().detach()
    total_tested_input = 0 
    
    
    case = torch.zeros(6, device = device)

#     iterate thru all x:
        
    for m in range(theta.shape[0]):

        if (theta[m,:,:] * x+delta_pos.squeeze()).sum() > 0:

            if eps* theta[m,x,y] > 0:
                case[0] +=1
            elif eps* theta[m,x,y] < 0 and eps* theta[m,x,y] > -(theta[m,:,:] * delta_pos.squeeze()).sum():
                case[1] +=1
            elif eps* theta[m,x,y] < -(theta[m,:,:] * delta_pos.squeeze()).sum():
                case[2] +=1   
        else:
            if eps* theta[m,x,y] < 0:
                case[3] +=1
            elif  0 < eps* theta[m,x,y] and eps* theta[m,x,y] < -(theta[m,:,:] * delta_pos.squeeze()).sum():
                case[4] +=1
            elif eps* theta[m,x,y] > -(theta[m,:,:] * delta_pos.squeeze()).sum():
                case[5] +=1
          
    return case/theta.shape[0]*100


def test_remove_freq(loader, model, f_size, device):
    dct_matrix_28 = getDCTmatrix(28)
    dct_matrix_32 = getDCTmatrix(32)
    total_loss, total_correct = 0., 0.
    n_samples = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)
        
        if x.shape[1] ==1:
            x_tilde = batch_dct2(x,dct_matrix_28)
            mask = torch.tensor(mask_radial(28,f_size)).repeat(x.shape[0],1,1).to(device)
            x_tilde *= mask
            x_back = batch_idct2(x_tilde, dct_matrix_28).unsqueeze(1)
        elif x.shape[1]==3:
            x_tilde = batch_dct2_3channel(x,dct_matrix_32)
            mask = torch.tensor(mask_radial(32,f_size)).repeat(x.shape[0],3,1,1).to(device)
            x_tilde *= mask
            x_back = batch_idct2_3channel(x_tilde, dct_matrix_32)
            
        with torch.no_grad():
            y_hat = model(x_back)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            
        total_correct += batch_correct
        total_loss += loss.item() * x.shape[0]
        
        n_samples += x.shape[0]
        if n_samples >= 2000:
            break
        
    test_acc = total_correct / n_samples * 100
    test_loss = total_loss / n_samples
    return test_acc, test_loss

def test_nothing_removed(loader, model, device):

    total_loss, total_correct = 0., 0.
    n_samples = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)
            
        with torch.no_grad():
            y_hat = model(x)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            
        total_correct += batch_correct
        total_loss += loss.item() * x.shape[0]
        
        n_samples += x.shape[0]
        if n_samples >= 2000:
            break
        
    test_acc = total_correct / n_samples * 100
    test_loss = total_loss / n_samples
    return test_acc, test_loss

def test_perturb_freq(loader, model, f_size, var, device):
    dct_matrix_28 = getDCTmatrix(28)
    dct_matrix_32 = getDCTmatrix(32)
    total_loss, total_correct = 0., 0.
    
    n_samples = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)
        
        if x.shape[1] ==1:
            x_tilde = batch_dct2(x,dct_matrix_28)
            mask = torch.tensor(mask_radial(28,f_size)).repeat(x.shape[0],1,1).to(device)
            x_noise = var**0.5 * torch.randn_like(mask) *(1-mask)
            x_tilde += x_noise
            x_back = batch_idct2(x_tilde, dct_matrix_28).unsqueeze(1)
        elif x.shape[1]==3:
            
            x_tilde = batch_dct2_3channel(x,dct_matrix_32)
            mask = torch.tensor(mask_radial(32,f_size)).repeat(x.shape[0],3,1,1).to(device)
            x_noise = var**0.5 * torch.randn_like(mask) *(1-mask)
            
            x_tilde += x_noise
            x_back = batch_idct2_3channel(x_tilde, dct_matrix_32)
            
        with torch.no_grad():
            y_hat = model(x_back)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()

        total_correct += batch_correct
        total_loss += loss.item() * x.shape[0]
        
        n_samples += x.shape[0]
        if n_samples >= 2000:
            break
        
    test_acc = total_correct / n_samples * 100
    test_loss = total_loss / n_samples
    return test_acc, test_loss


def return_g(train_loader,model,device):
    for x,y in train_loader:
        model.eval()
        model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        loss.backward()
        break
    g_list = torch.empty(0,device=device)
    w_list = torch.empty(0,device=device)
    for param in model.parameters():
        g_list = torch.cat([param.grad.data.detach().abs().flatten(), g_list])
        w_list = torch.cat([param.data.detach().abs().flatten(), w_list])
    return g_list,w_list
    

def test_remove_small_g(train_loader, test_loader, model, device, percentile=None, f_size=None, var=None, evaluation='vanilla', remove_zero_g=False):

    for x,y in train_loader:
        model.eval()
        model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        loss.backward()
        break
    
    g_list = torch.empty(0,device=device)
    for param in model.parameters():
        g_list = torch.cat([param.grad.data.detach().abs().flatten(), g_list])
    if remove_zero_g:
        g_list = g_list[g_list>1e-18]
    threshold = np.percentile(g_list.cpu(),percentile)

    for param in model.parameters():
        param.data = param.data * (param.grad.data.detach().abs() > threshold).float()
    
    if evaluation == 'vanilla':
        test_acc, test_loss = test_nothing_removed(test_loader, model, device)
    elif evaluation == 'remove':
        test_acc, test_loss = test_remove_freq(test_loader, model, f_size, device)
    elif evaluation == 'perturb':
        test_acc, test_loss = test_perturb_freq(test_loader, model, f_size, var, device)
    return test_acc, test_loss


def test_remove_random_g(train_loader, test_loader, model, device, percentile=None, f_size=None, var=None, evaluation='vanilla', remove_zero_g=False):
    
    seed_everything(0)
    for param in model.parameters():
        random_remove = torch.rand_like(param.data.detach().abs())>(percentile/100.)
        param.data = param.data * random_remove.float()
    
    if evaluation == 'vanilla':
        test_acc, test_loss = test_nothing_removed(test_loader, model, device)
    elif evaluation == 'remove':
        test_acc, test_loss = test_remove_freq(test_loader, model, f_size, device)
    elif evaluation == 'perturb':
        test_acc, test_loss = test_perturb_freq(test_loader, model, f_size, var, device)
    return test_acc, test_loss

def test_remove_large_g(train_loader, test_loader, model, device, percentile=None, f_size=None, var=None, evaluation='vanilla', remove_zero_g=False):
    
    for x,y in train_loader:
        model.eval()
        model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        loss.backward()
        break
    
    g_list = torch.empty(0,device=device)
    for param in model.parameters():
        g_list = torch.cat([param.grad.data.detach().abs().flatten(), g_list])
    if remove_zero_g:
        g_list = g_list[g_list>1e-18]
    threshold = np.percentile(g_list.cpu(),100-percentile)

    for param in model.parameters():
        param.data = param.data * (param.grad.data.detach().abs() < threshold).float()
    
    if evaluation == 'vanilla':
        test_acc, test_loss = test_nothing_removed(test_loader, model, device)
    elif evaluation == 'remove':
        test_acc, test_loss = test_remove_freq(test_loader, model, f_size, device)
    elif evaluation == 'perturb':
        test_acc, test_loss = test_perturb_freq(test_loader, model, f_size, var, device)
    return test_acc, test_loss


def test_clean_single_data(x, y, model, device):
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
    return loss

def test_gaussian_single_data(x, y, model, var, num_noise_samples, device):
    total_loss = 0
    model.eval()
    for i in range(num_noise_samples):
        noise = (var**0.5)*torch.randn_like(x, device = x.device)
        with torch.no_grad():
            y_hat = model(x+noise)
            total_loss += torch.nn.CrossEntropyLoss()(y_hat, y)

    loss = total_loss/num_noise_samples     
    return loss

def noise_l2norm_estimate(x, var, num_noise_samples):
    noise_norm = 0
    for i in range(num_noise_samples):
        noise = (var**0.5)*torch.randn_like(x)
        noise_norm += torch.norm(noise, p = 2)

    noise_norm /= num_noise_samples     
    return noise_norm

def exact_hessian_single_sample(X, y, model, device):
    ## modified : does not require grad -> no gpu memory consumption -> potential issue if backprop through this.

    if X.shape[0] is not 1:
        print("this function only computes hessian for a single sample!!!")
        return 0

    _dim = X.shape[1]*X.shape[2]*X.shape[3]
    X.requires_grad = True
    H = torch.empty(_dim, _dim, requires_grad = False, device = device)

#     with ctx_noparamgrad_and_eval(model):
    loss = nn.CrossEntropyLoss()(model(X),y)
    dldx = list(grad(loss, X, create_graph = True))[0].view(_dim)

    for j in range(0,_dim):
        #differentiating each element in dldx wrt input, then assign it to the row vector in hessian
        H[j,:] = list(grad(dldx[j],X,create_graph=True))[0].detach().view(1,_dim)

    return H.detach()

def estimate_2nd_order_term(h, var, num_noise_samples):
    total_estimate = 0
    for i in range(num_noise_samples):
        noise = (var**0.5)*torch.randn([28,28], device = 'cuda')
        total_estimate += torch.matmul(noise.view(1,784),torch.matmul(h, noise.view(784,1)))
        
    return total_estimate/num_noise_samples

def decompose_2nd_order_term(h, r, var, num_noise_samples):
    xl_h_xl, xh_h_xh = 0, 0
    
    mask_L = torch.tensor(mask_radial(28, r), dtype=torch.float32, device='cuda')
    mask_H = 1-mask_L
    for i in range(num_noise_samples):
        noise = (var**0.5)*torch.randn([28,28], device = 'cuda')
        dct_noise = dct2(noise)
        noise_L = idct2(dct_noise * mask_L)
        noise_H = idct2(dct_noise * mask_H)
        
#         assert (noise - (noise_L + noise_H)).abs().max() < 1e-6
#         print()
        _xl_h_xl = torch.matmul(noise_L.view(1,784),torch.matmul(h, noise_L.view(784,1)))
        _xh_h_xh = torch.matmul(noise_H.view(1,784),torch.matmul(h, noise_H.view(784,1)))
#         _xl_h_xh = torch.matmul(noise_L.view(1,784),torch.matmul(h, noise_H.view(784,1)))
#         _xh_h_xl = torch.matmul(noise_H.view(1,784),torch.matmul(h, noise_L.view(784,1)))
#         _sum = _xl_h_xl + _xh_h_xh + _xl_h_xh + _xh_h_xl
#         _sum = _xl_h_xl + _xh_h_xh
        xl_h_xl += _xl_h_xl
        xh_h_xh += _xh_h_xh
#         xl_h_xl += (_xl_h_xl / (_sum))
#         xh_h_xh += (_xh_h_xh / (_sum))
#         xl_h_xh += ((_xl_h_xh+_xh_h_xl) / (_sum) *100)
#         xh_h_xl += (_xh_h_xl / (_sum) *100)
#         x_h_x = torch.matmul(noise.view(1,784),torch.matmul(h, noise.view(784,1)))
#         print((x_h_x - (_xl_h_xl+_xh_h_xh+_xl_h_xh+_xh_h_xl)).abs().max())
#         assert (x_h_x - (_xl_h_xl+_xh_h_xh+_xl_h_xh+_xh_h_xl)).abs().max() < 1e-6

    return xl_h_xl.item()/num_noise_samples, xh_h_xh.item()/num_noise_samples


def return_ratio_of_xl_H_xl_over_xh_H_xh(loader, var, radius_list, model, num_data_samples, num_noise_samples, device, data_var=None):

    ratio_list = []
    for radius in radius_list:
        xl_H_xl, xh_H_xh= 0, 0
        _sample_counter =0

        for x,y in loader:
            if data_var is None:
                x = x.to(device)
                y = y.to(device)
            else:
                x = x.to(device) + (data_var**0.5)*torch.randn_like(x, device = device)
                y = y.to(device)
            hessian = exact_hessian_single_sample(x, y, model, device)
#             ipdb.set_trace()
            _xl_H_xl, _xh_H_xh = decompose_2nd_order_term(hessian, radius, var, num_noise_samples)
#             print(_xl_H_xl, _xh_H_xh)
            xl_H_xl += _xl_H_xl
            xh_H_xh += _xh_H_xh
                
            if _sample_counter+1>=num_data_samples:
                break
            else:
                 _sample_counter += 1

        ratio_list.append(xl_H_xl/xh_H_xh)

    return ratio_list


def computeSensitivityMap_v2(loader, dataset, model, eps, rad_list, device):
    # Aug 20, 2022
    if dataset in ['mnist', 'fashionmnist']:
        img_size = 28
        dct_matrix = getDCTmatrix(28)
    elif dataset in ['cifar10', 'cifar100', 'svhn']:
        img_size = 32
        dct_matrix = getDCTmatrix(32)
    elif dataset in ['tiny','dtd']:
        img_size = 64
        dct_matrix = getDCTmatrix(64)
    elif dataset in ['imagenette']:
        img_size = 224
        dct_matrix = getDCTmatrix(224)

    sens_map = torch.zeros(len(rad_list), device=device)
    
    with trange(len(rad_list)) as t:
        for i, radius in enumerate(rad_list):
            sens_map[i] = test_freq_sensitivity_v2(loader, dataset, model, eps, radius, dct_matrix, device)
            t.update()
    return sens_map.detach().cpu().numpy()

def test_freq_sensitivity_v2(loader, dataset, model, eps, r, dct_matrix, device):
    # Aug 20, 2022
    
    if dataset in ['mnist', 'fashionmnist']:
        img_size = 28
        channel = 1
    elif dataset in ['cifar10', 'cifar100', 'svhn']:
        img_size = 32
        channel = 3
    elif dataset in ['tiny','dtd']:
        img_size = 64
        channel = 3
    elif dataset in ['imagenette']:
        img_size = 224
        channel = 3
    mask = torch.tensor(equal_dist_from_top_left(img_size, r), 
                        device = device, 
                        dtype=torch.float32).view(1,1,img_size,img_size).expand(1,channel,img_size,img_size)
        
    total_same_pred = 0.
    
    dct_delta_masked = torch.full([1,channel,img_size, img_size], fill_value = eps, device = device)*mask
    
    if dataset in ['mnist', 'fashionmnist']:
        delta = batch_idct2(dct_delta_masked, dct_matrix).unsqueeze(1)
    else:
        delta = batch_idct2_3channel(dct_delta_masked, dct_matrix)

    model.eval()
    total_tested_input = 0 
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            
            pred_pos = model((X+delta.expand(X.shape[0], channel, img_size, img_size))).argmax(dim=1)
            pred_neg = model((X-delta.expand(X.shape[0], channel, img_size, img_size))).argmax(dim=1)
                
            pred = model(X).argmax(dim=1)
            batch_same_pred_pos = (pred_pos == pred)
            batch_same_pred_neg = (pred_neg == pred)
            batch_same_pred = (batch_same_pred_pos*batch_same_pred_neg).sum().item()
        
        total_same_pred += batch_same_pred
        
        total_tested_input += X.shape[0]
        if total_tested_input>1000:
            break
        
    sens = (total_tested_input-total_same_pred) / total_tested_input * 100
    return sens

def computeSensitivityMap_v3(loader, dataset, model, eps, radius, device):
    # Aug 22, 2022
    if dataset in ['mnist', 'fashionmnist']:
        img_size = 28
        channel = 1
        dct_matrix = getDCTmatrix(28)
    elif dataset in ['cifar10', 'cifar100', 'svhn']:
        img_size = 32
        channel = 3
        dct_matrix = getDCTmatrix(32)
    elif dataset in ['tiny','dtd']:
        img_size = 64
        channel = 3
        dct_matrix = getDCTmatrix(64)
    elif dataset in ['imagenette']:
        img_size = 224
        channel = 3
        dct_matrix = getDCTmatrix(224)
        
    _mask = torch.tensor(mask_radial(img_size, radius), 
                        device = device, 
                        dtype=torch.float32)
    
    total_tested_input = 0 
    total_same_pred_LF, total_same_pred_HF = 0., 0.
    
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)
        
        mask = _mask.expand(x.shape[0], channel, img_size, img_size)

        if dataset in ['mnist', 'fashionmnist']:
            delta_LF = batch_idct2(mask*eps, dct_matrix).unsqueeze(1)
            delta_HF = batch_idct2((1-mask)*eps, dct_matrix).unsqueeze(1)
        else:
            delta_LF = batch_idct2_3channel(mask*eps, dct_matrix)
            delta_HF = batch_idct2_3channel((1-mask)*eps, dct_matrix)
            
        with torch.no_grad(): 
            pred_pos_LF = model((x+delta_LF.expand(x.shape[0], channel, img_size, img_size))).argmax(dim=1)
            pred_neg_LF = model((x-delta_LF.expand(x.shape[0], channel, img_size, img_size))).argmax(dim=1)
            
            pred_pos_HF = model((x+delta_HF.expand(x.shape[0], channel, img_size, img_size))).argmax(dim=1)
            pred_neg_HF = model((x-delta_HF.expand(x.shape[0], channel, img_size, img_size))).argmax(dim=1)
            
            pred = model(x).argmax(dim=1)
            batch_same_pred_pos_LF = (pred_pos_LF == pred)
            batch_same_pred_neg_LF = (pred_neg_LF == pred)
            batch_same_pred_pos_HF = (pred_pos_HF == pred)
            batch_same_pred_neg_HF = (pred_neg_HF == pred)
            batch_same_pred_LF = (batch_same_pred_pos_LF*batch_same_pred_neg_LF).sum().item()
            batch_same_pred_HF = (batch_same_pred_pos_HF*batch_same_pred_neg_HF).sum().item()

        total_same_pred_LF += batch_same_pred_LF
        total_same_pred_HF += batch_same_pred_HF
        
        total_tested_input += x.shape[0]

    sens = np.array([(total_tested_input-total_same_pred_LF) / total_tested_input * 100, (total_tested_input-total_same_pred_HF) / total_tested_input * 100])
    return sens


def test_remove_small_g_v2(train_loader, test_loader, final_model, other_model, device, percentile=None, f_size=None, var=None, evaluation='vanilla', remove_zero_g=False):
    
    seed_everything(0)
    for x,y in train_loader:
        other_model.eval()
        other_model.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = other_model(x)
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        loss.backward()
        break
    
    g_list = torch.empty(0,device=device)
#     ipdb.set_trace()
    for other_param in other_model.parameters():
        g_list = torch.cat([other_param.grad.data.detach().abs().flatten(), g_list])
#     ipdb.set_trace()
#     if remove_zero_g:
#         ipdb.set_trace()
#         g_list = g_list[g_list>1e-18]
        
    threshold = np.percentile(g_list.cpu(),percentile)
#     threshold = 0.
#     print(threshold)
#     ipdb.set_trace()

    num_param_zeroed = 0
    for param, other_param in zip(final_model.parameters(),other_model.parameters()):
#         ipdb.set_trace()
        param.data = param.data * (other_param.grad.data.detach().abs() > threshold).float()
        num_param_zeroed += (other_param.grad.data.detach().abs() < threshold).sum()
    
    print(num_param_zeroed)
    if evaluation == 'return_model':
        return final_model

    if evaluation == 'vanilla':
        test_acc, test_loss = test_nothing_removed(test_loader, final_model, device)
    elif evaluation == 'remove':
        test_acc, test_loss = test_remove_freq(test_loader, final_model, f_size, device)
    elif evaluation == 'perturb':
        r_list = [7.24, 10.5, 13.03, 15.13, 17.02, 18.68, 20.15, 21.8, 23.05,24.3,25.55,26.8,27.85,29.65,32.1]
        num_noise = 1
        
        loss_sgd = np.zeros(16)
        loss_adam = np.zeros(16)

        acc_sgd = np.zeros(16)
        acc_adam = np.zeros(16)

        test_acc, test_loss = test_gaussian_LF_HF_v2(test_loader, 'mnist', final_model, 0.1, r_list, num_noise, device)
        
    return test_acc, test_loss

def z_sensitivity(loader, model, var, device):
    
    z_relative_diff = 0
    tested = 0
    
    for x,y in loader:
        x, y = x.to(device), y.to(device)
        
        noise = var**0.5 * torch.randn_like(x)
        
        z = model.return_z(x)
        z_delta = model.return_z(x+noise)
        z_diff_norm = torch.norm(z_delta-z, p=2, dim=1)
        z_norm = torch.norm(z, p=2, dim=1)
        z_relative_diff += (z_diff_norm/z_norm).sum().clone().detach().item()
        tested += x.shape[0]

    return z_relative_diff/tested

def z_correlation(loader, model, device):
    
#     z_relative_diff = 0
#     tested = 0
    
    for i, (x,y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
#         noise = var**0.5 * torch.randn_like(x)
        
        z = model.return_z(x)
#         ipdb.set_trace()
        if i ==0:
            z_holder = z
        else:
            z_holder = torch.cat([z_holder.clone().detach().cpu(), z.clone().detach().cpu()], dim = 0)
        if z_holder.shape[0]>=1000:
            z_holder = z_holder[:1000,:].detach().cpu().numpy()
            break
#         z_delta = model.return_z(x+noise)
#         z_diff_norm = torch.norm(z_delta-z, p=2, dim=1)
#         z_norm = torch.norm(z, p=2, dim=1)
#         z_relative_diff += (z_diff_norm/z_norm).sum().clone().detach().item()
#         tested += x.shape[0]
#     ipdb.set_trace()
    
    cc = np.corrcoef(z_holder, y=None, rowvar=False)
#     cc_isnan = np.isnan(cc)
    cc_nan_removed = np.nan_to_num(cc, copy=True, nan=0.0)
    cc_upper_tri = np.triu(cc_nan_removed)
    return np.abs(cc_upper_tri).sum()
        
           
def return_w(model):

    w_list = torch.empty(0)
    for param in model.parameters():
        w_list = torch.cat([param.data.detach().cpu().flatten(), w_list])
    return w_list

def return_Lipschitz_resnet_L2(model):
    
    L = torch.tensor(1.)
    
    U0 = model.conv1.weight.reshape(64, -1)
#     print(U0.shape)
    L *=torch.linalg.svd(U0.cpu())[1].max()
#     print('conv0: {:e}'.format(L))

    # layer1:
    __layer1_0_U1 = model.layer1[0].conv1.weight.detach().clone().cpu().view(64,-1)
    L_layer1_0_U1 = torch.linalg.svd(__layer1_0_U1)[1].max()
    __layer1_0_U2 = model.layer1[0].conv2.weight.detach().clone().cpu().view(64,-1)
    L_layer1_0_U2 = torch.linalg.svd(__layer1_0_U2)[1].max()
    __layer1_1_U1 = model.layer1[1].conv1.weight.detach().clone().cpu().view(64,-1)
    L_layer1_1_U1 = torch.linalg.svd(__layer1_1_U1)[1].max()
    __layer1_1_U2 = model.layer1[1].conv2.weight.detach().clone().cpu().view(64,-1)
    L_layer1_1_U2 = torch.linalg.svd(__layer1_1_U2)[1].max()
    
    L_layer1 = L_layer1_0_U1*L_layer1_0_U2+1
    L_layer1 *= (L_layer1_1_U1*L_layer1_1_U2+1)
    L *= L_layer1
#     print('L_layer1: {:e}'.format(L_layer1))
    # layer2:
    __layer2_0_U1 = model.layer2[0].conv1.weight.detach().clone().cpu().view(64,-1)
    L_layer2_0_U1 = torch.linalg.svd(__layer2_0_U1)[1].max()
    __layer2_0_U2 = model.layer2[0].conv2.weight.detach().clone().cpu().view(128,-1)
    L_layer2_0_U2 = torch.linalg.svd(__layer2_0_U2)[1].max()
    __layer2_shortcut = model.layer2[0].shortcut[0].weight.detach().clone().cpu().view(64,-1)
    L_layer2_shortcut = torch.linalg.svd(__layer2_shortcut)[1].max()
    __layer2_1_U1 = model.layer2[1].conv1.weight.detach().clone().cpu().view(128,-1)
    L_layer2_1_U1 = torch.linalg.svd(__layer2_1_U1)[1].max()
    __layer2_1_U2 = model.layer2[1].conv2.weight.detach().clone().cpu().view(128,-1)
    L_layer2_1_U2 = torch.linalg.svd(__layer2_1_U2)[1].max()
    
    L_layer2 = L_layer2_0_U1*L_layer2_0_U2+L_layer2_shortcut
    L_layer2 *= (L_layer2_1_U1*L_layer2_1_U2+1)
    L *= L_layer2
#     print('L_layer2: {:e}'.format(L_layer2))
    
    # layer3:
    __layer3_0_U1 = model.layer3[0].conv1.weight.detach().clone().cpu().view(128,-1)
    L_layer3_0_U1 = torch.linalg.svd(__layer3_0_U1)[1].max()
    __layer3_0_U2 = model.layer3[0].conv2.weight.detach().clone().cpu().view(256,-1)
    L_layer3_0_U2 = torch.linalg.svd(__layer3_0_U2)[1].max()
    __layer3_shortcut = model.layer3[0].shortcut[0].weight.detach().clone().cpu().view(128,-1)
    L_layer3_shortcut = torch.linalg.svd(__layer3_shortcut)[1].max()
    __layer3_1_U1 = model.layer3[1].conv1.weight.detach().clone().cpu().view(256,-1)
    L_layer3_1_U1 = torch.linalg.svd(__layer3_1_U1)[1].max()
    __layer3_1_U2 = model.layer3[1].conv2.weight.detach().clone().cpu().view(256,-1)
    L_layer3_1_U2 = torch.linalg.svd(__layer3_1_U2)[1].max()
    
    L_layer3 = L_layer3_0_U1*L_layer3_0_U2+L_layer3_shortcut
    L_layer3 *= (L_layer3_1_U1*L_layer3_1_U2+1)
    L *= L_layer3
#     print('L_layer3: {:e}'.format(L_layer3))
    
    # layer4:
    __layer4_0_U1 = model.layer4[0].conv1.weight.detach().clone().cpu().view(256,-1)
    L_layer4_0_U1 = torch.linalg.svd(__layer4_0_U1)[1].max()
    __layer4_0_U2 = model.layer4[0].conv2.weight.detach().clone().cpu().view(512,-1)
    L_layer4_0_U2 = torch.linalg.svd(__layer4_0_U2)[1].max()
    __layer4_shortcut = model.layer4[0].shortcut[0].weight.detach().clone().cpu().view(256,-1)
    L_layer4_shortcut = torch.linalg.svd(__layer4_shortcut)[1].max()
    __layer4_1_U1 = model.layer4[1].conv1.weight.detach().clone().cpu().view(512,-1)
    L_layer4_1_U1 = torch.linalg.svd(__layer4_1_U1)[1].max()
    __layer4_1_U2 = model.layer4[1].conv2.weight.detach().clone().cpu().view(512,-1)
    L_layer4_1_U2 = torch.linalg.svd(__layer4_1_U2)[1].max()
    
    L_layer4 = L_layer4_0_U1*L_layer4_0_U2+L_layer4_shortcut
    L_layer4 *= (L_layer4_1_U1*L_layer4_1_U2+1)
    L *= L_layer4
#     print('L_layer4: {:e}'.format(L_layer4))

    L *=torch.linalg.svd(model.linear.weight.cpu())[1].max()

    return L

def return_Lipschitz_c2_L1(model):
    
    L = torch.tensor(1.)
    # for l1 operator norm, we follow Eq 12.: 
    # first sum over abs value of each individual filter
    # then sum over i (dim=0), 
    # finally followed by taking max over j (dim=1)
    conv1 = model.conv1.weight.clone().detach().cpu()
    conv2 = model.conv2.weight.clone().detach().cpu()
    L *=conv1.abs().sum(dim=[2,3]).sum(dim=0).max()
    L *=conv2.abs().sum(dim=[2,3]).sum(dim=0).max()
    
    # for p = 1, the operator norm is the maximum absolute column sum norm
    fc1 = model.fc1.weight.clone().detach().cpu()
    fc2 = model.fc2.weight.clone().detach().cpu()
    L *=fc1.abs().sum(dim=0).max()
    L *=fc1.abs().sum(dim=0).max()
    
    return L

def return_Lipschitz_c2_L2(model):
    
    L = torch.tensor(1.)
    
    # reinterpreting the weight tensor of a convolutional layer as a matrix
    U1 = model.conv1.weight.reshape(16,-1).clone().detach().cpu()
    U2 = model.conv2.weight.reshape(32,-1).clone().detach().cpu()
    
    # compute the spectral norm of U
    # upper bound of the true spectral norm
    L *=torch.svd(U1, some=True, compute_uv=False)[1].max()
    L *=torch.svd(U2, some=True, compute_uv=False)[1].max()
    
    # for p = 2, the operator norm is given by the largest singular value of the weight matrix
    L *=torch.svd(model.fc1.weight.clone().detach().cpu(), some=True, compute_uv=False)[1].max()
    L *=torch.svd(model.fc2.weight.clone().detach().cpu(), some=True, compute_uv=False)[1].max()
    
    return L

def return_Lipschitz_resnet_L1(model):
    
    L = torch.tensor(1.)
    
    U0 = model.conv1.weight.reshape(64, -1)
    L *=torch.linalg.svd(U0.cpu())[1].max()
#     print('conv0: {:e}'.format(L))

    # layer1:
    __layer1_0_U1 = model.layer1[0].conv1.weight.detach().clone().cpu()
    L_layer1_0_U1 = __layer1_0_U1.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer1_0_U2 = model.layer1[0].conv2.weight.detach().clone().cpu()
    L_layer1_0_U2 = __layer1_0_U2.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer1_1_U1 = model.layer1[1].conv1.weight.detach().clone().cpu()
    L_layer1_1_U1 = __layer1_1_U1.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer1_1_U2 = model.layer1[1].conv2.weight.detach().clone().cpu()
    L_layer1_1_U2 = __layer1_1_U2.abs().sum(dim=[2,3]).sum(dim=0).max()
    
    L_layer1 = L_layer1_0_U1*L_layer1_0_U2+1
    L_layer1 *= (L_layer1_1_U1*L_layer1_1_U2+1)
    L *= L_layer1
#     print('L_layer1: {:e}'.format(L_layer1))
    
    # layer2:
    __layer2_0_U1 = model.layer2[0].conv1.weight.detach().clone().cpu()
    L_layer2_0_U1 = __layer2_0_U1.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer2_0_U2 = model.layer2[0].conv2.weight.detach().clone().cpu()
    L_layer2_0_U2 = __layer2_0_U2.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer2_shortcut = model.layer2[0].shortcut[0].weight.detach().clone().cpu()
    L_layer2_shortcut = __layer2_shortcut.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer2_1_U1 = model.layer2[1].conv1.weight.detach().clone().cpu()
    L_layer2_1_U1 = __layer2_1_U1.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer2_1_U2 = model.layer2[1].conv2.weight.detach().clone().cpu()
    L_layer2_1_U2 = __layer2_1_U2.abs().sum(dim=[2,3]).sum(dim=0).max()
    
    L_layer2 = L_layer2_0_U1*L_layer2_0_U2+L_layer2_shortcut
    L_layer2 *= (L_layer2_1_U1*L_layer2_1_U2+1)
    L *= L_layer2
#     print('L_layer2: {:e}'.format(L_layer2))
    
    # layer3:
    __layer3_0_U1 = model.layer3[0].conv1.weight.detach().clone().cpu()
    L_layer3_0_U1 = __layer3_0_U1.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer3_0_U2 = model.layer3[0].conv2.weight.detach().clone().cpu()
    L_layer3_0_U2 = __layer3_0_U2.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer3_shortcut = model.layer3[0].shortcut[0].weight.detach().clone().cpu()
    L_layer3_shortcut = __layer3_shortcut.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer3_1_U1 = model.layer3[1].conv1.weight.detach().clone().cpu()
    L_layer3_1_U1 = __layer3_1_U1.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer3_1_U2 = model.layer3[1].conv2.weight.detach().clone().cpu()
    L_layer3_1_U2 = __layer3_1_U2.abs().sum(dim=[2,3]).sum(dim=0).max()
    
    L_layer3 = L_layer3_0_U1*L_layer3_0_U2+L_layer3_shortcut
    L_layer3 *= (L_layer3_1_U1*L_layer3_1_U2+1)
    L *= L_layer3
#     print('L_layer3: {:e}'.format(L_layer3))
    
    # layer4:
    __layer4_0_U1 = model.layer4[0].conv1.weight.detach().clone().cpu()
    L_layer4_0_U1 = __layer4_0_U1.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer4_0_U2 = model.layer4[0].conv2.weight.detach().clone().cpu()
    L_layer4_0_U2 = __layer4_0_U2.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer4_shortcut = model.layer4[0].shortcut[0].weight.detach().clone().cpu()
    L_layer4_shortcut = __layer4_shortcut.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer4_1_U1 = model.layer4[1].conv1.weight.detach().clone().cpu()
    L_layer4_1_U1 = __layer4_1_U1.abs().sum(dim=[2,3]).sum(dim=0).max()
    __layer4_1_U2 = model.layer4[1].conv2.weight.detach().clone().cpu()
    L_layer4_1_U2 = __layer4_1_U2.abs().sum(dim=[2,3]).sum(dim=0).max()
    
    L_layer4 = L_layer4_0_U1*L_layer4_0_U2+L_layer4_shortcut
    L_layer4 *= (L_layer4_1_U1*L_layer4_1_U2+1)
    L *= L_layer4
#     print('L_layer4: {:e}'.format(L_layer4))

    L *=model.linear.weight.cpu().abs().sum(dim=0).max()

    return L

def return_Lipschitz(model, norm, dataset):
    if norm == 2:
        if dataset in ['mnist', 'fashionmnist']:
            return return_Lipschitz_c2_L2(model)
        else:
            return return_Lipschitz_resnet_L2(model)
    elif norm == 1:
        if dataset in ['mnist', 'fashionmnist']:
            return return_Lipschitz_c2_L1(model)
        else:
            return return_Lipschitz_resnet_L1(model)