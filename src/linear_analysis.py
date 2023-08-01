import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.utils_freq import batch_dct,dct, idct, getDCTmatrix, batch_idct

from collections import defaultdict
from tqdm import trange
import ipdb

from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings("ignore")


def loader_LR(w_tilde_star, total_size, sigma_tilde, d, batchsize):
        
    x, y = data_init_LR(w_tilde_star, sigma_tilde, d, total_size)
    dataset = TensorDataset(x.t(), y.t())
    loader = DataLoader(dataset, batch_size = batchsize, pin_memory = True)
    
    return loader

def data_init_LR(w_tilde_star, sigma_tilde, d, total_size = 1000):
    
    x_tilde = torch.zeros(d, total_size)
    x_tilde[0,:] = torch.normal(mean = 0, std = sigma_tilde[0].item(), size = (1, total_size))
    x_tilde[1,:] = torch.normal(mean = 0, std = sigma_tilde[1].item(), size = (1, total_size))
    x_tilde[2,:] = torch.normal(mean = 0, std = sigma_tilde[2].item(), size = (1, total_size))
    x = idct(x_tilde)
    
    y = torch.mm(w_tilde_star.t(), x_tilde)
    return x, y

def isNaNCheck(x,y):
    if torch.isnan(x).sum().item() != 0:
        print("NaN detected in data: removed", torch.isnan(x).sum().item(), "datapoints")
        nonNan_idx = torch.tensor(1-torch.isnan(x).sum(dim=0), dtype = torch.bool)
        x = x[:,nonNan_idx]
        y = y[nonNan_idx]
        ipdb.set_trace()

def returnTestLoss(model,test_loader,device):
    mseloss = torch.nn.MSELoss(reduction='mean')
    
    for x_test,y_test in test_loader:
        x_test, y_test = x_test.t().to(device), y_test.t().to(device)
        break
    y_hat_test = model(x_test)
#                 pop_loss = ((1/2) * (y_hat_test.t() - y_test) ** 2).mean()
    pop_loss = (1/2)*mseloss(y_hat_test.t(), y_test)
    return pop_loss.item()

# def train_LR_v3(args, train_loader, test_loader, model, opt, signGD, infinite_data, device):
def train_LR_v3(args, w_tilde_star, sigma_tilde, model, opt, signGD, device):
    
    iteration = args["itr"]
    _d = args["d"]
    _case = args["case"]
    _batchsize = args["bsize"]
    _mu = args["mu"]
    _std = args["std"]
#     _lambbda = args["lambbda"]
    _lr = args["lr"]
    
    log_dict = defaultdict(lambda: list())
    
    w = torch.zeros(_d, iteration, device = device)
    loss_logger = torch.zeros(1, iteration, device = device)
    pop_loss_logger = torch.zeros(1, iteration, device = device)
    
    mseloss = torch.nn.MSELoss(reduction='mean')
    
    i = 0
    while i < iteration:
#         if infinite_data: # basically we over-write the traiin_loader input param by initializing a new one every iteration
        train_loader = loader_LR(w_tilde_star, 5000, sigma_tilde, _d, 5000)
        test_loader = loader_LR(w_tilde_star, 100, sigma_tilde, _d, 100)
        
        for x, y in train_loader:
            w[:,i] = model.state_dict()['linear.weight'].squeeze().detach()

            x, y = x.t().to(device), y.t().to(device)
            isNaNCheck(x,y)
            opt.zero_grad()

            y_hat = model(x)
            loss = (1/2)*mseloss(y_hat.t(), y)
            loss_logger[:,i] = loss.item()
            loss.backward()
            
            pop_loss_logger[:,i] = returnTestLoss(model, test_loader, device)
            
            if not signGD:
                opt.step()
            else:
                curr_w = model.linear.weight.clone().detach()
                grad = model.linear.weight.grad.clone().detach()
                
                new_w = curr_w - opt.param_groups[0]['lr'] * torch.sign(grad)
                model.linear.weight = torch.nn.parameter.Parameter(new_w)

            i += 1
            if i >= iteration:
                break
            elif i % 200 ==0:
                print(i)
            
    log_dict["w"] = w
    log_dict["loss"] = loss_logger
    log_dict["pop_loss"] = pop_loss_logger
    return log_dict

# def grad_estimation(w_tilde, lr, case, d, mu, std, lambbda):
    
#     grad_est = torch.zeros(d,1, device= w_tilde.device)
    
#     if case == 1:
#         grad_est[0] = lr*(w_tilde[0]-1)*(mu**2 + std**2)
        
#     elif case == 2:
#         grad_est[0] = lr*(w_tilde[1:].sum()*mu**2 + (w_tilde[0]-1)*(mu**2 + std**2))
        
#         for i in range(1, d):
#             grad_est[i] = lr*(w_tilde.sum()*mu**2 - w_tilde[i]*mu**2 + w_tilde[i]*(mu**2 + std**2) - mu**2)
            
#     elif case == 3:
#         exp_i = torch.exp(-torch.tensor(lambbda, device = w_tilde.device)*(torch.range(1, d, device = w_tilde.device)-1))
#         grad_est[0] = lr*( (mu**2 * w_tilde * exp_i)[1:].sum() + (w_tilde[0]-1)*(mu**2 + std**2))
        
#         for i in range(1, d):
#             exp_k = torch.exp(-torch.tensor(lambbda, dtype = torch.float32, device = w_tilde.device)*torch.tensor(i))
#             grad_est[i] = lr*(((w_tilde*exp_i*exp_k).sum() - w_tilde[i]*exp_k**2)*mu**2 + w_tilde[i]*exp_k**2*(mu**2 + std**2) - mu**2*exp_k)

#     return grad_est

# def train_LR(args, model, opt, device):
    
#     iteration = args["itr"]
#     _d = args["d"]
#     _case = args["case"]
#     _batchsize = args["bsize"]
#     _mu = args["mu"]
#     _std = args["std"]
#     _lambbda = args["lambbda"]
#     _lr = args["lr"]
#     _method = args['method']
    
#     log_dict = defaultdict(lambda: list())
    
#     w = torch.zeros(_d, iteration, device = device)
#     w_tilde =  torch.zeros(_d, iteration, device = device)
#     dw_tilde =  torch.zeros(2, _d, iteration, device = device)
#     loss_logger = torch.zeros(1, iteration, device = device)
#     loss_adv_logger = torch.zeros(3, iteration, device = device) # 3 types of attacks
    
    
#     train_loader = loader_LR(_case, _d, _batchsize, _mu, _std, _lambbda, iteration)
#     dct_matrix = getDCTmatrix(_d)
    
#     i = 0
#     for x, y in train_loader:

#         x, y = x.t().to(device), y.t().to(device)
#         if torch.isnan(x).sum().item() != 0:
#             print("NaN detected in data: removed", torch.isnan(x).sum().item(), "datapoints")
#             nonNan_idx = torch.tensor(1-torch.isnan(x).sum(dim=0), dtype = torch.bool)
#             x = x[:,nonNan_idx]
#             y = y[nonNan_idx]
#             ipdb.set_trace()
        
#         w_tilde_prev = dct(model.state_dict()['linear.weight'].view(_d,1)).squeeze().detach()

#         opt.zero_grad()

#         y_hat = model(x)

#         loss = ((1/2) * (y_hat.t() - y) ** 2).mean()

#         loss_logger[:,i] = loss.item()
    
#         loss_adv = loss_under_attack(args, model, device)

#         if _method =='weighted_l1f':
#             factor = args['factor']
# #             ipdb.set_trace()
#             curr_w = model.linear.weight.t()
#             curr_w_tilde = dct(curr_w)
            
#             AVOID_ZERO_DIV = 1e-6
#             mean_abs_x_tilde = batch_dct(x.t(), dct_matrix).abs().mean(dim=0)
#             decay_factor = mean_abs_x_tilde/mean_abs_x_tilde[0]
#             M = (1/(decay_factor+AVOID_ZERO_DIV)).view(1,_d)
            
#             weighted_w_tilde = torch.mul(M, curr_w_tilde).squeeze()
            
#             l1_reg = torch.norm(weighted_w_tilde,p=1)
            
#             loss_reg = loss+factor*l1_reg 
#             loss_reg.backward()
#             opt.step()
#         elif _method =='l1f':
#             factor = args['factor']
# #             ipdb.set_trace()
#             curr_w = model.linear.weight.t()
#             curr_w_tilde = dct(curr_w)
            
#             AVOID_ZERO_DIV = 1e-6
#             mean_abs_x_tilde = batch_dct(x.t(), dct_matrix).abs().mean(dim=0)
#             decay_factor = mean_abs_x_tilde/mean_abs_x_tilde[0]
#             M = (1/(decay_factor+AVOID_ZERO_DIV)).view(1,_d)
            
#             M = torch.ones_like(M, device = device)
            
#             weighted_w_tilde = torch.mul(M, curr_w_tilde).squeeze()
            
#             l1_reg = torch.norm(weighted_w_tilde,p=1)
            
#             loss_reg = loss+factor*l1_reg 
#             loss_reg.backward()
#             opt.step()
#         elif _method == 'l1s':
#             factor = args['factor']
#             curr_w = model.linear.weight.squeeze()
#             l1_reg = torch.norm(curr_w,p=1)
#             loss_reg = loss+factor*l1_reg 
#             loss_reg.backward()
#             opt.step()
#         else:
#             loss.backward()
#             # manual update
#             curr_w = model.linear.weight.clone().detach()
#             grad = model.linear.weight.grad.clone().detach()

#             if _method == "weighted_lr":
#                 dct_grad = dct(grad.t())
#                 AVOID_ZERO_DIV = 1e-6
#                 mean_abs_x_tilde = batch_dct(x.t(), dct_matrix).abs().mean(dim=0)
#                 decay_factor = mean_abs_x_tilde/mean_abs_x_tilde[0]
#                 M = (1/(decay_factor+AVOID_ZERO_DIV)).view(_d, 1)
#     #             M = torch.ones_like(M, device = x.device) # for sanity check
#                 new_w = curr_w - idct(_lr * torch.mul(M,dct_grad)).t()
#             else:
#                 new_w = curr_w - _lr * grad

#             model.linear.weight = torch.nn.parameter.Parameter(new_w)

#         w[:,i] = model.state_dict()['linear.weight'].squeeze().detach()
#         w_tilde[:,i] = dct(w[:,i].view(_d,1)).squeeze().detach()
#         dw_tilde[0,:,i] = -(w_tilde[:,i] - w_tilde_prev).detach()
#         dw_tilde[1,:,i] = grad_estimation(w_tilde_prev, 
#                                           _lr, 
#                                           _case, 
#                                           _d, 
#                                           _mu, 
#                                           _std, 
#                                           _lambbda).squeeze().detach()
#         loss_adv_logger[:,i] = loss_adv
        
#         i += 1
            
#     log_dict["w"] = w
#     log_dict["w_tilde"] = w_tilde
#     log_dict["dw_tilde"] = dw_tilde
#     log_dict["loss"] = loss_logger
#     log_dict["loss_adv"] = loss_adv_logger

#     return log_dict

# def loss_under_attack(args, model, device):
    
#     loss_adv = torch.zeros(3, device = device) # 3 types of attacks
#     _d = args["d"]
#     _case = args["case"]
#     _batchsize = args["bsize"]
#     _mu = args["mu"]
#     _std = args["std"]
#     _lambbda = args["lambbda"]
#     _lr = args["lr"]
#     _eps = args["eps"]
    
#     x, y = data_init_LR(_case, _d, _batchsize, _mu, _std, _lambbda) 
#     x, y = x.to(device), y.to(device)
    
    
#     y_hat = model(x)
#     r = (y_hat.t() - y)
    
#     w = model.state_dict()['linear.weight'].squeeze().detach()
#     w_tilde = dct(w.view(_d,1)).detach()
    

# #     #attack 1: all k's
# #     delta_x_1_tilde = _eps * torch.sign(r) * w_tilde / torch.norm(w_tilde).detach()
# #     delta_x_1 = idct(delta_x_1_tilde)
# #     y_adv_1 = model(x+delta_x_1)
# #     loss_1 = ((1/2) * (y_adv_1.t() - y) ** 2).mean().item()
    
# #     #attack 2: highest k
# #     attack_k = torch.ones(_d, device = device)
# #     attack_k[0:-1] = 0
# #     V = torch.diag(attack_k)
# #     V_w_tilde = torch.mm(V, w_tilde)
# #     delta_x_2_tilde = _eps * torch.sign(r) * V_w_tilde / torch.norm(V_w_tilde).detach()
# #     delta_x_2 = idct(delta_x_2_tilde)
# #     y_adv_2 = model(x+delta_x_2)
# #     loss_2 = ((1/2) * (y_adv_2.t() - y) ** 2).mean().item()
    
#     #attack 3: k != 0
#     attack_k = torch.ones(_d, device = device)
#     attack_k[0] = 0
#     V = torch.diag(attack_k)
#     V_w_tilde = torch.mm(V, w_tilde)
# #     ipdb.set_trace()
# #     if _case == 1:
# #         print(w_tilde)
#     delta_x_3_tilde = _eps * torch.sign(r) * V_w_tilde / torch.norm(V_w_tilde).detach()
#     delta_x_3 = idct(delta_x_3_tilde)
#     y_adv_3 = model(x+delta_x_3)
#     loss_3 = ((1/2) * (y_adv_3.t() - y) ** 2).mean().item()
    
    
#     loss_adv[0], loss_adv[1], loss_adv[2] = 0, 0, loss_3
    
#     return loss_adv
    
def plot_loss_LR(log, threshold = 1e-6, plot_itr = 1000, xscale = 'linear'):
    
    THRESHOLD = threshold
    
    fix, axs = plt.subplots(nrows = 1, ncols=1, figsize=(15.5, 4))
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(1,1)
    
#     loss_var, loss_mean  = torch.var_mean(log, dim = 1) 
#     fill_up = loss_mean + loss_var
#     fill_low = loss_mean - loss_var

#     xrange = np.arange(log.shape[0])  
    
#     loss_below_threshold = loss_mean < THRESHOLD
    loss_below_threshold = log < THRESHOLD
    
    
#     for i in range(log.shape[1]):
#         print(i)
#         ipdb.set_trace()
    
    axs.plot(log[:plot_itr, 0], color = "C"+str(0), linewidth=3.0, marker = "", label='GD')
    axs.plot(log[:plot_itr, 1], color = "C"+str(1), linewidth=3.0, marker = "", label='Adam')
    axs.plot(log[:plot_itr, 2], color = "C"+str(2), linewidth=3.0, marker = "", label='signGD', alpha=0.5)
#         fig.add_subplot(gs[0,0]).set_xscale(xscale)
#         fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
        
    try:
        axs.axvline(x=loss_below_threshold[:plot_itr,0].tolist().index(1), color = "C"+str(0), linestyle = '--')
    except ValueError as e:
        print("Above loss threshold! (loss plot)")
        
    try:
        axs.axvline(x=loss_below_threshold[:plot_itr,1].tolist().index(1), color = "C"+str(1), linestyle = '--')
    except ValueError as e:
        print("Above loss threshold! (loss plot)")
        
#     try:
#         axs.axvline(x=loss_below_threshold[:plot_itr,2].tolist().index(1), color = "C"+str(2), linestyle = '--')
#     except ValueError as e:
#         print("Above loss threshold! (loss plot)")
            
#     fig.add_subplot(gs[0,0]).set_xlim([0, plot_itr])
    axs.legend(prop={"size": 10})
    axs.set_ylabel("loss")
    axs.set_xlabel("Training iteration")
    axs.grid()
#     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)

def plot_w_tilde_LR(log, threshold = 1e-3, plot_itr = None, robust_w = None):
    
    THRESHOLD = threshold
    fix, axs = plt.subplots(ncols = 6 if log.shape[1] >= 6 else log.shape[1], nrows=2, figsize=(20, 4))

    w_log = log.clone().detach()
    w_tilde_log = torch.zeros_like(w_log)
    num_run = w_tilde_log.shape[0]
    fill_scale = 0.3
    
    for _run in range(num_run):
#         ipdb.set_trace()
        w_tilde_log[_run,:,:,0] = batch_dct(w_log[_run,:,:,0].t(), getDCTmatrix(w_log.shape[1])).t()
        w_tilde_log[_run,:,:,1] = batch_dct(w_log[_run,:,:,1].t(), getDCTmatrix(w_log.shape[1])).t()
        w_tilde_log[_run,:,:,2] = batch_dct(w_log[_run,:,:,2].t(), getDCTmatrix(w_log.shape[1])).t()
        w_tilde_log[_run,:,:,3] = batch_dct(w_log[_run,:,:,3].t(), getDCTmatrix(w_log.shape[1])).t()
        if robust_w is not None:
            w_log[_run,:,:,:] = w_log[_run,:,:,:] - robust_w[0].view(w_log.shape[1],1,1).repeat(1,w_log.shape[2],4)
            w_tilde_log[_run,:,:,:] = w_tilde_log[_run,:,:,:] - robust_w[1].view(w_log.shape[1],1,1).repeat(1,w_log.shape[2],4)
            plotted = 'e'
        else:
            plotted = 'w'
#     ipdb.set_trace()
    w_var, w_mean  = torch.var_mean(w_log, dim = 0, unbiased = True)
    w_var = torch.sqrt(w_var/num_run)*fill_scale
    w_fill_up, w_fill_low = w_mean+w_var, w_mean-w_var
    
    w_tilde_var, w_tilde_mean  = torch.var_mean(w_tilde_log, dim = 0, unbiased = True)
    w_tilde_var = torch.sqrt(w_tilde_var/num_run)*fill_scale
    w_tilde_fill_up, w_tilde_fill_low = w_tilde_mean+w_tilde_var, w_tilde_mean-w_tilde_var
    xrange = np.arange(w_tilde_mean.shape[1]) 

    label = ['GD', 'Adam','RMSProp','SignGD']
    
    if plot_itr is None:
        plot_iter = w_log.shape[2]
    for i in [0,1,2,3]:
        for j in range(6 if w_log.shape[1] >= 6 else w_log.shape[1]):
            axs[0,j].plot(w_mean[j,:plot_itr,i], color = "C"+str(i), linewidth=3.0, marker = "", label=label[i], alpha = 0.5 if i==2 else 1)
#             axs[0,j].set_ylabel("$"+plotted+"("+str(j+1)+")$", rotation=0, labelpad=20, fontsize=15)
            axs[0,j].set_ylabel(r"$"+plotted+"_{"+str(j)+"}(t)$", rotation=0, labelpad=20, fontsize=15)
            axs[1,j].plot(w_tilde_mean[j,:plot_itr,i], color = "C"+str(i), linewidth=3.0, marker = "", label=label[i], alpha = 0.5 if i==2 else 1)
#             axs[1,j].set_ylabel(r"$\tilde{"+plotted+"}("+str(j+1)+")$", rotation=0, labelpad=20, fontsize=15)
            axs[1,j].set_ylabel(r"$\tilde{"+plotted+"}_{"+str(j)+"}(t)$", rotation=0, labelpad=20, fontsize=15)
            
#             axs[0,j].tick_params(axis="x", labelsize=15)
#             axs[0,j].tick_params(axis="y", labelsize=15)
            axs[0,j].tick_params(axis="both", labelsize=12)
            axs[1,j].tick_params(axis="both", labelsize=12)
#             axs[1,j].tick_params(axis="y", labelsize=15)
#             ipdb.set_trace()
            axs[0,j].fill_between(xrange, w_fill_up[j,:plot_itr,i], w_fill_low[j,:plot_itr,i], color = "C"+str(i), alpha=0.3)
            axs[1,j].fill_between(xrange, w_tilde_fill_up[j,:plot_itr,i], w_tilde_fill_low[j,:plot_itr,i], color = "C"+str(i), alpha=0.3)
#             try:
#                 axs[j].axvline(x=error_tilde_diff_below_threshold[j,:plot_itr,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#             except ValueError as e:
#                 print("Above loss threshold! (w_tilde plot) ", i, j)
            
    
#     fig.add_subplot(gs[0,0]).set_title("$\~e(k)$",fontsize = 20)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 8})
# #     fig.add_subplot(gs[5,0]).set_ylabel("Frequency")
    
#     axs[-1,1].set_xlabel("Training iteration (i)",fontsize=15)
#     axs[-1,2].set_xlabel("Training iteration (i)",fontsize=15)

    axs[0,0].legend(fontsize=13)
#     axs[0].legend()
    for i in range(w_log.shape[1]):
#     for i in range(6 if w_log.shape[0] >= 6 else w_log.shape[0]):
        axs[0,i].grid()
        axs[-1,i].set_xlabel("Training iteration (t)",fontsize=15)
        axs[1,i].grid()
#         axs[2,i].grid()
    fix.tight_layout()
    
def plot_w_tilde_LR_freq_only(log, threshold = 1e-3, plot_itr = None, robust_w = None, save=None):
    
    THRESHOLD = threshold
    fix, axs = plt.subplots(ncols = 6 if log.shape[1] >= 6 else log.shape[1], nrows=1, figsize=(20, 4))

    w_log = log.clone().detach()
    w_tilde_log = torch.zeros_like(w_log)
    num_run = w_tilde_log.shape[0]
    fill_scale = 0.3
    
    for _run in range(num_run):
#         ipdb.set_trace()
        w_tilde_log[_run,:,:,0] = batch_dct(w_log[_run,:,:,0].t(), getDCTmatrix(w_log.shape[1])).t()
        w_tilde_log[_run,:,:,1] = batch_dct(w_log[_run,:,:,1].t(), getDCTmatrix(w_log.shape[1])).t()
        w_tilde_log[_run,:,:,2] = batch_dct(w_log[_run,:,:,2].t(), getDCTmatrix(w_log.shape[1])).t()
        w_tilde_log[_run,:,:,3] = batch_dct(w_log[_run,:,:,3].t(), getDCTmatrix(w_log.shape[1])).t()
        if robust_w is not None:
            w_log[_run,:,:,:] = w_log[_run,:,:,:] - robust_w[0].view(w_log.shape[1],1,1).repeat(1,w_log.shape[2],4)
            w_tilde_log[_run,:,:,:] = w_tilde_log[_run,:,:,:] - robust_w[1].view(w_log.shape[1],1,1).repeat(1,w_log.shape[2],4)
            plotted = 'e'
        else:
            plotted = 'w'
#     ipdb.set_trace()
    w_var, w_mean  = torch.var_mean(w_log, dim = 0, unbiased = True)
    w_var = torch.sqrt(w_var/num_run)*fill_scale
    w_fill_up, w_fill_low = w_mean+w_var, w_mean-w_var
    
    w_tilde_var, w_tilde_mean  = torch.var_mean(w_tilde_log, dim = 0, unbiased = True)
    w_tilde_var = torch.sqrt(w_tilde_var/num_run)*fill_scale
    w_tilde_fill_up, w_tilde_fill_low = w_tilde_mean+w_tilde_var, w_tilde_mean-w_tilde_var
    xrange = np.arange(w_tilde_mean.shape[1]) 

    label = ['GD', 'Adam','RMSProp','SignGD']
    
    if plot_itr is None:
        plot_iter = w_log.shape[2]
    for i in [0,1,2,3]:
        for j in range(6 if w_log.shape[1] >= 6 else w_log.shape[1]):
#             axs[0,j].plot(w_mean[j,:plot_itr,i], color = "C"+str(i), linewidth=3.0, marker = "", label=label[i], alpha = 0.5 if i==2 else 1)
#             axs[0,j].set_ylabel("$"+plotted+"("+str(j+1)+")$", rotation=0, labelpad=20, fontsize=15)
#             axs[0,j].set_ylabel(r"$"+plotted+"_{"+str(j)+"}(t)$", rotation=0, labelpad=20, fontsize=15)
            if j ==0:
                axs[j].plot(w_tilde_mean[j,:plot_itr,i], color = "C"+str(i), linewidth=3.0, marker = "", label=label[i], alpha = 0.5 if i==2 else 1)
            else:
                axs[j].plot(w_tilde_mean[j,:plot_itr,i], color = "C"+str(i), linewidth=3.0, marker = "", alpha = 0.5 if i==2 else 1)
#             axs[1,j].set_ylabel(r"$\tilde{"+plotted+"}("+str(j+1)+")$", rotation=0, labelpad=20, fontsize=15)
            axs[j].set_ylabel(r"$\tilde{"+plotted+"}_{"+str(j)+"}(t)$", rotation=0, labelpad=20, fontsize=15)
            
#             axs[0,j].tick_params(axis="x", labelsize=15)
#             axs[0,j].tick_params(axis="y", labelsize=15)
#             axs[0,j].tick_params(axis="both", labelsize=12)
            axs[j].tick_params(axis="both", labelsize=15)
#             axs[1,j].tick_params(axis="y", labelsize=15)
#             ipdb.set_trace()
#             axs[0,j].fill_between(xrange, w_fill_up[j,:plot_itr,i], w_fill_low[j,:plot_itr,i], color = "C"+str(i), alpha=0.3)
            axs[j].fill_between(xrange, w_tilde_fill_up[j,:plot_itr,i], w_tilde_fill_low[j,:plot_itr,i], color = "C"+str(i), alpha=0.3)
#             try:
#                 axs[j].axvline(x=error_tilde_diff_below_threshold[j,:plot_itr,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#             except ValueError as e:
#                 print("Above loss threshold! (w_tilde plot) ", i, j)
            
    
#     fig.add_subplot(gs[0,0]).set_title("$\~e(k)$",fontsize = 20)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 8})
# #     fig.add_subplot(gs[5,0]).set_ylabel("Frequency")
    
#     axs[-1,1].set_xlabel("Training iteration (i)",fontsize=15)
#     axs[-1,2].set_xlabel("Training iteration (i)",fontsize=15)

    
#     axs[0].legend()
    for i in range(w_log.shape[1]):
#     for i in range(6 if w_log.shape[0] >= 6 else w_log.shape[0]):
        axs[i].grid()
        axs[i].set_xlabel("Training iteration (t)",fontsize=15)
#         axs[i].grid()
#         axs[2,i].grid()
#     ipdb.set_trace()
    p0_xlimit, p0_ylimit = axs[0].get_xlim(), axs[0].get_ylim()
    p1_xlimit, p1_ylimit = axs[1].get_xlim(), axs[1].get_ylim()
    p2_xlimit, p2_ylimit = axs[2].get_xlim(), axs[2].get_ylim()
    
    axs[0].legend(fontsize=15)
    axs[0].add_patch(Rectangle((60,-7), -70, 8, alpha = 0.1, color='C2'))
    axs[0].add_patch(Rectangle((60,-7), 100, 8, alpha = 0.1, color='C1'))
    
    axs[1].add_patch(Rectangle((60,-7), -70, 8, alpha = 0.1, color='C2'))
    axs[1].add_patch(Rectangle((60,-7), 100, 8, alpha = 0.1, color='C1'))
    
    axs[2].add_patch(Rectangle((60,-7), -70, 8, alpha = 0.1, color='C2',label=r'$|\tilde{e}_2|$ grows while learning $\tilde{w}_0$ and $\tilde{w}_1$'))
    axs[2].add_patch(Rectangle((60,-7), 100, 8, alpha = 0.1, color='C1',label=r'$|\tilde{e}_2|$ cannot be corrected'))
    axs[2].legend(fontsize=15,loc=(0.155,0.68))
    fix.tight_layout()
    if save is not None:
        plt.savefig('/scratch/ssd001/home/ama/workspace/ama-at-vector/freq-robust/notebook/TMLR/figures/synthetic_v3'+save+'.pdf', bbox_inches='tight')  
    

def x_correlation(x, device):
    
#     z_relative_diff = 0
#     tested = 0
    
#     for i, (x,y) in enumerate(loader):
#         x, y = x.to(device), y.to(device)
        
#         noise = var**0.5 * torch.randn_like(x)
        
#         z = model.return_z(x)
#         ipdb.set_trace()
#         if i ==0:
#             z_holder = z
#         else:
#             z_holder = torch.cat([z_holder.clone().detach().cpu(), z.clone().detach().cpu()], dim = 0)
#         if z_holder.shape[0]>=1000:
#             z_holder = z_holder[:1000,:].detach().cpu().numpy()
#             break
#         z_delta = model.return_z(x+noise)
#         z_diff_norm = torch.norm(z_delta-z, p=2, dim=1)
#         z_norm = torch.norm(z, p=2, dim=1)
#         z_relative_diff += (z_diff_norm/z_norm).sum().clone().detach().item()
#         tested += x.shape[0]
#     ipdb.set_trace()
    
    cc = np.corrcoef(x, y=None, rowvar=False)
#     print(cc)
#     cc_isnan = np.isnan(cc)
#     cc_nan_removed = np.nan_to_num(cc, copy=True, nan=0.0)
    cc_upper_tri = np.triu(cc)
    return np.abs(cc_upper_tri).mean()
        

    
# def plot_risk_LR(args, w_tilde_log, loss_log, threshold = 1e-3, plot_itr = 1000, xscale = 'linear'):
    
#     THRESHOLD = threshold
    
#     _std = args["std"]
#     _lr = args["lr"]
#     _d = args["d"]
#     _lambbda = args["lambbda"]
    
#     w_tilde_log_copy = w_tilde_log.clone().detach()
    
#     fig = plt.figure(figsize = [15,7])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(1,1)
    
#     loss_var, loss_mean  = torch.var_mean(loss_log, dim = 1) 
#     fill_up = loss_mean + loss_var
#     fill_low = loss_mean - loss_var

#     xrange = np.arange(loss_log.shape[0])  
    
#     loss_below_threshold = loss_mean < THRESHOLD
    
    
#     for i in range(loss_log.shape[2]):
#         if i == 3:
#             fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = r"case 3 ($\~\theta_{t+1} = \~\theta_{t} - \eta M \circ \Delta\~\theta_{t}$): loss from sample statistics", linewidth=3.0, linestyle = "-")
#             fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
#         elif i == 4:
#             fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = r"case 3 ($\lambda||M \circ \~\theta||, \lambda =1e-4$): loss from sample statistics", linewidth=3.0, linestyle = "-")
#             fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
#         elif i == 5:
#             fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = r"case 3 ($\lambda||\~\theta||, \lambda =1e-4$): loss from sample statistics", linewidth=3.0, linestyle = "-")
#             fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
#         elif i == 6:
#             fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = r"case 3 ($\lambda||\theta||, \lambda =1e-4$): loss from sample statistics", linewidth=3.0, linestyle = "-")
#             fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
#         else:
#             fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = "case "+str(i+1)+": loss from sample statistics", linewidth=3.0, linestyle = "-")
#             fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
        
#         if i == 0: # case 1
#             e_0 = w_tilde_log_copy[0,0,:,0] - 1 # only supports numb_runs = 1, so error will occur if we do average over multiple runs
#             risk = 0.5 * e_0**2 * _std**2 * torch.tensor(1 - 2*_lr*_std**2 + 3*_lr**2*_std**4)**torch.tensor(xrange)
#             fig.add_subplot(gs[0,0]).plot(risk, color = "C"+str(i), label = "case "+str(i+1)+": loss from population statistics", linewidth=3.0, linestyle = "--")
#         elif i == 1: # case 2
#             e_0 = w_tilde_log_copy[:,0,:,1] # only supports numb_runs = 1, so error will occur if we do average over multiple runs
#             e_0[0] = e_0[0] - 1
#             risk = 0.5 * torch.norm(e_0, p =2)**2 * _std**2 * torch.tensor(1 - 2*_lr*_std**2 + 3*_lr**2*_std**4)**torch.tensor(xrange)
#             fig.add_subplot(gs[0,0]).plot(risk, color = "C"+str(i), label = "case "+str(i+1)+": loss from population statistics", linewidth=3.0, linestyle = "--")
#         elif i ==2: # case 3
#             e_i = w_tilde_log_copy[:,0,:,2]
#             e_i[0] -= 1
#             bracket_term = [np.exp(-2*d*_lambbda)*torch.tensor(1 - 2 * _lr * _std**2 * np.exp(-2*d*_lambbda) + 3 * _lr**2 * _std**4 * np.exp(-4*d*_lambbda))**torch.tensor(xrange) for d in range(_d)]
#             sum_term = torch.stack(bracket_term).T @ (torch.tensor(e_i)**2)
#             risk = 0.5 * _std**2 * sum_term
#             fig.add_subplot(gs[0,0]).plot(risk, color = "C"+str(i), label = "case "+str(i+1)+": loss from population statistics", linewidth=3.0, linestyle = "--")
            
#         fig.add_subplot(gs[0,0]).set_xscale(xscale)

#         try:
#             fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#         except ValueError as e:
#             print("Above loss threshold! (loss plot)")
            
#     fig.add_subplot(gs[0,0]).set_xlim([0, plot_itr])
            
        
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 20})
# #     fig.add_subplot(gs[0,0]).set_ylabel("loss")
#     fig.add_subplot(gs[0,0]).set_xlabel("Training iteration")
# #     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)

    
# def plot_w_tilde_LR(log, threshold = 1e-3):
    
#     THRESHOLD = threshold
    
#     fig = plt.figure(figsize = [15,15])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(10,1)
    
#     w_tilde_log = log.clone().detach()
#     w_tilde_log[0,:,:,:] = w_tilde_log[0,:,:,:]-1
    
#     w_tilde_diff_var, w_tilde_diff_mean  = torch.var_mean(w_tilde_log, dim = 2, unbiased = True)

#     xrange = np.arange(log.shape[1])
    
#     w_tilde_diff_below_threshold = w_tilde_diff_mean.abs() < THRESHOLD
    
#     for i in range(log.shape[3]):
#         for j in range(10):
            
#             if i ==3:
#                 fig.add_subplot(gs[j,0]).plot(w_tilde_diff_mean[j,:,i], color = "C"+str(i), label = "case 3 (weighted lr)", linewidth=3.0, marker = "")
#                 fig.add_subplot(gs[j,0]).set_ylabel("$\~e("+str(j)+")$", rotation=0, labelpad=30)
#     #             fig.add_subplot(gs[j,0]).fill_between(xrange, fill_up[j,:,i-1], fill_low[j,:,i-1], color = "C"+str(i), alpha=0.3)
#                 try:
#                     fig.add_subplot(gs[j,0]).axvline(x=w_tilde_diff_below_threshold[j,:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#                 except ValueError as e:
#                     print("Above loss threshold! (w_tilde plot) ", i, j)
#             elif i ==4:
#                 fig.add_subplot(gs[j,0]).plot(w_tilde_diff_mean[j,:,i], color = "C"+str(i), label = "case 3 (weighted l1 in freq", linewidth=3.0, marker = "")
#                 fig.add_subplot(gs[j,0]).set_ylabel("$\~e("+str(j)+")$", rotation=0, labelpad=30)
#     #             fig.add_subplot(gs[j,0]).fill_between(xrange, fill_up[j,:,i-1], fill_low[j,:,i-1], color = "C"+str(i), alpha=0.3)
#                 try:
#                     fig.add_subplot(gs[j,0]).axvline(x=w_tilde_diff_below_threshold[j,:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#                 except ValueError as e:
#                     print("Above loss threshold! (w_tilde plot) ", i, j)
#             elif i ==5:
#                 fig.add_subplot(gs[j,0]).plot(w_tilde_diff_mean[j,:,i], color = "C"+str(i), label = "case 3 (l1 in freq)", linewidth=3.0, marker = "")
#                 fig.add_subplot(gs[j,0]).set_ylabel("$\~e("+str(j)+")$", rotation=0, labelpad=30)
#     #             fig.add_subplot(gs[j,0]).fill_between(xrange, fill_up[j,:,i-1], fill_low[j,:,i-1], color = "C"+str(i), alpha=0.3)
#                 try:
#                     fig.add_subplot(gs[j,0]).axvline(x=w_tilde_diff_below_threshold[j,:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#                 except ValueError as e:
#                     print("Above loss threshold! (w_tilde plot) ", i, j)
#             elif i ==6:
# #                 ipdb.set_trace()
#                 fig.add_subplot(gs[j,0]).plot(w_tilde_diff_mean[j,:,i], color = "C"+str(i), label = "case 3 (regular spatial l1)", linewidth=3.0, marker = "")
#                 fig.add_subplot(gs[j,0]).set_ylabel("$\~e("+str(j)+")$", rotation=0, labelpad=30)
#     #             fig.add_subplot(gs[j,0]).fill_between(xrange, fill_up[j,:,i-1], fill_low[j,:,i-1], color = "C"+str(i), alpha=0.3)
#                 try:
#                     fig.add_subplot(gs[j,0]).axvline(x=w_tilde_diff_below_threshold[j,:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#                 except ValueError as e:
#                     print("Above loss threshold! (w_tilde plot) ", i, j)
#             else:
#                 fig.add_subplot(gs[j,0]).plot(w_tilde_diff_mean[j,:,i], color = "C"+str(i), label = "case " + str(i+1), linewidth=3.0, marker = "")
#                 fig.add_subplot(gs[j,0]).set_ylabel("$\~e("+str(j)+")$", rotation=0, labelpad=30)
#     #             fig.add_subplot(gs[j,0]).fill_between(xrange, fill_up[j,:,i-1], fill_low[j,:,i-1], color = "C"+str(i), alpha=0.3)
#                 try:
#                     fig.add_subplot(gs[j,0]).axvline(x=w_tilde_diff_below_threshold[j,:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#                 except ValueError as e:
#                     print("Above loss threshold! (w_tilde plot) ", i, j)
    
#     fig.add_subplot(gs[0,0]).set_title("$\~e(k)$",fontsize = 20)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 8})
# #     fig.add_subplot(gs[5,0]).set_ylabel("Frequency")
#     fig.add_subplot(gs[9,0]).set_xlabel("Training iteration")
    
#     fig.tight_layout()


# def plot_dw_tilde_LR(log):
    
#     fig = plt.figure(figsize = [15,7])
#     fig.patch.set_facecolor('white')
# #     gs = fig.add_gridspec(10,1)
    
# #     dw_tilde_log = log.clone().detach()
    
# #     for j in range(10):
# #         fig.add_subplot(gs[j,0]).plot(dw_tilde_log[0,j,:].cpu().numpy(), color = "C1", label = "actual", linewidth=3.0, marker = "")
# #         fig.add_subplot(gs[j,0]).plot(dw_tilde_log[1,j,:].cpu().numpy(), color = "C2", label = "estimated", linewidth=3.0, marker = "")
        
#     gs = fig.add_gridspec(1,1)
    
#     dw_tilde_log = log.clone().detach()
    
#     for j in range(10):
#         fig.add_subplot(gs[0,0]).plot(dw_tilde_log[0,j,:].cpu().numpy(), color = "C"+str(j), label = "actual"+str(j), linewidth=3.0, marker = "",alpha=0.2)
#         fig.add_subplot(gs[0,0]).plot(dw_tilde_log[1,j,:].cpu().numpy(), color = "C"+str(j), label = "estimated"+str(j), linewidth=2.0, marker = "")
        
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})        
# #     fig.add_subplot(gs[0,0]).set_xlabel("Training iteration")
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
    
# def plot_risk_adv_LR(log, threshold = 1e-3, plot_itr = 1000):
#     #simona
#     return 0
    
# def plot_loss_adv_LR(log, threshold = 1e-3, plot_itr = 1000):
    
#     THRESHOLD = threshold
    
    
#     fig = plt.figure(figsize = [15,7])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(1,1)
    
    
    
#     loss_var, loss_mean  = torch.var_mean(log, dim = 2)
# #     print(loss_mean.shape)
#     fill_up = loss_mean + loss_var
#     fill_low = loss_mean - loss_var

#     xrange = np.arange(log.shape[0])
    

    
#     loss_below_threshold = loss_mean < THRESHOLD
    
# #     print(loss_below_threshold.shape)
    
#     #loss_mean: [iteration, attacks, case]
# #     for _case in range(loss_mean.shape[2]):
# #         for _attack in range(loss_mean.shape[1]):
# #             fig_label = " attack "+str(_attack+1)
# #             fig.add_subplot(gs[_case,0]).plot(loss_mean[:, _attack, _case], color = "C"+str(_attack), label = fig_label, linewidth=3.0, marker = "")
        
# #         fig.add_subplot(gs[_case,0]).set_title("case "+str(_case+1))
# #         fig.add_subplot(gs[_case,0]).legend(prop={"size": 10})
        
        
# #     for _attack in range(loss_mean.shape[1]):
# #         for _case in range(loss_mean.shape[2]):
# #             fig_label = " case "+str(_case+1)
# #             fig.add_subplot(gs[_attack+3,0]).plot(loss_mean[:, _attack, _case], color = "C"+str(_case), label = fig_label, linewidth=3.0, marker = "")
        
# #         fig.add_subplot(gs[_attack+3,0]).set_title("attack "+str(_attack+1))
# #         fig.add_subplot(gs[_attack+3,0]).legend(prop={"size": 10})
        
#     for _case in range(loss_mean.shape[2]):
#         fig_label = " case "+str(_case+1)
#         fig.add_subplot(gs[0,0]).plot(loss_mean[:, 2, _case], color = "C"+str(_case), label = fig_label, linewidth=3.0, marker = "")
#         try:
#             fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,2,_case].tolist().index(1), color = "C"+str(_case), linestyle = '--')
#         except ValueError as e:
#             print("Above loss threshold")
            
#     fig.add_subplot(gs[0,0]).set_xlim([0, plot_itr])

# #     fig.add_subplot(gs[0,0]).set_title("attack "+str(_case+1))
#     title_text = "loss under attack with "+ r"$ \Delta x = iDCT\{ \epsilon sign(r) \frac{V\~w}{||V\~w||}\}$ where $V =diag\{0,1,...,1\}$"
#     fig.add_subplot(gs[0,0]).set_title(title_text,fontsize = 20)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
        
        
        
#         fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
#         try:
#             fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#         except ValueError as e:
#             print("Above loss threshold")
            
        
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.add_subplot(gs[0,0]).set_ylabel("loss")
#     fig.add_subplot(gs[0,0]).set_xlabel("Training iteration")
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
    
    
    
    
    
    
### OBSOLETE ###   
# def train_LR_optim_lr(args, model, device):
    
#     iteration = args["itr"]
#     _d = args["d"]
#     _case = args["case"]
#     _batchsize = args["bsize"]
#     _mu = args["mu"]
#     _std = args["std"]
#     _lambbda = args["lambbda"]
#     _lr = args["lr"]
    
#     log_dict = defaultdict(lambda: list())
    
#     w = torch.zeros(_d, iteration, device = device)
#     w_tilde =  torch.zeros(_d, iteration, device = device)
#     dw_tilde =  torch.zeros(2, _d, iteration, device = device)
#     loss_log = torch.zeros(1, iteration, device = device)
    
#     optim_init = False
# #     opt = optim.SGD(model.parameters(), lr = 1e-2)
    
#     with trange(iteration) as t:
#         for i in range(iteration):

#             x, y = data_init_LR( case = _case, 
#                                  d = _d, 
#                                  batchsize = _batchsize, 
#                                  mu = _mu, 
#                                  std = _std, 
#                                  lambbda = _lambbda)

#             x, y = x.to(device), y.to(device)

#             w_tilde_prev = dct(list(model.parameters())[0][0].view(_d,1)).squeeze().detach()
            
            
#             #### compute optimum learning rate:
# #             if _case == 1 and not optim_init:
# #                 x_tilde = dct(x).squeeze()
# #                 eta = 1/y.item()**2
# #                 opt = optim.SGD(model.parameters(), lr = eta)
# #                 optim_init = True
# #                 print("init")
# #             elif _case == 2 and not optim_init:
# #                 x_tilde = dct(x).squeeze()
# #                 eta = ((w_tilde_prev[0]-1)/(x_tilde[0]*(x_tilde*w_tilde_prev)[1:].sum() + x_tilde[0]**2 * (w_tilde_prev[0]-1) )).item()
# #                 opt = optim.SGD(model.parameters(), lr = eta)
# #                 optim_init = True
# #                 print("init")
# #             elif _case == 3 and not optim_init:
                
#             if not optim_init:
#                 x_tilde = dct(x).squeeze()
#                 eta = ((w_tilde_prev[0]-1)/(x_tilde[0]*(x_tilde*w_tilde_prev)[1:].sum() + x_tilde[0]**2 * (w_tilde_prev[0]-1) )).item()
#                 opt = optim.SGD(model.parameters(), lr = eta)
#                 optim_init = True
#                 print("init")
                

#             opt.zero_grad()

#             y_hat = model(x)
#             ipdb.set_trace()
# #             print(y_hat)

#             loss = ((1/2) * (y_hat.t() - y) ** 2).mean()
#             print("loss:", loss.item())
            
            
# #             print(w_tilde_prev - eta*(y_hat-y).item()*x_tilde)
#             loss_log[:,i] = loss.item()


#             loss.backward()
#             opt.step()

#             w[:,i] = list(model.parameters())[0][0].detach()
            
#             w_tilde[:,i] = dct(w[:,i].view(_d,1)).squeeze().detach()
# #             print(-eta*(x_tilde[0]*(x_tilde*w_tilde_prev)[1:].sum() + y.item()**2*(w_tilde_prev[0]-1)).item())
# #             ipdb.set_trace()
# #             print(-eta*(((x_tilde*w_tilde_prev).sum() - y.item())*x_tilde)[0].item())
#             print("gradient update at dim 0 : ", w_tilde[0,i] - w_tilde_prev[0])
#             print(w_tilde[0,i])
#             dw_tilde[0,:,i] = -(w_tilde[:,i] - w_tilde_prev).detach()
#             dw_tilde[1,:,i] = grad_estimation(w_tilde_prev, 
#                                               _lr, 
#                                               _case, 
#                                               _d, 
#                                               _mu, 
#                                               _std, 
#                                               _lambbda).squeeze().detach()

#         t.set_postfix(loss = loss.item())
#         t.update()
    
#     log_dict["w"] = w
#     log_dict["w_tilde"] = w_tilde
#     log_dict["dw_tilde"] = dw_tilde
#     log_dict["loss"] = loss_log

#     return log_dict

# def plot_first_dim_comparison_LR():
#     fig_test = plt.figure(figsize = [15,5])
#     gs = fig_test.add_gridspec(1,1)
#     p1 = fig_test.add_subplot(gs[0,0]).imshow(diff_map.cpu().detach().numpy(), cmap = 'Reds', aspect = 40.0, vmax = diff_map.max(), vmin = 0)
#     fig_test.colorbar(p1)
#     fig_test.add_subplot(gs[0,0]).set_ylabel("Frequency")
#     fig_test.add_subplot(gs[0,0]).set_xlabel("Training iteration")
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)