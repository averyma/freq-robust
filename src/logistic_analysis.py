import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.utils_freq import batch_dct, dct, idct, getDCTmatrix

from collections import defaultdict
from tqdm import trange
import ipdb

from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



def loader_LR(case, p, d, batchsize, mu, std, lambbda, iteration):
    
    total_size = batchsize*iteration
    
    x, y = data_init_LR(case, p, d, total_size, mu, std, lambbda)
    dataset = TensorDataset(x.t(), y.t())
    loader = DataLoader(dataset, batch_size = batchsize, pin_memory = True, shuffle=True)
    
    return loader

def data_init_LR(case = 1, p = 3, d = 10, total_size = 100000, mu = 1, std = 0.5, lambbda = 1):
    
    x_tilde = torch.zeros(d, total_size)
    y = torch.zeros(total_size)
    
    x_tilde[:p, :int(total_size/2)] = torch.normal(mean = mu, std = std, size = (p,int(total_size/2)))
    y[:int(total_size/2)] = 0
    
    x_tilde[:p, int(total_size/2):] = torch.normal(mean = -mu, std = std, size = (p,int(total_size/2)))
    y[int(total_size/2):] = 1

    if case ==1:
        x_tilde[p:,:] = 0
    elif case ==2:
        rand_sign = torch.rand_like(x_tilde[p:,:])-0.5
        rand_sign = (rand_sign/rand_sign.abs()).detach()
        x_tilde[p:,:] = torch.normal(mean = mu, std = std, size = (d-p,total_size)) * rand_sign
    elif case == 3:
        rand_sign = torch.rand_like(x_tilde[p:,:])-0.5
        rand_sign = (rand_sign/rand_sign.abs()).detach()
        x_tilde[p:,:] = torch.normal(mean = mu, std = std, size = (d-p,total_size)) * rand_sign
        decay = torch.exp(-lambbda*(torch.range(1, d-p))).view(d-p,1).repeat(1, total_size)
        x_tilde[p:,:] = x_tilde[p:,:] * decay

    x = idct(x_tilde)
    return x, y

def train_LR(args, model, opt, device):
    
    iteration = args["itr"]
    _d = args["d"]
    _p = args["p"]
    _case = args["case"]
    _batchsize = args["bsize"]
    _mu = args["mu"]
    _std = args["std"]
    _lambbda = args["lambbda"]
    _lr = args['lr']
    _method = args['method']
    
    log_dict = defaultdict(lambda: list())
    
    w_tilde = torch.zeros(_d, iteration, device = device)
    loss_logger = torch.zeros(1, iteration, device = device)
    acc_logger = torch.zeros(1, iteration, device = device)
    
    
    train_loader = loader_LR(_case, _p, _d, _batchsize, _mu, _std, _lambbda, iteration)
    
    dct_matrix = getDCTmatrix(_d)
        
    i = 0
    for x, y in train_loader:
        
        prev_w_tilde = dct(model.state_dict()['linear.weight'].view(_d,1)).squeeze().detach()

        x, y = x.t().to(device), y.t().to(device)
    
        if torch.isnan(x).sum().item() != 0:
            print("NaN detected in data: removed", torch.isnan(x).sum().item(), "datapoints")
            nonNan_idx = torch.tensor(torch.isnan(x).sum(dim=0)==0, dtype = torch.bool)
            x = x[:,nonNan_idx]
            y = y[nonNan_idx]
            
        opt.zero_grad()

        z = model(x).view(y.shape)
        y_hat = torch.sigmoid(z)
        
        loss = torch.nn.BCEWithLogitsLoss()(z, y)
        
        batch_correct = ((z > 0) == (y==1)).sum().item()
        batch_acc = batch_correct /x.shape[1]*100

        loss_logger[:,i] = loss.item()
        acc_logger[:,i] = batch_acc

        if _method == 'weighted_l1f':
            factor = args['factor']
#             ipdb.set_trace()
            curr_w = model.linear.weight.t()
            curr_w_tilde = dct(curr_w)
            
            AVOID_ZERO_DIV = 1e-6
            mean_abs_x_tilde = batch_dct(x.t(), dct_matrix).abs().mean(dim=0)
            decay_factor = mean_abs_x_tilde/mean_abs_x_tilde[0]
            M = (1/(decay_factor+AVOID_ZERO_DIV)).view(1,_d)
            
            weighted_w_tilde = torch.mul(M, curr_w_tilde).squeeze()
            
            l1_reg = torch.norm(weighted_w_tilde,p=1)
            
            loss_reg = loss+factor*l1_reg 
            loss_reg.backward()
            opt.step()
        elif _method == 'l1f':
            factor = args['factor']
#             ipdb.set_trace()
            curr_w = model.linear.weight.t()
            curr_w_tilde = dct(curr_w)
            
            AVOID_ZERO_DIV = 1e-6
            mean_abs_x_tilde = batch_dct(x.t(), dct_matrix).abs().mean(dim=0)
            decay_factor = mean_abs_x_tilde/mean_abs_x_tilde[0]
            M = (1/(decay_factor+AVOID_ZERO_DIV)).view(1,_d)
            M = torch.ones_like(M, device =device)
            
            weighted_w_tilde = torch.mul(M, curr_w_tilde).squeeze()
            
            l1_reg = torch.norm(weighted_w_tilde,p=1)
            
            loss_reg = loss+factor*l1_reg 
            loss_reg.backward()
            opt.step()
        elif _method == 'l1s':
            factor = args['factor']
            curr_w = model.linear.weight.squeeze()
            l1_reg = torch.norm(curr_w,p=1)
            loss_reg = loss+factor*l1_reg 
            loss_reg.backward()
            opt.step()
        else:
            loss.backward()
            curr_w = model.linear.weight.clone().detach()
            grad = model.linear.weight.grad.clone().detach()
            if _method == 'weighted_lr':
                dct_grad = dct(grad.t())
                AVOID_ZERO_DIV = 1e-6
                mean_abs_x_tilde = batch_dct(x.t(), dct_matrix).abs().mean(dim=0)
                decay_factor = mean_abs_x_tilde/mean_abs_x_tilde[0]
                M = (1/(decay_factor+AVOID_ZERO_DIV)).view(_d, 1)
    #             M = M/M[0]
        #         M = torch.ones_like(M, device = x.device) # for sanity check
                new_w = curr_w - idct(_lr * torch.mul(M,dct_grad)).t()
            else:
                new_w = curr_w - _lr * grad

            model.linear.weight = torch.nn.parameter.Parameter(new_w)
        
        curr_w_tilde = dct(model.state_dict()['linear.weight'].view(_d,1)).squeeze().detach()

        w_tilde[:,i] = curr_w_tilde
        
        i += 1
            
    log_dict["w_tilde"] = w_tilde
    log_dict["loss"] = loss_logger
    log_dict["acc"] = acc_logger
    
    return log_dict

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
    
# def plot_loss_LR(log, threshold = 1e-3, plot_itr = 1000):
    
#     THRESHOLD = threshold
    
#     fig = plt.figure(figsize = [15,7])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(1,1)
    
#     loss_var, loss_mean  = torch.var_mean(log, dim = 1) 
#     fill_up = loss_mean + loss_var
#     fill_low = loss_mean - loss_var

#     xrange = np.arange(log.shape[0])  
    
#     loss_below_threshold = loss_mean < THRESHOLD
    
    
#     for i in range(log.shape[2]):
#         fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = "case "+str(i+1), linewidth=3.0, marker = "")
#         fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
        
#         try:
#             fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#         except ValueError as e:
#             print("Above loss threshold! (loss plot)")
            
#     fig.add_subplot(gs[0,0]).set_xlim([0, plot_itr])
            
        
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.add_subplot(gs[0,0]).set_ylabel("loss")
#     fig.add_subplot(gs[0,0]).set_xlabel("Training iteration")
#     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
# def plot_risk_LR(args, w_tilde_log, loss_log, threshold = 1e-3, plot_itr = 1000):
    
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
# #     for i in [2]:
#         fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = "case "+str(i+1), linewidth=3.0, marker = "")
#         fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
        
#         if i == 0: # case 1
#             e_0 = w_tilde_log_copy[0,0,:,0] - 1 # only supports numb_runs = 1, so error will occur if we do average over multiple runs
#             risk = 0.5 * e_0**2 * _std**2 * torch.tensor(1 - 2*_lr*_std**2 + 3*_lr**2*_std**4)**torch.tensor(xrange)
#             fig.add_subplot(gs[0,0]).plot(risk, color = "C"+str(i+3), label = "case "+str(i+1)+" risk", linewidth=3.0, marker = "")
#         elif i == 1: # case 2
#             e_0 = w_tilde_log_copy[:,0,:,1] # only supports numb_runs = 1, so error will occur if we do average over multiple runs
#             e_0[0] = e_0[0] - 1
#             risk = 0.5 * torch.norm(e_0, p =2)**2 * _std**2 * torch.tensor(1 - 2*_lr*_std**2 + 3*_lr**2*_std**4)**torch.tensor(xrange)
#             fig.add_subplot(gs[0,0]).plot(risk, color = "C"+str(i+3), label = "case "+str(i+1)+" risk", linewidth=3.0, marker = "")
#         elif i ==2: # case 3
#             e_i = w_tilde_log_copy[:,0,:,2]
#             e_i[0] -= 1
# #             ipdb.set_trace()
#             bracket_term = [np.exp(-2*d*_lambbda)*torch.tensor(1 - 2 * _lr * _std**2 * np.exp(-2*d*_lambbda) + 3 * _lr**2 * _std**4 * np.exp(-4*d*_lambbda))**torch.tensor(xrange) for d in range(_d)]
#             sum_term = torch.stack(bracket_term).T @ (torch.tensor(e_i)**2)
#             risk = 0.5 * _std**2 * sum_term
#             fig.add_subplot(gs[0,0]).plot(risk, color = "C"+str(i+3), label = "case "+str(i+1)+" risk", linewidth=3.0, marker = "")

#         try:
#             fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#         except ValueError as e:
#             print("Above loss threshold! (loss plot)")
            
#     fig.add_subplot(gs[0,0]).set_xlim([0, plot_itr])
            
        
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.add_subplot(gs[0,0]).set_ylabel("loss")
#     fig.add_subplot(gs[0,0]).set_xlabel("Training iteration")
#     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
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
#             fig.add_subplot(gs[j,0]).plot(w_tilde_diff_mean[j,:,i], color = "C"+str(i), label = "case " + str(i+1), linewidth=3.0, marker = "")
# #             fig.add_subplot(gs[j,0]).fill_between(xrange, fill_up[j,:,i-1], fill_low[j,:,i-1], color = "C"+str(i), alpha=0.3)
#             try:
#                 fig.add_subplot(gs[j,0]).axvline(x=w_tilde_diff_below_threshold[j,:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#             except ValueError as e:
#                 print("Above loss threshold! (w_tilde plot) ", i, j)
    
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10}) 
#     fig.add_subplot(gs[0,0]).set_title("$\~e(k)$",fontsize = 20)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.add_subplot(gs[5,0]).set_ylabel("Frequency")
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
        
        
        
# #         fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
# #         try:
# #             fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
# #         except ValueError as e:
# #             print("Above loss threshold")
            
        
# #     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
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