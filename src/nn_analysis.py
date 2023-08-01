import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.utils_freq import dct, idct, idct2, getDCTmatrix, batch_dct, batch_dct2, batch_idct2

from collections import defaultdict
from tqdm import trange
import ipdb

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import chain
import warnings
warnings.filterwarnings("ignore")


avoid_zero_div = 1e-12


def plot_theta_tilde_NN_image_v3(theta_tilde_log, x_per_case, y_per_case, numb_dct_basis, y_limit):
    
    theta_tilde = theta_tilde_log.clone().detach()
    input_d = theta_tilde.shape[1]
    hidden_d = theta_tilde.shape[0]
    iteration = theta_tilde.shape[2]
    case = theta_tilde.shape[3]
    
    fig = plt.figure(figsize = [20,10])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(case*y_per_case, x_per_case)
    
    
    xrange = np.arange(iteration)
    
    dct_basis_freq = input_d//numb_dct_basis # theta_tilde(dct_basis_freq * j)
    dct_basis = torch.range(start = 0, end = input_d, step = dct_basis_freq).tolist()

    
    random_theta_m_index = torch.randint(low=0, high = hidden_d, size = (x_per_case*y_per_case,))
    
    for j in range(case):
        for i in range(y_per_case):
            for k in range(x_per_case):

                theta_m_index = random_theta_m_index[i*x_per_case+k]
                select_theta_tilde = theta_tilde[theta_m_index,:,:,j]

                plot_theta_tilde = select_theta_tilde
                plot_theta_tilde_shifted_by_1 = plot_theta_tilde[:,1:]
                plot_theta_tilde_remove_last = plot_theta_tilde[:,:-1]
                plot_theta_tilde_rate = torch.abs(plot_theta_tilde_shifted_by_1 - plot_theta_tilde_remove_last)

                gs_y_axis = j*y_per_case+i
                p1 = fig.add_subplot(gs[gs_y_axis,k]).imshow(plot_theta_tilde_rate.detach().cpu().numpy(), aspect = "auto", cmap = "hot")
                fig.colorbar(p1)
                title = r"$|\Delta\tilde{\theta}_{"+ str(theta_m_index.item())+"}(k)|$ for case " + str(j+1)
                fig.add_subplot(gs[gs_y_axis,k]).set_title(title)
                yticklabel = [ r"$\tilde{\theta}$("+ str(int(dct_basis[i]))+")" for i in range(len(dct_basis))]

                fig.add_subplot(gs[gs_y_axis,k]).set_yticks(np.arange(start = 0, stop = input_d, step = dct_basis_freq), minor = False)
                fig.add_subplot(gs[gs_y_axis,k]).set_ylabel("Frequency [k]", fontsize = 10)
                fig.add_subplot(gs[gs_y_axis,k]).set_xlabel("Iteration", fontsize = 10)
                
                if j == 0 and case != 1:
                    fig.add_subplot(gs[gs_y_axis,k]).set_ylim([5,0])
                    fig.add_subplot(gs[gs_y_axis,k]).set_yticks(np.arange(start = 0, stop = 10, step = 3), minor = False)
                else:
                    fig.add_subplot(gs[gs_y_axis,k]).set_ylim([y_limit,0])
        
    fig.tight_layout()
    
def plot_theta_tilde_NN_image_MNIST(theta_tilde_log, numb_ckpt):    
    
    theta_tilde = theta_tilde_log[:,:,:,:,0]
    input_d = theta_tilde.shape[1]
    hidden_d = theta_tilde.shape[0]
    iteration = theta_tilde.shape[3]
    
    fig = plt.figure(figsize = [25,10])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(5,1)
    
    itr_freq = iteration//numb_ckpt # theta_tilde(dct_basis_freq * j)
    itr_2plot = torch.range(start = 1, end = iteration, step = itr_freq).tolist()
    
    random_theta_m_index = torch.randint(low=0, high = hidden_d, size = (5,))
    select_theta_tilde = theta_tilde[random_theta_m_index,:,:,:]
    theta_tilde_shifted_by_1 = select_theta_tilde[:,:,:,1:]
    theta_tilde_remove_last = select_theta_tilde[:,:,:,:-1]
    delta_theta_tilde = torch.abs(theta_tilde_shifted_by_1 - theta_tilde_remove_last)
    
#     ipdb.set_trace()
    for i in range(5):
        pl = torch.cat([delta_theta_tilde[i,:,:,j] for j in range(len(itr_2plot)) ], dim = 1)
        p1 = fig.add_subplot(gs[i,0]).imshow(pl.detach().cpu().numpy(), aspect = "auto", cmap = "Blues")
        fig.colorbar(p1)

        fig.add_subplot(gs[i,0]).set_ylabel(r"$|\Delta\tilde{\theta}_{"+ str(int(random_theta_m_index[i]))+"}|$", fontsize = 15,rotation=0, labelpad=25)
        fig.add_subplot(gs[i,0]).set_xticklabels([int(itr_2plot[j]) for j in range(len(itr_2plot))], minor = False)
        fig.add_subplot(gs[i,0]).set_xticks(np.arange(start = input_d/2 -1, stop = pl.shape[1], step = input_d), minor = False)
    fig.add_subplot(gs[4,0]).set_xlabel("Iteration", fontsize = 15)
    fig.tight_layout()


def plot_loss_NN(log, threshold = 1e-3, plot_itr = 1000):
    
    THRESHOLD = threshold
    
    fig = plt.figure(figsize = [15,7])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1,1)
    
#     loss_var, loss_mean  = torch.var_mean(log, dim = 1) 
#     fill_up = loss_mean + loss_var
#     fill_low = loss_mean - loss_var

#     xrange = np.arange(log.shape[0])  
    
#     loss_below_threshold = loss_mean < THRESHOLD
    
    for j in range(log.shape[1]):
        fig.add_subplot(gs[0,0]).plot(log[:,j].detach().cpu().numpy(), linewidth=3.0, marker = "",color = "C"+str(j), label = "case"+ str(j+1))
    
#     for i in range(log.shape[2]):
#         fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = "case "+str(i+1), linewidth=3.0, marker = "")
#         fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
        
#         try:
#             fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#         except ValueError as e:
#             print("Above loss threshold! (loss plot)")
            
#     fig.add_subplot(gs[0,0]).set_xlim([0, plot_itr])
            
        
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
    fig.add_subplot(gs[0,0]).set_ylabel("loss", fontsize = 20)
    fig.add_subplot(gs[0,0]).set_xlabel("Iteration", fontsize = 20)
#     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    

def plot_acc_NN(log, threshold = 1e-3, plot_itr = 1000):
    
    THRESHOLD = threshold
    
    fig = plt.figure(figsize = [15,7])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1,1)
    
    for j in range(log.shape[1]):
        fig.add_subplot(gs[0,0]).plot(log[:,j].detach().cpu().numpy(), linewidth=3.0, marker = "",color = "C"+str(j), label = "case"+ str(j+1))
    
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
    fig.add_subplot(gs[0,0]).set_ylabel("Accuracy (%)", fontsize = 20)
    fig.add_subplot(gs[0,0]).set_xlabel("Iteration", fontsize = 20)
#     fig.add_subplot(gs[0,0]).set_title("Acc",fontsize = 20)
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    
    
######################################################################
# OBSOLETE
######################################################################

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

    
# def plot_theta_tilde_NN_all(theta_tilde_log):
    
#     fig = plt.figure(figsize = [15,15])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(10,1)
    
#     theta_tilde = theta_tilde_log.clone().detach()
    
    
#     input_d = theta_tilde.shape[1]
#     hidden_d = theta_tilde.shape[0]
#     iteration = theta_tilde.shape[2]
#     case = theta_tilde_log.shape[3]
# #     xrange = np.arange(iteration)
    
# #     for i in range(1):
#     i = 1
#     for j in range(10): #we only plot 10 of the total dct basis
#         for k in range(hidden_d):
#             dct_basis = 0+j*int(input_d/10)
#             fig.add_subplot(gs[j,0]).plot(theta_tilde[k,dct_basis,:,i].detach().cpu().numpy(), color = "C"+str(i), label = "case " + str(i+1), linewidth=3.0, marker = "")
#             fig.add_subplot(gs[j,0]).set_ylabel(r"$\tilde{\theta}$("+ str(dct_basis)+")", fontsize = 10)
    
# #     fig.add_subplot(gs[0,0]).set_ylabel("Frequency [DCT basis]", fontsize = 20)
# #     fig.add_subplot(gs[0,0]).set_xlabel("Iteration", fontsize = 10)
# #     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
# #     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)

# def plot_theta_tilde_NN_image_raw_concat(theta_tilde_log):
    
#     fig = plt.figure(figsize = [15,15])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(3,1)
    
#     theta_tilde = theta_tilde_log.clone().detach()
    
#     input_d = theta_tilde.shape[1]
#     hidden_d = theta_tilde.shape[0]
#     iteration = theta_tilde.shape[2]
#     xrange = np.arange(iteration)
# #     ipdb.set_trace()
    
# #     for j in range(theta_tilde_log.shape[3]):
#     plot_theta_tilde_0 = theta_tilde[:,:,:,0].transpose(0,1).reshape(input_d*hidden_d,iteration)
#     plot_theta_tilde_1 = theta_tilde[:,:,:,1].transpose(0,1).reshape(input_d*hidden_d,iteration)
#     plot_theta_tilde_2 = theta_tilde[:,:,:,2].transpose(0,1).reshape(input_d*hidden_d,iteration)
#     stacked = torch.cat([plot_theta_tilde_0,plot_theta_tilde_1,plot_theta_tilde_2], dim = 0)


#     p1 = fig.add_subplot(gs[0,0]).imshow(stacked.detach().cpu().numpy(), aspect =5, cmap = "gray")
                
    
# #     fig.add_subplot(gs[0,0]).set_ylabel("Frequency [DCT basis]", fontsize = 20)
# #     fig.add_subplot(gs[0,0]).set_xlabel("Iteration", fontsize = 10)
# #     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
# #     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
# def plot_theta_tilde_NN_image_raw(theta_tilde_log):
    
#     fig = plt.figure(figsize = [15,15])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(3,1)
    
#     theta_tilde = theta_tilde_log.clone().detach()
    
#     input_d = theta_tilde.shape[1]
#     hidden_d = theta_tilde.shape[0]
#     iteration = theta_tilde.shape[2]
#     xrange = np.arange(iteration)
# #     ipdb.set_trace()
    
#     for j in range(theta_tilde_log.shape[3]):
#         plot_theta_tilde = theta_tilde[:,:,:,j].transpose(0,1).reshape(input_d*hidden_d,iteration)

#         p1 = fig.add_subplot(gs[j,0]).imshow(plot_theta_tilde.detach().cpu().numpy(), aspect =5, cmap = "gray")
                
    
# #     fig.add_subplot(gs[0,0]).set_ylabel("Frequency [DCT basis]", fontsize = 20)
# #     fig.add_subplot(gs[0,0]).set_xlabel("Iteration", fontsize = 10)
# #     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
# #     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
# def plot_theta_tilde_NN_image_v1(theta_tilde_log):
    
#     fig = plt.figure(figsize = [15,15])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(theta_tilde_log.shape[3],1)
    
#     theta_tilde = theta_tilde_log.clone().detach()
    
#     input_d = theta_tilde.shape[1]
#     hidden_d = theta_tilde.shape[0]
#     iteration = theta_tilde.shape[2]
#     xrange = np.arange(iteration)
    
#     dct_basis_freq = input_d//10 # theta_tilde(dct_basis_freq * j)
#     dct_basis = torch.range(start = 0, end = input_d-dct_basis_freq, step = dct_basis_freq).tolist()
    
#     select_theta_tilde = theta_tilde[:10,dct_basis,:,:]    
    
    
#     for j in range(theta_tilde_log.shape[3]):
# #     j = 2
# #         ipdb.set_trace()
# #     plot_theta_tilde = theta_tilde[:,:,:,j].reshape(input_d*hidden_d,iteration)
    
# #     plot_theta_tilde = theta_tilde.transpose(0,1).reshape(input_d*hidden_d,iteration,3)[:,:,j]
# #     plot_theta_tilde = theta_tilde.transpose(0,1).reshape(input_d*hidden_d,iteration,3).reshape(input_d*hidden_d*3,iteration)
# #     plot_theta_tilde = plot_theta_tilde - plot_theta_tilde.min(1)[0].view(100,1).repeat(1,2001)
# #     plot_theta_tilde = plot_theta_tilde / plot_theta_tilde.max(1)[0].view(100,1).repeat(1,2001)
# #     ipdb.set_trace()
# #     for k in range(10):
# #     p1 = fig.add_subplot(gs[0,0]).imshow(plot_theta_tilde.detach().cpu().numpy(), aspect =5)
# #     fig.colorbar(p1)
#         plot_theta_tilde = select_theta_tilde[:,:,:,j].transpose(0,1).reshape(100,iteration)
#         threshold = (plot_theta_tilde.max(1)[0] - plot_theta_tilde.min(1)[0]).view(100,1) > 1e-3 
#         plot_theta_tilde = plot_theta_tilde - plot_theta_tilde.min(1)[0].view(100,1)
#         plot_theta_tilde = plot_theta_tilde / plot_theta_tilde.max(1)[0].view(100,1) * threshold
#     #     ipdb.set_trace()

# #         for k in range(plot_theta_tilde.shape[0]):
# #             if plot_theta_tilde[k,-1] > plot_theta_tilde[k,0]:
# #                 plot_theta_tilde[k,:] = plot_theta_tilde[k,:].flip(0)
#         p1 = fig.add_subplot(gs[j,0]).imshow(plot_theta_tilde.detach().cpu().numpy(), aspect =5, cmap = "gray")
# #         fig.add_subplot(gs[j,0]).set_yticks([0,100])
# #         ipdb.set_trace()
#         yticklabel = [ r"$\tilde{\theta}$("+ str(int(dct_basis[i]))+")" for i in range(10)]
#         yticklabel_major = [ "" for i in range(10)]
# #         yticklabel = list(chain(*yticklabel))
# #         yticklabel = [ ["",r"$\tilde{\theta}$("+ str(int(dct_basis[i]))+")"] for i in range(10)]
# #         yticklabel = list(chain(*yticklabel))
# #         print(yticklabel)
# #         yticklabel = [ str(i-1) for i in range(11)]
# #         ipdb.set_trace()
# #         fig.add_subplot(gs[j,0]).yaxis.set_major_locator(MaxNLocator(10))
#         fig.add_subplot(gs[j,0]).set_yticks(np.arange(start = 0, stop = 4*25, step =10), minor = False)
# #         fig.add_subplot(gs[j,0]).set_yticks([], minor = False)
#         fig.add_subplot(gs[j,0]).set_yticks(np.arange(start = 4, stop = 4*25, step =10), minor = True)
#         fig.add_subplot(gs[j,0]).set_yticklabels(yticklabel, minor = True)
#         fig.add_subplot(gs[j,0]).set_yticklabels(yticklabel_major, minor = False)
#         fig.add_subplot(gs[j,0]).tick_params(axis = "y", which = "minor", length = 0)
        
                
    
# #     fig.add_subplot(gs[0,0]).set_ylabel("Frequency [DCT basis]", fontsize = 20)
# #     fig.add_subplot(gs[0,0]).set_xlabel("Iteration", fontsize = 10)
# #     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
# #     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
# #     plt.gca().spines['top'].set_visible(False)
# #     plt.gca().spines['right'].set_visible(False)

# def plot_theta_tilde_NN_image_v2_backup(theta_tilde_log, x_per_case, y_per_case, numb_dct_basis, y_limit):
    
#     theta_tilde = theta_tilde_log.clone().detach()
#     input_d = theta_tilde.shape[1]
#     hidden_d = theta_tilde.shape[0]
#     iteration = theta_tilde.shape[2]
#     case = theta_tilde.shape[3]
    
#     fig = plt.figure(figsize = [20,10])
#     fig.patch.set_facecolor('white')
# #     print(case*y_per_case)
#     gs = fig.add_gridspec(case*y_per_case, x_per_case)
    
    
#     xrange = np.arange(iteration)
    
#     dct_basis_freq = input_d//numb_dct_basis # theta_tilde(dct_basis_freq * j)
#     dct_basis = torch.range(start = 0, end = input_d, step = dct_basis_freq).tolist()
# #     print(dct_basis)
    
# #     select_theta_tilde = theta_tilde[:1,dct_basis,:,:]    
    
#     random_theta_m_index = torch.randint(low=0, high = hidden_d, size = (x_per_case*y_per_case,))
    
#     for j in range(case):
# #     j = 0
#         for i in range(y_per_case):
#             for k in range(x_per_case):
#     #                 ipdb.set_trace()

#                 theta_m_index = random_theta_m_index[i*x_per_case+k]
#     #                 print(theta_m_index)
#     #                 select_theta_tilde = theta_tilde[theta_m_index,dct_basis,:,j] 
#                 select_theta_tilde = theta_tilde[theta_m_index,:,:,j]

#     #                 plot_theta_tilde = select_theta_tilde[:,:,:,j].transpose(0,1).reshape(numb_dct_basis,iteration)
#                 plot_theta_tilde = select_theta_tilde
#     #                 ipdb.set_trace()
#                 plot_theta_tilde_shifted_by_1 = plot_theta_tilde[:,1:]
#                 plot_theta_tilde_remove_last = plot_theta_tilde[:,:-1]
#                 plot_theta_tilde_rate = torch.abs(plot_theta_tilde_shifted_by_1 - plot_theta_tilde_remove_last)
#     #                 plot_theta_tilde_rate = [torch.abs(plot_theta_tilde[:,l] - plot_theta_tilde[:,l-1]).view(numb_dct_basis,1) for l in range(1,iteration)]
#     #                 plot_theta_tilde_rate = torch.cat(plot_theta_tilde_rate, dim = 1)

#     #                 threshold = (plot_theta_tilde.max(1)[0] - plot_theta_tilde.min(1)[0]).view(numb_dct_basis,1) > 1e-3 
#     #                 plot_theta_tilde = plot_theta_tilde - plot_theta_tilde.min(1)[0].view(numb_dct_basis,1)
#     #                 plot_theta_tilde = plot_theta_tilde / plot_theta_tilde.max(1)[0].view(numb_dct_basis,1) * threshold
#     #                 plot_theta_tilde = plot_theta_tilde / plot_theta_tilde[:,0].view(numb_dct_basis,1)
#             #     ipdb.set_trace()

#     #                 for l in range(plot_theta_tilde.shape[0]):
#     #                     if plot_theta_tilde[l,-1] > plot_theta_tilde[l,0]:
#     #                         plot_theta_tilde[l,:] = plot_theta_tilde[l,:].flip(0)
#     #                 print([j*y_per_case+i,k])

#                 gs_y_axis = j*y_per_case+i
#                 p1 = fig.add_subplot(gs[gs_y_axis,k]).imshow(plot_theta_tilde_rate.detach().cpu().numpy(), aspect = "auto", cmap = "hot")
#                 fig.colorbar(p1)
#                 title = r"$|\Delta\tilde{\theta}_{"+ str(theta_m_index.item())+"}(k)|$ for case " + str(j+1)
#                 fig.add_subplot(gs[gs_y_axis,k]).set_title(title)
#                 yticklabel = [ r"$\tilde{\theta}$("+ str(int(dct_basis[i]))+")" for i in range(len(dct_basis))]
# #                 yticklabel = [ r"$\tilde{\theta}$("+ str(int(dct_basis[i]))+")" for i in range(len(dct_basis))]
#     #             print(yticklabel)
#     #             break
#     #             yticklabel_major = [ "" for i in range(numb_dct_basis)]
#     #                 fig.add_subplot(gs[gs_y_axis,k]).set_yticks(np.arange(start = 0, stop = numb_dct_basis, step =1), minor = False)
#     #             fig.add_subplot(gs[gs_y_axis,k]).set_yticks(np.arange(start = 0, stop = input_d, step =1), minor = False)
#                 fig.add_subplot(gs[gs_y_axis,k]).set_yticks(np.arange(start = 0, stop = input_d, step = dct_basis_freq), minor = False)
#     #             fig.add_subplot(gs[gs_y_axis,k]).set_yticklabels(yticklabel, minor = True)
# #                 fig.add_subplot(gs[gs_y_axis,k]).set_yticklabels(yticklabel, minor = False)
#                 fig.add_subplot(gs[gs_y_axis,k]).set_ylabel("Frequency [k]", fontsize = 10)
#                 fig.add_subplot(gs[gs_y_axis,k]).set_xlabel("Iteration", fontsize = 10)
                
#                 if j == 0:
#                     fig.add_subplot(gs[gs_y_axis,k]).set_ylim([5,0])
#                     fig.add_subplot(gs[gs_y_axis,k]).set_yticks(np.arange(start = 0, stop = 10, step = 3), minor = False)
# #                     yticklabel = [ r"$\tilde{\theta}(0)$", r"$\tilde{\theta}(3)$", r"$\tilde{\theta}(6)$", r"$\tilde{\theta}(9)$"]
# #                     fig.add_subplot(gs[gs_y_axis,k]).set_yticklabels(yticklabel, minor = False)
#                 else:
#                     fig.add_subplot(gs[gs_y_axis,k]).set_ylim([y_limit,0])
#     #             fig.add_subplot(gs[gs_y_axis,k]).set_yticklabels(yticklabel, minor = False)
#     #             fig.add_subplot(gs[gs_y_axis,k]).tick_params(axis = "y", which = "minor", length = 0)
        
                
    
# #     fig.add_subplot(gs[0,0]).set_ylabel("Frequency [DCT basis]", fontsize = 20)
# #     fig.add_subplot(gs[0,0]).set_xlabel("Iteration", fontsize = 10)
# #     fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
# #     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
# #     plt.gca().spines['top'].set_visible(False)
# #     plt.gca().spines['right'].set_visible(False)
    
# def plot_theta_tilde_NN_mean(theta_tilde_log):
    
#     fig = plt.figure(figsize = [15,15])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(10,1)
    
#     theta_tilde = theta_tilde_log.clone().detach()
    
    
#     input_d = theta_tilde.shape[1]
#     hidden_d = theta_tilde.shape[0]
#     xrange = np.arange(theta_tilde.shape[2])
#     theta_tilde_var, theta_tilde_mean  = torch.var_mean(theta_tilde, dim = 0)

#     for j in range(theta_tilde_log.shape[3]):
#         for i in range(10):
#             dct_basis = 0+i*int(input_d/10)
#             fig.add_subplot(gs[i,0]).plot(theta_tilde[:,dct_basis,:,j].mean(dim=0).detach().cpu().numpy(), color = "C"+str(j), label = "case " + str(j+1), linewidth=3.0, marker = "")
#             fig.add_subplot(gs[i,0]).set_ylabel(r"$\tilde{\theta}$("+ str(dct_basis)+")", fontsize = 10)
    
#     fig.add_subplot(gs[9,0]).set_xlabel("Iteration", fontsize = 10)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
# def plot_theta_tilde_NN_abs_mean(theta_tilde_log):
    
#     fig = plt.figure(figsize = [15,15])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(10,1)
    
#     theta_tilde = theta_tilde_log.clone().detach()
    
    
#     input_d = theta_tilde.shape[1]
#     hidden_d = theta_tilde.shape[0]
#     xrange = np.arange(theta_tilde.shape[2])
#     theta_tilde_var, theta_tilde_mean  = torch.var_mean(theta_tilde, dim = 0)

#     for j in range(theta_tilde_log.shape[3]):
#         for i in range(10):
#             dct_basis = 0+i*int(input_d/10)
#             fig.add_subplot(gs[i,0]).plot(theta_tilde[:,dct_basis,:,j].abs().mean(dim=0).detach().cpu().numpy(), color = "C"+str(j), label = "case " + str(j+1), linewidth=3.0, marker = "")
#             fig.add_subplot(gs[i,0]).set_ylabel(r"$\tilde{\theta}$("+ str(dct_basis)+")", fontsize = 10)
    
#     fig.add_subplot(gs[9,0]).set_xlabel("Iteration", fontsize = 10)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
# def plot_theta_tilde_NN_single_j(theta_tilde_log, plot_dim = 0):
    
#     fig = plt.figure(figsize = [15,15])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(10,1)
    
#     theta_tilde = theta_tilde_log.clone().detach()
    
    
#     input_d = theta_tilde.shape[1]
#     hidden_d = theta_tilde.shape[0]
#     xrange = np.arange(theta_tilde.shape[2])

#     for j in range(theta_tilde_log.shape[3]):
#         for i in range(10):
#             dct_basis = 0+i*int(input_d/10)
#             fig.add_subplot(gs[i,0]).plot(theta_tilde[plot_dim,dct_basis,:,j].detach().cpu().numpy(), color = "C"+str(j), label = "case " + str(j+1), linewidth=3.0, marker = "")
#             fig.add_subplot(gs[i,0]).set_ylabel(r"$\tilde{\theta}$("+ str(dct_basis)+")", fontsize = 10)

#     fig.add_subplot(gs[9,0]).set_xlabel("Iteration", fontsize = 10)
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.tight_layout()
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)


# def plot_u_NN(log, threshold = 1e-3, plot_itr = 1000):
#     fig = plt.figure(figsize = [15,7])
#     fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(1,1)
#     fig.add_subplot(gs[0,0]).plot(log[:,:,0].detach().cpu().numpy()[:], linewidth=3.0, marker = ".", label = "u_init")
#     fig.add_subplot(gs[0,0]).plot(log[:,:,1].detach().cpu().numpy()[:], linewidth=3.0, marker = ".", label = "u_final")
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
#     fig.add_subplot(gs[0,0]).set_xlabel(r"$u_j$", fontsize = 20)