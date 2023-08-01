import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import ipdb
from random import randrange
from src.utils_freq import dct2
    
def plot_theta_tilde_NN_flatten(theta_tilde_log, y_limit=0):
    
    theta_tilde = theta_tilde_log.clone().detach()
    hidden_d = theta_tilde.shape[0]
    input_d = theta_tilde.shape[1]
    iteration = theta_tilde.shape[2]
    
    fig = plt.figure(figsize = [25,10])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(hidden_d,1)
    
    
    xrange = np.arange(iteration)
    
#     dct_basis_freq = input_d//numb_dct_basis # theta_tilde(dct_basis_freq * j)
#     dct_basis = torch.range(start = 0, end = input_d - dct_basis_freq, step = dct_basis_freq).tolist()
#     print(dct_basis)
    
#     random_theta_m_index = torch.randint(low=0, high = hidden_d, size = (x_per_case*y_per_case,))
    
#     theta_m_index = random_theta_m_index[i*x_per_case+k]
#     select_theta_tilde = theta_tilde[theta_m_index,:,:,j]
#     ipdb.set_trace()
    plot_theta_tilde = theta_tilde
    plot_theta_tilde_shifted_by_1 = plot_theta_tilde[:,:,1:]
    plot_theta_tilde_remove_last = plot_theta_tilde[:,:,:-1]
    plot_theta_tilde_rate = torch.abs(plot_theta_tilde_shifted_by_1 - plot_theta_tilde_remove_last)
#     ipdb.set_trace()
    for i in range(hidden_d):
#     gs_y_axis = j*y_per_case+i
#         ipdb.set_trace()
        p1 = fig.add_subplot(gs[i,0]).imshow(plot_theta_tilde_rate[i,:,:].detach().cpu().numpy(), aspect = "auto", cmap = "Blues")
        fig.colorbar(p1)
#         title = r"$|\Delta\tilde{\theta}_{"+ str(i)+"}(k)|$"
#         fig.add_subplot(gs[i,0]).set_title(title)
#         yticklabel = [ r"$\tilde{\theta}$("+ str(int(dct_basis[i]))+")" for i in range(len(dct_basis))]
#         ytick = [ str(int(dct_basis[i])) for i in range(len(dct_basis))]
#         print(ytick)
#         fig.add_subplot(gs[i,0]).set_yticks(np.arange(start = 0, stop = input_d - dct_basis_freq, step = dct_basis_freq), minor = False)
#         fig.add_subplot(gs[i,0]).set_yticklabels(ytick)
        fig.add_subplot(gs[i,0]).set_ylabel(r"$|\Delta\tilde{\theta}_{"+ str(int(i))+"}(k)|$", fontsize = 15,rotation=0, labelpad=40)
        if y_limit:
            fig.add_subplot(gs[i,0]).set_ylim([y_limit,0])
    fig.add_subplot(gs[4,0]).set_xlabel("Iteration", fontsize = 10)

#         if j == 0 and case != 1:
#             fig.add_subplot(gs[gs_y_axis,k]).set_ylim([5,0])
#             fig.add_subplot(gs[gs_y_axis,k]).set_yticks(np.arange(start = 0, stop = 10, step = 3), minor = False)
#         else:
#             fig.add_subplot(gs[gs_y_axis,k]).set_ylim([y_limit,0])
        
    fig.tight_layout()
    
def plot_theta_tilde_NN_conv(theta_tilde_log, numb_ckpt):    
    
    theta_tilde = theta_tilde_log.clone().detach()
    input_d = theta_tilde.shape[1]
    hidden_d = theta_tilde.shape[0]
    iteration = theta_tilde.shape[3]
    
    fig = plt.figure(figsize = [25,hidden_d*2])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(hidden_d,1)
    
    itr_freq = iteration//numb_ckpt # theta_tilde(dct_basis_freq * j)
    itr_2plot = torch.range(start = 1, end = iteration, step = itr_freq).tolist()
    
#     random_theta_m_index = torch.randint(low=0, high = hidden_d, size = (5,))
#     select_theta_tilde = theta_tilde[random_theta_m_index,:,:,:]
    select_theta_tilde = theta_tilde
    theta_tilde_shifted_by_1 = select_theta_tilde[:,:,:,1:]
    theta_tilde_remove_last = select_theta_tilde[:,:,:,:-1]
    delta_theta_tilde = torch.abs(theta_tilde_shifted_by_1 - theta_tilde_remove_last)
    
#     ipdb.set_trace()
    for i in range(hidden_d):
        pl = torch.cat([delta_theta_tilde[i,:,:,j] for j in range(len(itr_2plot)) ], dim = 1)
        p1 = fig.add_subplot(gs[i,0]).imshow(pl.detach().cpu().numpy(), aspect = "auto", cmap = "Blues")
        fig.colorbar(p1)

        fig.add_subplot(gs[i,0]).set_ylabel(r"$|\Delta\tilde{\theta}_{"+ str(int(i))+"}|$", fontsize = 15,rotation=0, labelpad=25)
        fig.add_subplot(gs[i,0]).set_xticklabels([int(itr_2plot[j]) for j in range(len(itr_2plot))], minor = False)
        fig.add_subplot(gs[i,0]).set_xticks(np.arange(start = input_d/2 -1, stop = pl.shape[1], step = input_d), minor = False)
    fig.add_subplot(gs[-1,0]).set_xlabel("Iteration", fontsize = 15)
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


def visualize_two_layer(model, init_weight):
    # num_plot = model.conv1.weight.shape[0] if model.conv1.weight.shape[0]<5 else 5
    num_plot = init_weight.shape[0]
    
    fig, axes = plt.subplots(nrows=3, ncols=num_plot, figsize=((num_plot)*6,15))
    
    for i, ax in enumerate(axes.flat):
        
        row, col = i//num_plot, i-(i//num_plot)*num_plot
        
#         print(row,col)
        if row == 0:
            init_w = init_weight[col].squeeze().clone().detach()
            dct_plot = dct2(init_w).abs().detach()
            title = "$|{\~\Theta}_{init}|$"
        elif row == 1:
            final_w = model.conv1.weight[col].squeeze().clone().detach()
            dct_plot = dct2(final_w).abs().detach()
            title = "$|{\~\Theta}_{final}|$"
        elif row ==2:
            init_w = init_weight[col].squeeze().clone().detach()
            dct_init = dct2(init_w).abs().detach()
            final_w = model.conv1.weight[col].squeeze().clone().detach()
            dct_final = dct2(final_w).abs().detach()
            dct_plot = (dct_init-dct_final).squeeze().abs()
            title = "$| {\~\Theta}_{init} - {\~\Theta}_{final}|$"
            
        im = ax.imshow(dct_plot.cpu().numpy(), cmap='gray')
        fig.colorbar(im, ax = ax)
        ax.set_title(title,fontsize = 15)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    return fig

def visualize_attack(x, x_adv):
    print(x.shape,x_adv.shape)
    
    num_plot = x_adv.shape[0] if x_adv.shape[0]<5 else 5
    
    fig, axes = plt.subplots(nrows=3, ncols=num_plot, figsize=((num_plot)*6,10))

    for i, ax in enumerate(axes.flat):
        
        row, col = i//num_plot, i-(i//num_plot)*num_plot
        
#         print(row,col)
        if row == 0:
            img = x[col].squeeze().detach().cpu().numpy()
        elif row == 1:
            img = x_adv[col].squeeze().detach().cpu().numpy()
        elif row ==2:
            img = (x_adv[col]-x[col]).squeeze().detach().cpu().numpy()
            
            
        im = ax.imshow(img, cmap='gray')
            
        fig.colorbar(im, ax = ax)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    return fig
