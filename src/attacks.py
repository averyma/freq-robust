import torch
import torch.nn as nn
import numpy as np
from src.utils_freq import rgb2gray, dct, batch_dct2, batch_idct2, getDCTmatrix
import ipdb

avoid_zero_div = 1e-12

class pgd(object):
    """ PGD attacks, with random initialization within the specified lp ball """
    def __init__(self, **kwargs):
        # define default attack parameters here:
        self.param = {'ord': np.inf,
                      'epsilon': 8./255.,
                      'alpha': 2./255.,
                      'num_iter': 20,
                      'restarts': 1,
                      'rand_init': True,
                      'clip': True,
                      'loss_fn': nn.CrossEntropyLoss()}
        # parse thru the dictionary and modify user-specific params
        self.parse_param(**kwargs) 
        
    def generate(self, model, x, y):
        epsilon = self.param['epsilon']
        num_iter = self.param['num_iter']
        alpha = epsilon if num_iter ==1 else self.param['alpha']
        rand_init = self.param['rand_init']
        clip = self.param['clip']
        restarts = self.param['restarts']
        loss_fn = self.param['loss_fn']
        p_norm = self.param['ord']
        
        # implementation begins:
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = torch.zeros_like(x)
        _dim = x.shape[1] * x.shape[2] * x.shape[3]
        
        for i in range(restarts):
            if p_norm == np.inf:
                
                if rand_init:
                    delta = torch.rand_like(x, requires_grad=True)
                    delta.data = delta.data * 2. * epsilon - epsilon
                    if clip:
                        delta.data = (x.data + delta.data).clamp(min = 0, max = 1.0) - x.data
                else:
                    delta = torch.zeros_like(x, requires_grad=True)
                for t in range(num_iter):
                    model.zero_grad()
                    
                    loss = loss_fn(model(x + delta), y)
                        
                    loss.backward()
                    # first we need to make sure delta is within the specified lp ball
                    delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(min = -epsilon, max = epsilon)
                    # then we need to make sure x+delta in the next iteration is within the [0,1] range
                    if clip:
                        delta.data = (x.data + delta.data).clamp(min = 0, max = 1.) - x.data
                    delta.grad.zero_()

            elif p_norm == 2:
                if rand_init:
                    # first we sample a random direction and normalize it
                    delta = torch.rand_like(x, requires_grad = True) # U[0,1]
                    delta.data = delta.data * 2.0 - 1.0 # U[-1,1]
                    delta_norm = torch.norm(delta.detach(), p = 2 , dim = (1,2,3), keepdim = True).clamp(min = avoid_zero_div) # get norm
                    # next, we get a random radius < epsilon
                    rand_radius = torch.rand(x.shape[0], requires_grad = False,device=x.device).view(x.shape[0],1,1,1) * epsilon # get random radius
                    
                    # finally we re-assign delta
                    delta.data = epsilon * delta.data/delta_norm

                    if clip:
                        delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data
                else:
                    delta = torch.zeros_like(x, requires_grad = True)
                    
                for t in range(num_iter):
                    model.zero_grad()
                    loss = loss_fn(model(x + delta), y)
                    loss.backward()
                    
                    # computing norm of loss gradient wrt input
                    # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
                    grad_norm = torch.norm(delta.grad.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
                    # one step in the direction of normalized gradient (stepsize = alpha)
                    delta.data = delta + alpha*delta.grad.detach()/grad_norm
                    # computing the norm of the new delta term
                    # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
                    delta_norm = torch.norm(delta.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
                    # here the clip factor is used to **clip** to within the norm ball
                    # not to **normalize** onto the surface of the ball
                    factor = torch.min(epsilon/delta_norm, torch.tensor(1., device = x.device ))

                    delta.data = delta.data * factor
                    if clip:
                        delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data

                    delta.grad.zero_()
            else: 
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)
            
            # added the if condition to cut 1 additional unnecessary foward pass
            if restarts > 1:
                all_loss = nn.CrossEntropyLoss(reduction='none')(model(x+delta),y)
                max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
                max_loss = torch.max(max_loss, all_loss)
            else:
                max_delta = delta.detach()
        return max_delta

    def parse_param(self, **kwargs):
        for key,value in kwargs.items():
            if key in self.param:
                self.param[key] = value
            
# class pgd_rand_nn(object):
#     """ PGD attacks, with random initialization within the specified lp ball """
#     def __init__(self, **kwargs):
#         # define default attack parameters here:
#         self.param = {'ord': 2,
#                       'epsilon': 4,
#                       'alpha': 2.5*4/100,
#                       'num_iter': 100,
#                       'restarts': 1,
#                       'loss_fn': nn.CrossEntropyLoss(),
#                       'clip': False}
#         # parse thru the dictionary and modify user-specific params
#         self.parse_param(**kwargs) 
        
#     def generate(self, model, x, y):
#         epsilon = self.param['epsilon']
#         alpha = self.param['alpha']
#         num_iter = self.param['num_iter']
#         restarts = self.param['restarts']
#         loss_fn = self.param['loss_fn']
#         p_norm = self.param['ord']
#         clip = self.param["clip"]
        
#         # implementation begins:
#         max_loss = torch.zeros(y.shape[0]).to(y.device)
#         max_delta = torch.zeros_like(x)

#         _dim = x.shape[1] * x.shape[2] * x.shape[3]
# #         _dim = x.shape[0]
        
#         for i in range(restarts):
# #             if p_norm == np.inf:
# #                 delta = torch.rand_like(x, requires_grad=True)
# #                 delta.data = delta.data * 2. * epsilon - epsilon
# #                 delta.data = (x.data + delta.data).clamp(min = 0, max = 1.0) - x.data
# #                 for t in range(num_iter):
# #                     model.zero_grad()
# #                     loss = loss_fn(model(x + delta), y)
# #                     loss.backward()
# #                     # first we need to make sure delta is within the specified lp ball
# #                     delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(min = -epsilon, max = epsilon)
# #                     # then we need to make sure x+delta in the next iteration is within the [0,1] range
# #                     delta.data = (x.data + delta.data).clamp(min = 0, max = 1.) - x.data
# #                     delta.grad.zero_()

# #             elif p_norm == 2:
#             if p_norm == 2:
#                 delta = torch.rand_like(x, requires_grad = True)
#                 delta.data = delta.data * 2.0 - 1.0
#                 delta_norm = torch.norm(delta.detach(), p = 2 , dim = (1,2,3), keepdim = True).clamp(min = avoid_zero_div)
#                 delta.data = epsilon * delta.data/delta_norm
#                 if clip:
#                     delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data
#                 for t in range(num_iter):
#                     model.zero_grad()
# #                     loss = loss_fn(model(x + delta), y.float().view(-1,1))
# #                     ipdb.set_trace()
#                     if len(y.unique()) > 2:
#                         loss = loss_fn(model(x + delta), y.long())
#                     else:
                        
#                         loss = loss_fn(model(x + delta), y.float().view(-1,1))
#                     loss.backward()

#                     # computing norm of loss gradient wrt input
#                     # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
#                     grad_norm = torch.norm(delta.grad.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
#                     # one step in the direction of normalized gradient (stepsize = alpha)
#                     delta.data = delta + alpha*delta.grad.detach()/grad_norm
#                     # computing the norm of the new delta term
#                     # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
#                     delta_norm = torch.norm(delta.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
#                     # here the clip factor is used to **clip** to within the norm ball
#                     # not to **normalize** onto the surface of the ball
#                     factor = torch.min(epsilon/delta_norm, torch.tensor(1., device = x.device ))

#                     delta.data = delta.data * factor
#                     if clip:
#                         delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data

#                     delta.grad.zero_()
#             else: 
#                 error = "Only ord = inf and ord = 2 have been implemented"
#                 raise NotImplementedError(error)
            
#             # added the if condition to cut 1 additional unnecessary foward pass
#             if restarts > 1:
#                 all_loss = nn.CrossEntropyLoss(reduction='none')(model(x+delta),y)
#                 max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
#                 max_loss = torch.max(max_loss, all_loss)
#             else:
#                 max_delta = delta.detach()
#         return max_delta

#     def parse_param(self, **kwargs):
#         for key,value in kwargs.items():
#             if key in self.param:
#                 self.param[key] = value
                
# class fgsm_nn(object):
#     """ FGSM attack """
#     def __init__(self, **kwargs):
#         # define default attack parameters here:
#         self.param = {'ord': 2,
#                       'epsilon': 4,
#                       'clip': False,
#                       'loss_fn': nn.CrossEntropyLoss()}
#         # parse thru the dictionary and modify user-specific params
#         self.parse_param(**kwargs) 
        
#     def generate(self, model, x, y):
#         epsilon = self.param['epsilon']
#         loss_fn = self.param['loss_fn']
#         p_norm = self.param['ord']
#         clip = self.param['clip']
        
#         # implementation begins:

#         _dim = x.shape[1] * x.shape[2] * x.shape[3]
# #         _dim = x.shape[0]
        
# #         for i in range(restarts):
# #             if p_norm == np.inf:
# #                 delta = torch.rand_like(x, requires_grad=True)
# #                 delta.data = delta.data * 2. * epsilon - epsilon
# #                 delta.data = (x.data + delta.data).clamp(min = 0, max = 1.0) - x.data
# #                 for t in range(num_iter):
# #                     model.zero_grad()
# #                     loss = loss_fn(model(x + delta), y)
# #                     loss.backward()
# #                     # first we need to make sure delta is within the specified lp ball
# #                     delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(min = -epsilon, max = epsilon)
# #                     # then we need to make sure x+delta in the next iteration is within the [0,1] range
# #                     delta.data = (x.data + delta.data).clamp(min = 0, max = 1.) - x.data
# #                     delta.grad.zero_()

# #             elif p_norm == 2:
#         if p_norm == 2:
#             delta = torch.zeros_like(x, requires_grad = True)
# #                 delta.data = delta.data * 2.0 - 1.0
# #                 delta_norm = torch.norm(delta.detach(), p = 2 , dim = (1,2,3), keepdim = True).clamp(min = avoid_zero_div)
# #                 delta.data = epsilon * delta.data/delta_norm
# #                 delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data
# #                 model.zero_grad()
# #                     loss = loss_fn(model(x + delta), y.float().view(-1,1))
# #                     ipdb.set_trace()
#             if len(y.unique()) > 2:
# #                 ipdb.set_trace()
#                 loss = loss_fn(model(x + delta), y.long())
                
#             else:

#                 loss = loss_fn(model(x + delta), y.float().view(-1,1))
#             loss.backward()
# #             ipdb.set_trace()
#             # computing norm of loss gradient wrt input
#             # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
# #                 grad_norm = torch.norm(delta.grad.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
#             delta_norm = torch.norm(delta.grad.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
#             # one step in the direction of normalized gradient (stepsize = alpha)
# #                 delta.data = delta + alpha*delta.grad.detach()/grad_norm
#             delta.data = epsilon*delta.grad.detach()/delta_norm
#             # computing the norm of the new delta term
#             # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
# #                 delta_norm = torch.norm(delta.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
#             # here the clip factor is used to **clip** to within the norm ball
#             # not to **normalize** onto the surface of the ball
# #                 factor = torch.min(epsilon/delta_norm, torch.tensor(1., device = x.device ))

# #                 delta.data = delta.data * factor
#             if clip:
#                 delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data

# #                 delta.grad.zero_()
#         else: 
#             error = "Only ord = inf and ord = 2 have been implemented"
#             raise NotImplementedError(error)
            
#         return delta.detach()

#     def parse_param(self, **kwargs):
#         for key,value in kwargs.items():
#             if key in self.param:
#                 self.param[key] = value
