import torch
import torch.nn as nn

from tqdm import trange
import numpy as np

from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
from src.utils_general import ep2itr
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct, batch_dct2, batch_idct2, getDCTmatrix,batch_dct2_3channel,batch_idct2_3channel
import ipdb

from collections import defaultdict

AVOID_ZERO_DIV = 1e-6

def train_standard_experiments(logger, epoch, loader, model, opt, device):
    total_loss, total_correct = 0., 0.
    curr_itr = ep2itr(epoch, loader)
    dct_matrix = getDCTmatrix(28).to(device)
    w_history = []
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)

            # ipdb.set_trace()
            new_y = torch.rand(len(y)).to(device)
            new_y[y==0] *= -1
            # y[y==0] = torch.rand((y==0).sum().item(), device=device).long()
            # y[y==1] = -1*torch.rand((y==1).sum().item(), device=device).long()

            yp = model(X)
            # ipdb.set_trace()
            if len(loader.dataset.classes) == 2:
                new_y = new_y.float().view(-1,1)
                loss = torch.nn.MSELoss(reduction='mean')(yp,new_y)
                # loss = torch.nn.BCEWithLogitsLoss()(yp, y)
                # batch_correct = ((yp > 0) == (y==1)).sum().item()
                batch_correct = ((yp > 0) == (new_y>0)).sum().item()
            else:
                loss = nn.CrossEntropyLoss()(yp, y)
                batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()
            curr_itr += 1
            if logger is not None:
                logger.add_scalar("train/acc_itr", batch_acc, curr_itr)
                logger.add_scalar("train/loss_itr", loss, curr_itr)

            curr_w = model.conv1.weight.clone().detach()
            num_small_weight = (batch_dct2(curr_w,dct_matrix).abs()<0.05).detach().sum().item()
            w_history.append(num_small_weight)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss, w_history

def train_standard_rotate_in_freq(logger, epoch, loader, model, opt, device):
    total_loss, total_correct = 0., 0.
    curr_itr = ep2itr(epoch, loader)
    dct_matrix = getDCTmatrix(28).to(device)
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)
            X = batch_idct2(torch.rot90(batch_dct2(X.squeeze(),dct_matrix) ,-2, dims=[1,2]),dct_matrix).unsqueeze(1)


            yp = model(X)

            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                loss = torch.nn.BCEWithLogitsLoss()(yp, y)
                batch_correct = ((yp > 0) == (y==1)).sum().item()
            else:
                loss = nn.CrossEntropyLoss()(yp, y)
                batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()
            curr_itr += 1
            if logger is not None:
                logger.add_scalar("train/acc_itr", batch_acc, curr_itr)
                logger.add_scalar("train/loss_itr", loss, curr_itr)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss

def train_standard(logger, epoch, loader, model, opt, device):
    total_loss, total_correct = 0., 0.
    # curr_itr = ep2itr(epoch, loader)
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)

            yp = model(X)

            loss = nn.CrossEntropyLoss()(yp, y)

            batch_correct = (yp.argmax(dim=1) == y).sum().item()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss

def train_regularized_c13(logger, epoch, loader, model, opt, lambbda, conv_only, device):
    total_loss, total_correct = 0., 0.
    curr_itr = ep2itr(epoch, loader)

    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)

            yp = model(X)
            
            w_l2_norm =0
            for param in model.parameters():
                w_l2_norm += ((param)**2).sum()
            w_l2_norm = torch.sqrt(w_l2_norm)
            # regularized conv layer only
            if conv_only:
                conv1_l2_norm = torch.norm(model.conv1.weight,p=2)
                conv2_l2_norm = torch.norm(model.conv2.weight,p=2)
                conv3_l2_norm = torch.norm(model.conv3.weight,p=2)
                conv4_l2_norm = torch.norm(model.conv4.weight,p=2)
                penalty = conv1_l2_norm+conv2_l2_norm+conv3_l2_norm+conv4_l2_norm
            else:
                penalty = w_l2_norm

            loss = nn.CrossEntropyLoss()(yp, y)
            loss_regularized = loss + lambbda * penalty

            batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss_regularized.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss_regularized.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100),
                          w_l2_norm='{0:.2f}'.format(w_l2_norm.item()),
                          penalty='{0:.2f}'.format(penalty.item()))
            t.update()
            curr_itr += 1
    # if logger is not None:
        # # logger.add_scalar("train/acc_itr", batch_acc, curr_itr)
        # # logger.add_scalar("train/loss_itr", loss, curr_itr)
        # logger.add_scalar("train/loss_regularized_itr", loss_regularized, epoch+1)
        # logger.add_scalar("train/w_l2_norm_itr", w_l2_norm.item(), epoch+1)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss

def train_hp_filtered(logger, epoch, loader, model, opt, freq_mask, device):
    total_loss, total_correct = 0., 0.
    
    _dim = freq_mask.shape[0]
    dct_matrix = getDCTmatrix(_dim).to(device)
    if _dim == 28:
        freq_mask = freq_mask.unsqueeze(0).to(device)
    else:
        freq_mask = freq_mask.unsqueeze(0).unsqueeze(0).expand(-1,3,-1,-1).to(device)


    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)
            bs = X.shape[0]

            if _dim == 28:
                X_dct = batch_dct2(X, dct_matrix)*freq_mask.expand(bs,-1,-1)
                X = batch_idct2(X_dct, dct_matrix).unsqueeze(1)
            else:
                X_dct = batch_dct2_3channel(X, dct_matrix)*freq_mask.expand(bs,-1,-1,-1)
                X = batch_idct2_3channel(X_dct, dct_matrix)

            yp = model(X)


            loss = nn.CrossEntropyLoss()(yp, y)
            loss_regularized = loss

            batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss_regularized.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss_regularized.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)

    return acc, total_loss

def train_amp_filtered(logger, epoch, loader, model, opt, threshold, dataset, device):
    #train using the top {threshold} percentile of each input
    total_loss, total_correct = 0., 0.

    if dataset in ['mnist', 'fashionmnist']:
        _dim = 28
    elif dataset in ['cifar10', 'cifar100', 'svhn']:
        _dim = 32
    elif dataset in ['caltech', 'imagenette']:
        _dim = 224

    dct_matrix = getDCTmatrix(_dim).to(device)
    
    assert threshold >=0 and threshold <=100
    
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)
            
            bs = X.shape[0]

            if dataset in ['mnist', 'fashionmnist']:
                X_dct = batch_dct2(X, dct_matrix)
                X_dct_abs_mean = X_dct.abs()
            else:
                X_dct = batch_dct2_3channel(X, dct_matrix)
                X_dct_abs_mean = X_dct.abs().mean(dim=1)

            threshold_for_each_sample = torch.quantile(X_dct_abs_mean.view(-1, _dim*_dim), 1.-threshold/100., dim=1, keepdim=False)
            
            X_threshold = X_dct_abs_mean >= threshold_for_each_sample.view(bs,1,1)

            if dataset in ['mnist', 'fashionmnist']:
                X_dct *= X_threshold
            else:
                X_dct *= X_threshold.unsqueeze(1).expand(-1,3,-1,-1)

            if dataset in ['mnist', 'fashionmnist']:
                X = batch_idct2(X_dct, dct_matrix).unsqueeze(1)
            else:
                X = batch_idct2_3channel(X_dct, dct_matrix)


            yp = model(X)

            loss = nn.CrossEntropyLoss()(yp, y)
            loss_regularized = loss

            batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss_regularized.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss_regularized.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)

    return acc, total_loss

def train_gaussian_speckle_partial(logger, epoch, noise_type, std, loader, model, opt, device):
    
    assert noise_type in ['gaussian', 'speckle']
    
    total_loss, total_correct = 0., 0.
    curr_itr = ep2itr(epoch, loader)
    dct_matrix = getDCTmatrix(28).to(device)
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)
            
            noise = (std**0.5)*torch.randn_like(X)
            dct_noise = batch_dct2(noise, dct_matrix)
            dct_noise[:,:20,:20]=0
            idct_noise = batch_idct2(dct_noise, dct_matrix).unsqueeze(1)

            if noise_type == 'gaussian':
                X = (X+idct_noise).clamp(min=0,max=1.0)
            elif noise_type == 'speckle':
                X = (X +(X*idct_noise)).clamp(min=0,max=1.0)
                
            yp = model(X)

            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                loss = torch.nn.BCEWithLogitsLoss()(yp, y)
                batch_correct = ((yp > 0) == (y==1)).sum().item()
            else:
                loss = nn.CrossEntropyLoss()(yp, y)
                batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()
            curr_itr += 1
            if logger is not None:
                logger.add_scalar("train/acc_itr", batch_acc, curr_itr)
                logger.add_scalar("train/loss_itr", loss, curr_itr)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss

def train_gaussian_speckle_rotate_in_freq(logger, epoch, noise_type, std, loader, model, opt, device):
    
    assert noise_type in ['gaussian', 'speckle']
    
    total_loss, total_correct = 0., 0.
    curr_itr = ep2itr(epoch, loader)
    dct_matrix = getDCTmatrix(28).to(device)
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)
            X = batch_idct2(torch.rot90(batch_dct2(X.squeeze(),dct_matrix) ,-2, dims=[1,2]),dct_matrix).unsqueeze(1)
            
            noise = (std**0.5)*torch.randn_like(X)
            if noise_type == 'gaussian':
                X = (X+noise).clamp(min=0,max=1.0)
            elif noise_type == 'speckle':
                X = (X +(X*noise)).clamp(min=0,max=1.0)
                
            yp = model(X)

            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                loss = torch.nn.BCEWithLogitsLoss()(yp, y)
                batch_correct = ((yp > 0) == (y==1)).sum().item()
            else:
                loss = nn.CrossEntropyLoss()(yp, y)
                batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()
            curr_itr += 1
            if logger is not None:
                logger.add_scalar("train/acc_itr", batch_acc, curr_itr)
                logger.add_scalar("train/loss_itr", loss, curr_itr)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss

def train_gaussian_speckle(logger, epoch, noise_type, std, loader, model, opt, device):
    
    assert noise_type in ['gaussian', 'speckle']
    
    total_loss, total_correct = 0., 0.
    curr_itr = ep2itr(epoch, loader)
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)
            
            noise = (std**0.5)*torch.randn_like(X)
            if noise_type == 'gaussian':
                X = (X+noise).clamp(min=0,max=1.0)
            elif noise_type == 'speckle':
                X = (X +(X*noise)).clamp(min=0,max=1.0)
                
            yp = model(X)

            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                loss = torch.nn.BCEWithLogitsLoss()(yp, y)
                batch_correct = ((yp > 0) == (y==1)).sum().item()
            else:
                loss = nn.CrossEntropyLoss()(yp, y)
                batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()
            curr_itr += 1
            if logger is not None:
                logger.add_scalar("train/acc_itr", batch_acc, curr_itr)
                logger.add_scalar("train/loss_itr", loss, curr_itr)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss

def train_adv(logger, epoch, loader, epsilon, pgd_steps, model, opt, device, norm='l2'):
    total_loss_adv = 0.
    total_correct_adv = 0.

    attack = pgd
    curr_itr = ep2itr(epoch, loader)

    if norm == 'l2':
        param = {"ord":2, "epsilon": epsilon, "alpha":epsilon*2.5/pgd_steps, "num_iter": pgd_steps}
    elif norm == 'linf':
        param = {"ord":np.inf, "epsilon": epsilon, "alpha":0.01, "num_iter": pgd_steps}
    else:
        raise ValueError('incorrect norm specified!')

    if len(loader.dataset.classes) != 2:
        param['loss_fn'] = nn.CrossEntropyLoss()
    else:
        param['loss_fn'] = nn.BCEWithLogitsLoss()

    with trange(len(loader)) as t:
        for X,y in loader:
            model.train()
            X,y = X.to(device), y.to(device)

            with ctx_noparamgrad_and_eval(model):
                delta = attack(**param).generate(model, X, y)

            yp = model(X+delta)
            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                loss_adv = torch.nn.BCEWithLogitsLoss()(yp, y)
                batch_correct_adv = ((yp > 0) == (y==1)).sum().item()
            else:
                loss_adv = nn.CrossEntropyLoss()(yp, y)
                batch_correct_adv = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()
            loss_adv.backward()
            opt.step()

            total_correct_adv += batch_correct_adv

            batch_acc_adv = batch_correct_adv / X.shape[0]
            total_loss_adv += loss_adv.item() * X.shape[0]

            t.set_postfix(loss_adv = loss_adv.item(),
                          acc_adv = '{0:.2f}%'.format(batch_acc_adv*100))
            t.update()
            curr_itr += 1
            if logger is not None:
                logger.add_scalar("train/acc_itr", batch_acc_adv, curr_itr)
                logger.add_scalar("train/loss_itr", loss_adv, curr_itr)

    acc_adv = total_correct_adv / len(loader.dataset) * 100
    total_loss_adv = total_loss_adv / len(loader.dataset)
    return acc_adv, total_loss_adv

def train_two_layer(logger, method, epoch, loader, lambbda, input_d, lr, model, opt, device, lambbda_2nd=0):
    total_loss, total_correct = 0., 0.
    curr_itr = ep2itr(epoch, loader)
    dct_matrix = getDCTmatrix(input_d).to(device)
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)

            yp = model(X)

            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                loss = torch.nn.BCEWithLogitsLoss()(yp, y)
                batch_correct = ((yp > 0) == (y==1)).sum().item()
            else:
                loss = nn.CrossEntropyLoss()(yp, y)
                batch_correct = (yp.argmax(dim=1) == y).sum().item()

            opt.zero_grad()

            if method[:3] == 'l1s':
                curr_w = model.conv1.weight

                l1_reg = torch.norm(curr_w, p=1 ,dim=[2,3]).mean()
                
                if method[-3:] == '2nd':
                    second_layer = model.linear2.weight
                    l1_reg_second = torch.norm(second_layer.squeeze(), p=1 ,dim=[0]).mean()
                    loss_reg = loss + lambbda*l1_reg + lambbda_2nd*l1_reg_second
                else:
                    loss_reg = loss + lambbda*l1_reg

                loss_reg.backward()
                opt.step()
            elif method[:3] == 'l1f':
                curr_w = model.conv1.weight

                
                if curr_w.shape[-1] != input_d:
                    pad_curr_w = nn.ZeroPad2d(int((input_d-curr_w.shape[-1])/2))(curr_w)
                    assert pad_curr_w.shape[-1] == input_d
                    dct_w = batch_dct2(pad_curr_w.squeeze(), dct_matrix)
                else:
                    # dct_w = batch_dct2(curr_w.squeeze(), dct_matrix)
                    dct_w = batch_dct2(curr_w, dct_matrix)

                l1_reg = torch.norm(dct_w, p=1 ,dim=[1,2]).mean()
                
                if method[-3:] == '2nd':
                    second_layer = model.linear2.weight
                    l1_reg_second = torch.norm(second_layer.squeeze(), p=1 ,dim=[0]).mean()
                    loss_reg = loss + lambbda*l1_reg + lambbda_2nd*l1_reg_second
                else:
                    loss_reg = loss + lambbda*l1_reg

                loss_reg.backward()
                opt.step()
            elif method[:4] =='wl1f':
                curr_w = model.conv1.weight
                
                if curr_w.shape[-1] != input_d:
                    pad_curr_w = nn.ZeroPad2d(int((input_d-curr_w.shape[-1])/2))(curr_w)
                    assert pad_curr_w.shape[-1] == input_d
                    dct_w = batch_dct2(pad_curr_w.squeeze(), dct_matrix)
                else:
                    dct_w = batch_dct2(curr_w, dct_matrix)

                mean_abs_x_tilde = batch_dct2(X, dct_matrix).abs().mean(dim=0)
                decay_factor = mean_abs_x_tilde/mean_abs_x_tilde.max()
                M = (1/(decay_factor**2+AVOID_ZERO_DIV)).unsqueeze(0) #Used to be just 1/decay_factor, now we are using 1/decay_factor**2

                l1_reg = torch.norm(torch.mul(M,dct_w), p=1 ,dim=[1,2]).mean()
                
                if method[-3:] == '2nd':
                    second_layer = model.linear2.weight
                    l1_reg_second = torch.norm(second_layer.squeeze(), p=1 ,dim=[0]).mean()
                    loss_reg = loss + lambbda*l1_reg + lambbda_2nd*l1_reg_second
                else:
                    loss_reg = loss + lambbda*l1_reg

                loss_reg.backward()
                opt.step()
            elif method[:3] =='wlr':
                loss.backward()

                curr_w = model.conv1.weight.clone().detach()
                grad = model.conv1.weight.grad.clone().detach()
                
                if grad.shape[-1] != input_d:
                    pad_size = int((input_d-grad.shape[-1])/2)
                    pad_grad = nn.ZeroPad2d(pad_size)(grad)
                    assert pad_grad.shape[-1] == input_d
                    dct_grad = batch_dct2(pad_grad.squeeze(), dct_matrix)
                else:
                    dct_grad = batch_dct2(grad, dct_matrix)

                mean_abs_x_tilde = batch_dct2(X, dct_matrix).abs().mean(dim=0)
                decay_factor = mean_abs_x_tilde/mean_abs_x_tilde.max()
                M = (1/(decay_factor**2+AVOID_ZERO_DIV)).unsqueeze(0) #Used to be just 1/decay_factor, now we are using 1/decay_factor**2
                
                if grad.shape[-1] != input_d:
                    new_grad = batch_idct2(torch.mul(M,dct_grad),dct_matrix).unsqueeze(1)
                    new_grad = new_grad[:,:,pad_size:-pad_size,pad_size:-pad_size]
                    opt.step()
                else:
                    new_grad = batch_idct2(torch.mul(M,dct_grad),dct_matrix).unsqueeze(1)

                
                new_w = curr_w - lr * new_grad

                model.conv1.weight = torch.nn.parameter.Parameter(new_w)
            elif method == 'exp':
                curr_w = model.conv1.weight
                
                if curr_w.shape[-1] != input_d:
                    pad_curr_w = nn.ZeroPad2d(int((input_d-curr_w.shape[-1])/2))(curr_w)
                    assert pad_curr_w.shape[-1] == input_d
                    dct_w = batch_dct2(pad_curr_w.squeeze(), dct_matrix)
                else:
                    dct_w = batch_dct2(curr_w.squeeze(), dct_matrix)
                    
                    
                curr_w = model.conv2.weight
                
                if curr_w.shape[-1] != input_d:
                    pad_curr_w = nn.ZeroPad2d(int((input_d-curr_w.shape[-1])/2))(curr_w)
                    assert pad_curr_w.shape[-1] == input_d
                    dct_w = batch_dct2(pad_curr_w.squeeze(), dct_matrix)
                else:
                    dct_w = batch_dct2(curr_w.squeeze(), dct_matrix)

                l1_reg = torch.norm(dct_w.squeeze(), p=1 ,dim=[1,2]).mean()
                loss_reg = loss + lambbda*l1_reg

                loss_reg.backward()
                opt.step()
            else:
                raise ValueError('Incorrect method!')

            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()
            curr_itr += 1
            if logger is not None:
                logger.add_scalar("train/acc_itr", batch_acc, curr_itr)
                logger.add_scalar("train/loss_itr", loss, curr_itr)
                if method in ['l1f', 'l1s', 'wl1f']:
                    logger.add_scalar("train/reg_itr", lambbda*l1_reg.item(), curr_itr)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)
    return acc, total_loss

def train_two_layer_two_neuron(loader, args, model, opt, device):
    #two-layer two neuron settings , binary synthetic case, binaryMNIST case
    
    _case = args["case"]
    _iteration = args["itr"]
    _input_d = args["input_d"]
    _output_d = args["output_d"]
    _hidden_d = args["hidden_d"]
    _lr = args["lr"]
    _method = args['method']
    
    log_dict = defaultdict(lambda: list())
    
    loss_logger = torch.zeros(1, _iteration, device = device)
    acc_logger = torch.zeros(1, _iteration, device = device)
    
    dct_matrix = getDCTmatrix(_input_d).to(device)
    theta_tilde_logger = torch.zeros(_input_d, _input_d, 2, device = device)
    
    
    if _method == "adv":
        param = {'ord': 2,
                 'epsilon': 2,
                 'alpha': 2,
                 'num_iter': 1,
                 'restarts': 1,
                 'loss_fn': torch.nn.BCEWithLogitsLoss(),
                 'rand_init': True,
                 'clip': True if _case in ["binaryMNIST", 'MNIST'] else False}
        param['num_iter'] = args['num_iter']
        param['alpha'] = 2.0*2.5/args['num_iter']   
    
    
    i = 0
    with trange(_iteration) as t:
        while(i < _iteration):
            for x, y in loader:
                    
                x, y = x.to(device), y.to(device)
                x_len = x.shape[0]
                opt.zero_grad()
                
                
                if _method == "adv":
                    delta = pgd(**param).generate(model, x, y)
                    y_hat = model(x+delta)
                else:   
                    y_hat = model(x)

                y = y.float().view(-1,1)
#                 loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
#                 batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
                batch_correct = ((y_hat > 0) == (y==1)).sum().item()

                loss_logger[:,i] = loss.item()
                batch_acc = batch_correct / x_len * 100
                
                acc_logger[:,i] = batch_acc
            
            
            
            
                if _method in ["std", 'adv']:
                    loss.backward()
                    opt.step()
                elif _method == "W-LR":
                    loss.backward()
                    curr_w = model.conv1.weight.clone().detach()
                    dct_w = batch_dct2(curr_w.squeeze(), dct_matrix)
                    grad = model.conv1.weight.grad.clone().detach()
                    dct_grad = batch_dct2(grad.squeeze(), dct_matrix)
            
                    AVOID_ZERO_DIV = 1e-6
                    mean_abs_x_tilde = batch_dct2(x, dct_matrix).abs().mean(dim=0)
                    if args["case"] == "binaryMNIST":
                        decay_factor = mean_abs_x_tilde/mean_abs_x_tilde.max()
                    else:
                        decay_factor = mean_abs_x_tilde/mean_abs_x_tilde[:10,:10].mean()
                    M = (1/(decay_factor+AVOID_ZERO_DIV)).unsqueeze(0)
                    new_grad = batch_idct2(torch.mul(M,dct_grad),dct_matrix).unsqueeze(1)
                    model.conv1.weight = torch.nn.parameter.Parameter(new_w)
                
                elif _method == "W-L1F":
                    curr_w = model.conv1.weight

                    dct_w = batch_dct2(curr_w.squeeze(), dct_matrix)
                    AVOID_ZERO_DIV = 1e-6
                    mean_abs_x_tilde = batch_dct2(x, dct_matrix).abs().mean(dim=0)
                    if args["case"] == "binaryMNIST":
                        decay_factor = mean_abs_x_tilde/mean_abs_x_tilde.max()
                    else:
                        decay_factor = mean_abs_x_tilde/mean_abs_x_tilde[:10,:10].mean()
                    M = (1/(decay_factor+AVOID_ZERO_DIV)).view(1,_input_d, _input_d)
                    ipdb.set_trace()
                    l1_reg = torch.norm(torch.mul(M,dct_w).squeeze(), p =1,dim=[1,2]).mean()
                    loss_reg = loss+args['factor']*l1_reg
                    loss_reg.backward()
                    opt.step()

                elif _method == "L1F":
                    curr_w = model.conv1.weight

                    dct_w = batch_dct2(curr_w.squeeze(), dct_matrix)
                    
                    l1_reg = torch.norm(dct_w.squeeze(), p =1,dim=[1,2]).mean()
                    loss_reg = loss+args['factor']*l1_reg
                    loss_reg.backward()
                    opt.step()
                elif _method == "L1S":
                    curr_w = model.conv1.weight
                    l1_reg = torch.norm(curr_w.squeeze(), p =1,dim=[1,2]).mean()
                    loss_reg = loss+args['factor']*l1_reg
                    loss_reg.backward()
                    opt.step()
                else:
                    raise NotImplemented("method not impelmeneted")

                i += 1
                
                t.set_postfix(loss = loss.item(),
                              acc = '{0:.2f}%'.format(batch_acc))
                t.update()
                if i == _iteration:
                    break
    
    theta_tilde_logger[:,:,0] = dct2(model.state_dict()['conv1.weight'][0,0,:,:])
    theta_tilde_logger[:,:,1] = dct2(model.state_dict()['conv1.weight'][1,0,:,:])
    log_dict["loss"] = loss_logger
    log_dict["acc"] = acc_logger
    log_dict["theta_tilde"] = theta_tilde_logger

    return log_dict

def train_two_layer_MNIST(loader, args, model, opt, device):
    #two-layer nn for full MNIST
    
    _case = args["case"]
    _iteration = args["itr"]
    _input_d = args["input_d"]
    _output_d = args["output_d"]
    _hidden_d = args["hidden_d"]
    _lr = args["lr"]
    _method = args['method']
    
    log_dict = defaultdict(lambda: list())
    
    loss_logger = torch.zeros(1, _iteration, device = device)
    acc_logger = torch.zeros(1, _iteration, device = device)
    
    dct_matrix = getDCTmatrix(_input_d).to(device)
#     theta_tilde_logger = torch.zeros(_input_d, _input_d, 2, device = device)

    if _method == "adv":
        param = {'ord': 2,
                 'epsilon': args['eps'],
                 'alpha': 2,
                 'num_iter': 1,
                 'restarts': 1,
                 'loss_fn': torch.nn.CrossEntropyLoss(),
                 'rand_init': True,
                 'clip': True if _case in ["binaryMNIST", 'MNIST','fashionMNIST'] else False}
        param['num_iter'] = args['num_iter']
        param['alpha'] = args['eps']*2.5/args['num_iter']   
    
    i = 0
    with trange(_iteration) as t:
        while(i < _iteration):
            for x, y in loader:
                    
                x, y = x.to(device), y.long().to(device)
                x_len = x.shape[0]
                opt.zero_grad()

                if _method == "adv":
                    delta = pgd(**param).generate(model, x, y)
                    y_hat = model(x+delta)
                else:   
                    y_hat = model(x)
                
                
                loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()

                loss_logger[:,i] = loss.item()
                batch_acc = batch_correct / x_len * 100
                
                acc_logger[:,i] = batch_acc
                
                if _method in ["std", 'adv']:
                    loss.backward()
                    opt.step()
                elif _method == "W-LR":
                    loss.backward()
                    # ipdb.set_trace()
                    curr_w = model.conv1.weight.clone().detach()
                    dct_w = batch_dct2(curr_w.squeeze(), dct_matrix)

                    grad = model.conv1.weight.grad.clone().detach()
                    dct_grad = batch_dct2(grad.squeeze(), dct_matrix)

                    AVOID_ZERO_DIV = 1e-6
                    mean_abs_x_tilde = batch_dct2(x, dct_matrix).abs().mean(dim=0)
                    decay_factor = mean_abs_x_tilde/mean_abs_x_tilde.max()
                    M = (1/(decay_factor+AVOID_ZERO_DIV)).view(1,_input_d, _input_d)
                    # M = torch.ones_like(M)
                    new_grad = batch_idct2(_lr * torch.mul(M, dct_grad) ,dct_matrix).unsqueeze(1)
                    new_w = curr_w - new_grad
                    # new_w = curr_w - batch_idct2(_lr * torch.mul(M,dct_grad),dct_matrix).view(_hidden_d,1,_input_d,_input_d)

                    model.conv1.weight = torch.nn.parameter.Parameter(new_w)
                
                elif _method == "W-L1F":
                    curr_w = model.conv1.weight

                    dct_w = batch_dct2(curr_w.squeeze(), dct_matrix)
                    AVOID_ZERO_DIV = 1e-6
                    mean_abs_x_tilde = batch_dct2(x, dct_matrix).abs().mean(dim=0)
                    decay_factor = mean_abs_x_tilde/mean_abs_x_tilde.max()
                    M = (1/(decay_factor+AVOID_ZERO_DIV)).view(1,_input_d, _input_d)
                    
                    l1_reg = torch.norm(torch.mul(M,dct_w).squeeze(), p =1)
                    loss_reg = loss+args['factor']*l1_reg
                    loss_reg.backward()
                    opt.step()

                elif _method == "L1F":
                    curr_w = model.conv1.weight

                    dct_w = batch_dct2(curr_w.squeeze(), dct_matrix)
                    
                    l1_reg = torch.norm(dct_w.squeeze(), p =1)
                    loss_reg = loss+args['factor']*l1_reg
                    loss_reg.backward()
                    opt.step()
                elif _method == "L1S":
                    curr_w = model.conv1.weight
                    l1_reg = torch.norm(curr_w.squeeze(), p =1)
                    loss_reg = loss+args['factor']*l1_reg
                    loss_reg.backward()
                    opt.step()
                else:
                    raise NotImplementedError("method not impelmeneted")

                i += 1
                
                # t.set_postfix(loss = loss.item(),
                              # acc = '{0:.2f}%'.format(batch_acc))
                t.set_postfix(loss = loss.item(),
                              acc = '{0:.2f}%'.format(batch_acc),
                              norm = '{0:.2f}%'.format(torch.norm(dct_w.squeeze(),p=1)))
                t.update()
                if i == _iteration:
                    break
    
#     theta_tilde_logger[:,:,0] = dct2(model.state_dict()['conv1.weight'][0,0,:,:])
#     theta_tilde_logger[:,:,1] = dct2(model.state_dict()['conv1.weight'][1,0,:,:])
    log_dict["loss"] = loss_logger
    log_dict["acc"] = acc_logger
#     log_dict["theta_tilde"] = theta_tilde_logger

    return log_dict


def train_NN_real(loader, args, model, opt, log_theta_tilde, device):
    
    _case = args["case"]
    _iteration = args["itr"]
    _output_d = args["output_d"]
    _input_d = args["input_d"]
    _hidden_d = args["hidden_d"]
    _eps = args["eps"]
    

    log_dict = defaultdict(lambda: list())
    
    if log_theta_tilde:
        numb_theta_logged = log_theta_tilde
        theta_tilde_logger = torch.zeros(numb_theta_logged, _input_d, _input_d, _iteration)
        dct_matrix = getDCTmatrix(_input_d).to(device)
        random_theta_m_index = torch.randint(low=0, high = _hidden_d, size = (numb_theta_logged,))
        while len(random_theta_m_index.unique()) != numb_theta_logged:
            random_theta_m_index = torch.randint(low=0, high = _hidden_d, size = (numb_theta_logged,))
    
    loss_logger = torch.zeros(1, _iteration, device = device)
    acc_logger = torch.zeros(1, _iteration, device = device)
    u_logger = torch.zeros(2, _iteration, device = device)
    
    
    i = 0
    with trange(_iteration) as t:
        while(i < _iteration):
            for x, y in loader:
                
                if log_theta_tilde:
#                     ipdb.set_trace()
                    theta = model.state_dict()["conv1.weight"].detach()[random_theta_m_index,:,:,:]
                    theta_tilde_logger[:,:,:,i] =  batch_dct2(theta, dct_matrix)
#                     u_logger[:,i] = model.state_dict()['linear2.weight'].squeeze().detach()

                x, y = x.to(device), y.to(device)
                x_len = x.shape[0]
                
                
                opt.zero_grad()
                
#                 if adv: 
#                     delta = pgd_rand_nn(**param).generate(model, x, y)
#                     y_hat = model(x+delta)
#                 else:
                y_hat = model(x)

                if len(loader.dataset.classes) == 2:
                    y = y.float().view(-1,1)
                    loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                    batch_correct = ((y_hat > 0) == (y==1)).sum().item()
                    
                else:
                    y = y.long()
                    loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                    batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()

                loss_logger[:,i] = loss.item()
                batch_acc = batch_correct / x_len * 100
                acc_logger[:,i] = batch_acc

                loss.backward()
                opt.step()

                
                i += 1
                
                t.set_postfix(loss = loss.item(),
                              acc = '{0:.2f}%'.format(batch_acc))
                t.update()
                if i == _iteration:
                    break
    
    log_dict["loss"] = loss_logger
    log_dict["acc"] = acc_logger
    if log_theta_tilde:
        log_dict["theta_tilde"] = theta_tilde_logger
    log_dict["u"] = u_logger
    
    return log_dict

def train_NN_real_lazy(loader, args, model, opt, l1, device):
    
    _iteration = args["itr"]
    _output_d = args["output_d"]
    _input_d = args["input_d"]
    _hidden_d = args["hidden_d"]
#     _eps = args["eps"]
    
    log_dict = defaultdict(lambda: list())
    
#     if log_theta_tilde:
#         numb_theta_logged = log_theta_tilde
#         theta_tilde_logger = torch.zeros(numb_theta_logged, _input_d, _input_d, _iteration)
    dct_matrix = getDCTmatrix(_input_d).to(device)
#         random_theta_m_index = torch.randint(low=0, high = _hidden_d, size = (numb_theta_logged,))
#         while len(random_theta_m_index.unique()) != numb_theta_logged:
#             random_theta_m_index = torch.randint(low=0, high = _hidden_d, size = (numb_theta_logged,))
    
    loss_logger = torch.zeros(1, _iteration, device = device)
    acc_logger = torch.zeros(1, _iteration, device = device)
    u_logger = torch.zeros(2, _iteration, device = device)
    theta_logger = torch.zeros(5, _iteration, device = device)
    
    new_theta_tilde_logger = torch.zeros(2,784, _iteration, device = device)
        
    u_prev = model.state_dict()['linear2.weight'].squeeze().clone().detach()
    theta_unflattened = model.state_dict()["conv1.weight"].clone().detach()
    log_dict["init_theta_tilde_unflattened"] = batch_dct2(theta_unflattened, dct_matrix)

    theta_prev = torch.flatten(theta_unflattened, start_dim=1, end_dim=-1)
    theta_tilde_prev = torch.flatten(batch_dct2(theta_unflattened, dct_matrix), start_dim=1, end_dim=-1)
    
    u_init = u_prev.clone().detach()
    theta_init = theta_prev.clone().detach()
    theta_tilde_init = theta_tilde_prev.clone().detach()
    new_theta_tilde_logger[:,:,0] = theta_tilde_prev
    
    i = 0
    with trange(_iteration) as t:
        while(i < _iteration):
            for x, y in loader:
                
#                 if log_theta_tilde:
# #                     ipdb.set_trace()
#                     theta = model.state_dict()["conv1.weight"].detach()[random_theta_m_index,:,:,:]
#                     theta_tilde_logger[:,:,:,i] =  batch_dct2(theta, dct_matrix)
#                     u_logger[:,i] = model.state_dict()['linear2.weight'].squeeze().detach()

                x, y = x.to(device), y.to(device)
                x_len = x.shape[0]
                
                
                opt.zero_grad()
                
#                 if adv: 
#                     delta = pgd_rand_nn(**param).generate(model, x, y)
#                     y_hat = model(x+delta)
#                 else:
                y_hat = model(x)

                if len(loader.dataset.classes) == 2:
                    y = y.float().view(-1,1)
                    loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                    batch_correct = ((y_hat > 0) == (y==1)).sum().item()
                    
                else:
                    y = y.long()
                    loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                    batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()

                if l1:
                    for p in model.parameters():
                        theta_tilde = batch_dct2(p, dct_matrix)
#                         theta_tilde = p
                        break
                    l1_reg = torch.norm(theta_tilde, p=1)

#                     factor = 0.0005
                    factor = 0.01
                    loss += factor * l1_reg
                else:
                    l1_reg = torch.zeros(1)
                
                loss_logger[:,i] = loss.item()
                batch_acc = batch_correct / x_len * 100
                acc_logger[:,i] = batch_acc
                
                loss.backward()
                opt.step()
                
                
                u_curr = model.state_dict()['linear2.weight'].squeeze().clone().detach()
                theta_unflattened = model.state_dict()["conv1.weight"].clone().detach()
#                 print(u_curr, theta_unflattened)
                theta_curr = torch.flatten(theta_unflattened, start_dim=1, end_dim=-1)
                theta_tilde_curr = torch.flatten(batch_dct2(theta_unflattened, dct_matrix), start_dim=1, end_dim=-1)
                new_theta_tilde_logger[:,:,i] = theta_tilde_curr
                
#                 ipdb.set_trace()
                
                u_logger[0,i] = (u_prev-u_curr).abs().mean()
                u_logger[1,i] = (u_init-u_curr).abs().mean()
                theta_logger[0,i] = torch.norm((theta_prev-theta_curr).detach(), p =2 , dim = 1).mean()
                theta_logger[1,i] = torch.norm((theta_tilde_prev-theta_tilde_curr).detach(), p =2, dim = 1).mean()
                theta_logger[2,i] = torch.norm((theta_init-theta_curr).detach(), p =2 , dim = 1).mean()
                theta_logger[3,i] = torch.norm((theta_tilde_init-theta_tilde_curr).detach(), p =2, dim = 1).mean()
                
                u_prev = u_curr
                theta_prev = theta_curr
                theta_tilde_prev = theta_tilde_curr

                t.set_postfix(loss = loss.item(), acc = '{0:.2f}%'.format(batch_acc), norm = l1_reg.item())
                t.update()
                i += 1
                if i == _iteration:
                    break
    
    log_dict["loss"] = loss_logger
    log_dict["acc"] = acc_logger
    log_dict["theta_tilde"] = theta_logger
    
    log_dict["final_theta_tilde_unflattened"] = batch_dct2(theta_unflattened, dct_matrix)
    log_dict["new_theta_tilde"] = new_theta_tilde_logger
    log_dict["u"] = u_logger
    
    return log_dict


# def train_two_layer(loader, method, args, model, opt, device):
    
    # _iteration = args["itr"]
    # _output_d = args["output_d"]
    # _input_d = args["input_d"]
    # _hidden_d = args["hidden_d"]
    # _lambda = args['lambda']
    
    # log_dict = defaultdict(lambda: list())
    
    # dct_matrix = getDCTmatrix(_input_d).to(device)
    
    # loss_logger = torch.zeros(1, _iteration, device = device)
    # loss_only_logger = torch.zeros(1, _iteration, device = device)
    # acc_logger = torch.zeros(1, _iteration, device = device)
    # reg_logger = torch.zeros(1, _iteration, device = device)
    
    # if method == 'adv':
        # param =  {'ord': args['ord'],
                        # 'epsilon': args['eps'],
                        # 'alpha': args['alpha'],
                        # 'num_iter': args['num_iter'],
                        # 'clip': args['clip'],
                        # 'loss_fn': args['loss_fn']}
    # i = 0
    # with trange(_iteration) as t:
        # while(i < _iteration):
            # for x, y in loader:
                
                # x, y = x.to(device), y.to(device)
                
                # opt.zero_grad()
                
                # if method == 'adv_fgsm':
                    # delta = fgsm_nn(**fgsm_param).generate(model, x, y)
                    # y_hat = model(x+delta)
                # elif method == 'adv_pgd':
                    # delta = pgd(**pgd_param).generate(model, x, y)
                    # y_hat = model(x+delta)
                # elif method in ['std', 'l1s', 'l1f', 'l2s', 'l2f']:
                    # y_hat = model(x)

                # if len(loader.dataset.classes) == 2:
                    # y = y.float().view(-1,1)
                    # loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                    # batch_correct = ((y_hat > 0) == (y==1)).sum().item()
                # else:
                    # y = y.long()
                    # loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                    # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
                
# #                 factor = 0.01 # for binary mnist (sigmoid)
# #                 factor = 0.1 # for binary mnist (relu)
                # factor = _lambda
                # if method in ['l1f', 'l1s', 'l2f', 'l2s']:
                    # for p in model.parameters():
                        # theta = p
                        # break
                    
                    # assert int(method[1]) in [1,2]
                    # if method[-1] == 'f':
                        # theta_tilde = batch_dct2(theta, dct_matrix)
                        # reg = torch.norm(theta_tilde, p = int(method[1]))
                    # elif method[-1] =='s':
                        # reg = torch.norm(theta, p = int(method[1]))
                        
                    # reg_logger[:,i] = reg.item()
                    # loss_only_logger[:,i] = loss.item()
                    # loss += factor * reg

                # loss_logger[:,i] = loss.item()
                # batch_acc = batch_correct / x.shape[0] * 100
                # acc_logger[:,i] = batch_acc
                
                # loss.backward()
                # opt.step()

                # t.set_postfix(loss = loss.item(), acc = '{0:.2f}%'.format(batch_acc))
                # t.update()
                # i += 1
                # if i == _iteration:
                    # break
    
    # log_dict["loss"] = loss_logger
    # log_dict["acc"] = acc_logger
    # log_dict["reg"] = reg_logger
    # log_dict["loss_only"] = loss_only_logger
    
    # return log_dict
