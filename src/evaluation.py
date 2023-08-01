import torch
import torch.nn as nn
from src.attacks import pgd
from src.context import ctx_noparamgrad_and_eval
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct, batch_dct2, batch_idct2, getDCTmatrix, mask_radial, batch_idct2_3channel, batch_dct2_3channel, mask_radial_multiple_radius
import ipdb
from tqdm import trange
from autoattack import AutoAttack
import numpy as np

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test_clean_experiments(loader, model, device):
    total_loss, total_correct = 0., 0.
    for x,y in loader:
        model.eval()
        if len(x.shape) ==2:
            x, y = x.t().to(device), y.t().to(device)
        else:
            x, y = x.to(device), y.to(device)
            new_y = torch.rand(len(y)).to(device)
            new_y[y==0] *= -1

        with torch.no_grad():
            y_hat = model(x)
            if len(loader.dataset.classes) == 2:
                new_y = new_y.float().view(-1,1)
                # loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                loss = torch.nn.MSELoss(reduction='mean')(y_hat,new_y)
                # batch_correct = ((y_hat > 0) == (y==1)).sum().item()
                batch_correct = ((y_hat > 0) == (new_y>0)).sum().item()
            else:
                y = y.long()
                loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
        
        total_correct += batch_correct
        total_loss += loss.item() * x.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss

def test_clean(loader, model, device):
    total_loss, total_correct = 0., 0.
    total_correct_5 = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            y_hat = model(x)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            batch_acc = accuracy(y_hat, y, topk=(1,5))
            batch_correct = batch_acc[0].sum().item()*x.shape[0]/100
            batch_correct_5 = batch_acc[1].sum().item()*x.shape[0]/100
        # print(accuracy(y_hat, y, topk=(1,5)), batch_correct/128*100)
        
        total_correct += batch_correct
        total_correct_5 += batch_correct_5
        total_loss += loss.item() * x.shape[0]
    # ipdb.set_trace()
    test_acc = total_correct / len(loader.dataset) * 100
    test_acc_5 = total_correct_5 / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss, test_acc_5

def test_gaussian(loader, model, var, device):
    total_loss, total_correct = 0., 0.
    total_correct_5 = 0
    for x,y in loader:
        model.eval()
        x, y = x.to(device), y.to(device)
        
        noise = (var**0.5)*torch.randn_like(x, device = x.device)

        with torch.no_grad():
            x_noise = (x+noise).clamp(min=0.,max=1.)
            y_hat = model(x_noise)
            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
            batch_acc = accuracy(y_hat, y, topk=(1,5))
#             ipdb.set_trace()
            batch_correct = batch_acc[0].item()*x.shape[0]/100
            batch_correct_5 = batch_acc[1].item()*x.shape[0]/100
        
        total_correct += batch_correct
        total_correct_5 += batch_correct_5
        total_loss += loss.item() * x.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_acc_5 = total_correct_5 / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss, test_acc_5

def test_gaussian_LF_HF(loader, dataset, model, var, radius, num_noise, device):
    
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
    
    total_loss = 0
    total_loss_LF = 0
    total_loss_HF = 0
    total_correct = 0
    total_correct_LF = 0
    total_correct_HF = 0
    total_samples = 0
    CELoss_sum = torch.nn.CrossEntropyLoss(reduction = 'sum')
    CELoss_mean = torch.nn.CrossEntropyLoss(reduction = 'mean')
    
    for x,y in loader:
        total_samples += x.shape[0]
        model.eval()
        x, y = x.to(device), y.to(device)
        
        mask = _mask.expand(x.shape[0], channel, img_size, img_size)

        for _n in range(num_noise):
            noise = (var**0.5)*torch.randn_like(x, device = x.device)
            if dataset in ['mnist', 'fashionmnist']:
                noise_dct = batch_dct2(noise, dct_matrix).unsqueeze(1)
                noise_LF = batch_idct2(noise_dct * mask, dct_matrix).unsqueeze(1)
                noise_HF = batch_idct2(noise_dct * (1-mask), dct_matrix).unsqueeze(1)
            else:
                noise_dct = batch_dct2_3channel(noise, dct_matrix)
                noise_LF = batch_idct2_3channel(noise_dct * mask, dct_matrix)
                noise_HF = batch_idct2_3channel(noise_dct * (1-mask), dct_matrix)
            with torch.no_grad(): 
                y_hat_LF = model(x+noise_LF)
                y_hat_HF = model(x+noise_HF)
                total_loss_LF += CELoss_sum(y_hat_LF, y)
                total_correct_LF += accuracy(y_hat_LF, y, topk=(1,))[0]*x.shape[0]/100
                total_loss_HF += CELoss_sum(y_hat_HF, y)
                total_correct_HF += accuracy(y_hat_HF, y, topk=(1,))[0]*x.shape[0]/100
                if _n ==0:
                    loss = CELoss_sum(model(x), y)
                    correct = accuracy(model(x), y, topk=(1,))[0]*x.shape[0]/100

        total_loss += loss
        total_correct += correct
        
    total_loss /= total_samples
    total_loss_LF /= (total_samples * num_noise)
    total_loss_HF /= (total_samples * num_noise)
    
    total_acc = total_correct/total_samples*100
    total_acc_LF = total_correct_LF/num_noise/total_samples*100
    total_acc_HF = total_correct_HF/num_noise/total_samples*100

    return [total_loss, total_loss_LF, total_loss_HF], [total_acc, total_acc_LF, total_acc_HF]

def test_gaussian_LF_HF_v2(loader, dataset, model, var, radius_list, num_noise, device):
    
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
    elif dataset in ['imagenette','caltech']:
        img_size = 224
        channel = 3
        dct_matrix = getDCTmatrix(224)
    _mask = torch.tensor(mask_radial_multiple_radius(img_size, radius_list), 
                        device = device, 
                        dtype=torch.float32)
    
    total_loss = 0
    total_loss_noise = np.zeros(len(radius_list)+1)
    total_correct = 0
    total_correct_noise = np.zeros(len(radius_list)+1)
    total_samples = 0
    CELoss_sum = torch.nn.CrossEntropyLoss(reduction = 'sum')
    CELoss_mean = torch.nn.CrossEntropyLoss(reduction = 'mean')
    
    with torch.no_grad(): 
        for x,y in loader:
            total_samples += x.shape[0]
            model.eval()
            x, y = x.to(device), y.to(device)

            mask = _mask.expand(x.shape[0], channel, img_size, img_size)

            for _n in range(num_noise):
                _noise = (var**0.5)*torch.randn_like(x, device = x.device)
                for k in range(len(radius_list)+1):
                    if dataset in ['mnist', 'fashionmnist']:
                        noise_dct = batch_dct2(_noise, dct_matrix).unsqueeze(1)
                        noise = batch_idct2(noise_dct * (mask == k), dct_matrix).unsqueeze(1)
    #                     ipdb.set_trace()
                    else:
                        noise_dct = batch_dct2_3channel(_noise, dct_matrix)
                        noise = batch_idct2_3channel(noise_dct * (mask == k), dct_matrix)

#                     ipdb.set_trace()
                    y_hat_noise = model(x+noise)
                    total_loss_noise[k] += CELoss_sum(y_hat_noise, y)
                    total_correct_noise[k] += (accuracy(y_hat_noise, y, topk=(1,))[0]*x.shape[0]/100)

                if _n ==0:
                    loss = CELoss_sum(model(x), y)
                    correct = accuracy(model(x), y, topk=(1,))[0]*x.shape[0]/100
                    
            total_loss += loss
            total_correct += correct
        
    total_loss /= total_samples
    total_loss_noise /= (total_samples * num_noise)
    
    total_acc = total_correct/total_samples*100
    total_acc_noise = total_correct_noise/num_noise/total_samples*100

    return [total_loss.item(), total_loss_noise], [total_acc.item(), total_acc_noise]

def test_gaussian_LF_HF_v3(loader, dataset, model, var, num_noise, device):
    
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
    
    total_loss = 0
    total_loss_noise = np.zeros(img_size)
    total_correct = 0
    total_correct_noise = np.zeros(img_size)
    total_samples = 0
    CELoss_sum = torch.nn.CrossEntropyLoss(reduction = 'sum')
    CELoss_mean = torch.nn.CrossEntropyLoss(reduction = 'mean')
    
    with torch.no_grad(): 
        for x,y in loader:
            total_samples += x.shape[0]
            model.eval()
            x, y = x.to(device), y.to(device)

            for _n in range(num_noise):
                _noise = (var**0.5)*torch.randn_like(x, device = x.device)
                for k in range(img_size):
                    if dataset in ['mnist', 'fashionmnist']:
                        noise_dct = batch_dct2(_noise, dct_matrix).unsqueeze(1)
                        tmp_noise_dct = torch.zeros_like(noise_dct)
                        tmp_noise_dct[:,:,k,k] = noise_dct[:,:,k,k]
                        noise = batch_idct2(tmp_noise_dct, dct_matrix).unsqueeze(1)
                    else:
                        noise_dct = batch_dct2_3channel(_noise, dct_matrix)
                        tmp_noise_dct = torch.zeros_like(noise_dct)
                        tmp_noise_dct[:,:,k,k] = noise_dct[:,:,k,k]
                        noise = batch_idct2_3channel(tmp_noise_dct, dct_matrix)

                    y_hat_noise = model(x+noise)
                    total_loss_noise[k] += CELoss_sum(y_hat_noise, y)
                    total_correct_noise[k] += (accuracy(y_hat_noise, y, topk=(1,))[0]*x.shape[0]/100)

                if _n ==0:
                    loss = CELoss_sum(model(x), y)
                    correct = accuracy(model(x), y, topk=(1,))[0]*x.shape[0]/100
                    
            total_loss += loss
            total_correct += correct
        
    total_loss /= total_samples
    total_loss_noise /= (total_samples * num_noise)
    
    total_acc = total_correct/total_samples*100
    total_acc_noise = total_correct_noise/num_noise/total_samples*100

    return [total_loss.item(), total_loss_noise], [total_acc.item(), total_acc_noise]

def test_clean_rotate_in_freq(loader, model, device):
    total_loss, total_correct = 0., 0.
    dct_matrix = getDCTmatrix(28).to(device)
    for x,y in loader:
        model.eval()
        if len(x.shape) ==2:
            x, y = x.t().to(device), y.t().to(device)
        else:
            x, y = x.to(device), y.to(device)
        x = batch_idct2(torch.rot90(batch_dct2(x.squeeze(),dct_matrix) ,-2, dims=[1,2]),dct_matrix).unsqueeze(1)

        with torch.no_grad():
            y_hat = model(x)
            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                batch_correct = ((y_hat > 0) == (y==1)).sum().item()
            else:
                y = y.long()
                loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
        
        total_correct += batch_correct
        total_loss += loss.item() * x.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss

def test_transfer_adv(loader, transferred_model, attacked_model, attack, param, device):
    total_loss, total_correct = 0.,0.
    for X,y in loader:
        transferred_model.eval()
        attacked_model.eval()
        X,y = X.to(device), y.to(device)
        with ctx_noparamgrad_and_eval(transferred_model):
            delta = attack(**param).generate(transferred_model,X,y)
        with torch.no_grad():
            yp = attacked_model(X+delta)
            loss = nn.CrossEntropyLoss()(yp,y)
        
        total_correct += (yp.argmax(dim = 1) == y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss

def return_pgd100(loader, model, epsilon, device, norm='l2'):
    total_correct = 0.
    
    if norm =='l2':
        param = {"ord":2, "epsilon": epsilon, "alpha": epsilon*2.5/100, "num_iter": 100}
    elif norm =='linf':
        param = {"ord":np.inf, "epsilon": epsilon, "alpha": 0.01, "num_iter": 100}
    else:
        raise ValueError('norm not supported!')
    
#     if len(loader.dataset.classes) != 2:
#         param['loss_fn'] = nn.CrossEntropyLoss()
#     else:
#         param['loss_fn'] = nn.BCEWithLogitsLoss()

    delta_norm = 0

    delta_holder = []
    
    n_samples = 0

    with trange(len(loader)) as t:
        for X,y in loader:
            model.eval()
            X,y = X.to(device), y.to(device)

#             if len(loader.dataset.classes) != 2:
#                 y = y.long()

            delta = pgd(**param).generate(model, X, y)

            if norm =="l2":
                delta_norm += torch.norm(delta,p=param['ord'],dim=[1,2,3]).sum().item()
            elif norm =='linf':
                delta_norm +=delta.view(X.shape[0],-1).max(dim=1)[0].sum().item()

            # delta_holder.append(delta.detach().cpu())
            delta_holder.append(delta)

            t.update()
            
            n_samples += X.shape[0]
            if n_samples >= 1000:
                break
    delta_holder = torch.cat(delta_holder)
    return delta_holder

def test_pgd100(loader, model, epsilon, device, norm='l2', clip = True, num_test=None):
    total_correct = 0.
    total_correct_5 = 0
    
    if norm =='l2':
        param = {"ord":2, "epsilon": epsilon, "alpha": epsilon*2.5/100, "num_iter": 100, 'clip':clip}
    elif norm =='linf':
        param = {"ord":np.inf, "epsilon": epsilon, "alpha": 0.01, "num_iter": 100}
    else:
        raise ValueError('norm not supported!')

    delta_norm = 0
    tested = 0
    
    with trange(len(loader)) as t:
        for X,y in loader:
            model.eval()
            X,y = X.to(device), y.to(device)

            delta = pgd(**param).generate(model, X, y)

            if norm =="l2":
                delta_norm += torch.norm(delta,p=param['ord'],dim=[1,2,3]).sum().item()
            elif norm =='linf':
                delta_norm +=delta.view(X.shape[0],-1).max(dim=1)[0].sum().item()

            with torch.no_grad():
                y_hat = model(X+delta)
                # batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
                batch_acc = accuracy(y_hat, y, topk=(1,5))
                batch_correct = batch_acc[0].sum().item()*X.shape[0]/100
                batch_correct_5 = batch_acc[1].sum().item()*X.shape[0]/100

            total_correct += batch_correct
            total_correct_5 += batch_correct_5
            
            t.set_postfix(acc = '{0:.2f}%'.format(batch_correct/X.shape[0]*100))
            t.update()
            
            tested += X.shape[0]
            if num_test is not None and tested > num_test:
                break
    delta_norm /= tested
    test_acc = total_correct / tested * 100
    test_acc_5 = total_correct_5 / len(loader.dataset) * 100
    # return test_acc, delta_norm
    return test_acc, delta_norm, test_acc_5

def test_pgd100_experiments(loader, model, epsilon, device, norm='l2'):
    total_correct, total_loss = 0.,0.
    
    if norm =='l2':
        param = {"ord":2, "epsilon": epsilon, "alpha": epsilon*2.5/100, "num_iter": 100}
    elif norm =='linf':
        param = {"ord":np.inf, "epsilon": epsilon, "alpha": 0.01, "num_iter": 100}
    else:
        raise ValueError('norm not supported!')
    
    if len(loader.dataset.classes) != 2:
        # param['loss_fn'] = nn.CrossEntropyLoss()
        param['loss_fn'] = nn.MSELoss()
    else:
        param['loss_fn'] = nn.MSELoss()
        # param['loss_fn'] = nn.BCEWithLogitsLoss()

    delta_norm = 0
    delta_holder = []
    x_holder = []
    with trange(len(loader)) as t:
        for X,y in loader:
            model.eval()
            X,y = X.to(device), y.to(device)

            new_y = torch.rand(len(y)).to(device)
            new_y[y==0] *= -1

            # if len(loader.dataset.classes) != 2:
                # y = y.long()

            delta = pgd(**param).generate(model, X, y)

            if norm =="l2":
                delta_norm += torch.norm(delta,p=param['ord'],dim=[1,2,3]).sum().item()
            elif norm =='linf':
                delta_norm +=delta.view(X.shape[0],-1).max(dim=1)[0].sum().item()

            with torch.no_grad():
                y_hat = model(X+delta)
                loss = torch.nn.MSELoss(reduction='sum')(y_hat,new_y)
                if len(loader.dataset.classes) == 2:
                    new_y = new_y.float().view(-1,1)
                    # batch_correct = ((y_hat > 0) == (y==1)).sum().item()
                    batch_correct = ((y_hat > 0) == (new_y>0)).sum().item()
                else:
                    y = y.long()
                    batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()

            total_correct += batch_correct
            total_loss += loss
            
            t.set_postfix(acc = '{0:.2f}%'.format(batch_correct/X.shape[0]*100))
            t.update()

            delta_holder.append(delta)
            x_holder.append(X)

    delta_norm /= len(loader.dataset)
    test_acc = total_correct / len(loader.dataset) * 100
    loss_mean = total_loss / len(loader.dataset)

    delta_holder = torch.cat(delta_holder)
    x_holder = torch.cat(x_holder)
    # return test_acc, delta_norm, delta_holder, x_holder, loss_mean
    return test_acc,loss_mean.item()

def test_attack(loader, model, args, device, output_actual_norm = False):
    total_correct = 0.
    
    param = {'ord': args['ord'],
                 'epsilon': args['eps'],
                 'alpha': args['alpha'],
                 'num_iter': args['num_iter'],
                 'restarts': 1,
                 'loss_fn': args['loss_fn'],
                 'rand_init': args['rand_init'],
                 'clip': args['clip']}
    
    delta_norm = 0
    with trange(len(loader)) as t:
        for X,y in loader:
            model.eval()
            X,y = X.to(device), y.to(device)
            
            if len(loader.dataset.classes) != 2:
                y = y.long()

            delta = pgd(**param).generate(model, X, y)
            delta_norm += torch.norm(delta,p=param['ord'],dim=[1,2,3]).sum().item()

            with torch.no_grad():
                y_hat = model(X+delta)
                if len(loader.dataset.classes) == 2:
                    y = y.float().view(-1,1)
#                     loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                    batch_correct = ((y_hat > 0) == (y==1)).sum().item()
                else:
                    y = y.long()
#                     loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                    batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()

            total_correct += batch_correct
#             total_loss += loss.item() * X.shape[0]
            
            t.set_postfix(acc = '{0:.2f}%'.format(batch_correct/X.shape[0]*100))
            t.update()
    delta_norm /= len(loader.dataset)
        
    test_acc = total_correct / len(loader.dataset) * 100
    if output_actual_norm:
        return test_acc, delta_norm
    else:
        return test_acc

def test_AA(loader, model, norm, eps, attacks_to_run=None, verbose=False, visualization_only = False):

    assert norm in ['L2', 'Linf']

    adversary = AutoAttack(model, norm=norm, eps=eps, version='standard', verbose=verbose)
    if attacks_to_run is not None:
        adversary.attacks_to_run = attacks_to_run

    lx, ly = [],[]
    for x,y in loader:
        lx.append(x)
        ly.append(y)
    x_test = torch.cat(lx, 0)
    y_test = torch.cat(ly, 0)

    if not visualization_only:
        x_test,y_test = x_test[:1000],y_test[:1000]
        bs = 32 if x_test.shape[2] == 224 else 100

        with torch.no_grad():

            result = adversary.run_standard_evaluation_return_robust_accuracy(x_test, y_test, bs=bs, return_perturb=True)

            total_correct = 0
            total_correct_5 = 0
            for i in range(20):

                x_adv = result[1][i*50:(i+1)*50,:].to('cuda')
                y_adv = y_test[i*50:(i+1)*50].to('cuda')
                y_hat = model(x_adv)
                batch_acc = accuracy(y_hat, y_adv,topk=(1,5))
                batch_correct = batch_acc[0].sum().item()*50/100
                batch_correct_5 = batch_acc[1].sum().item()*50/100
                total_correct += batch_correct
                total_correct_5 += batch_correct_5
            test_acc = total_correct / 1000 * 100
            test_acc_5 = total_correct_5 / 1000 * 100
            
        return test_acc, test_acc_5
    else:
        x_test,y_test = x_test[:10],y_test[:10]
        bs = 10
        result = adversary.run_standard_evaluation_return_robust_accuracy(x_test, y_test, bs=bs, return_perturb=True)
        
        successful_perturb = torch.bitwise_not(torch.all(x_test.view(10,-1)==result[1].view(10,-1), dim=1))
        assert successful_perturb.sum() >= 2
        
        return result[1][successful_perturb]

    
CORRUPTIONS_MNIST=['identity',
 'shot_noise',
 'impulse_noise',
 'glass_blur',
 'motion_blur',
 'shear',
 'scale',
 'rotate',
 'brightness',
 'translate',
 'stripe',
 'fog',
 'spatter',
 'dotted_line',
 'zigzag',
 'canny_edges'] 


CORRUPTIONS_CIFAR10=['brightness',         
'gaussian_noise',    
'saturate',
'contrast',           
'glass_blur',        
'shot_noise',
'defocus_blur',       
'impulse_noise',     
'snow',
'elastic_transform',  
'jpeg_compression',  
'spatter',
'fog',         
'speckle_noise',
'frost',              
'motion_blur',       
'zoom_blur',
'gaussian_blur',      
'pixelate'] 

def eval_corrupt(model, dataset, freq, severity, device):
    dct_matrix = getDCTmatrix(28)
    acc = []
    acc_5 = []
    total_correct = 0
    total_correct_5 = 0
    
    if dataset =='mnist':
        corruptions_list = CORRUPTIONS_MNIST
    elif dataset in ['cifar10', 'cifar100']:
        corruptions_list = CORRUPTIONS_CIFAR10
    
    for corrupt_type in corruptions_list:
        if dataset =='mnist': 
            data_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/mnist_c/' + corrupt_type + '/test_images.npy'
            label_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/mnist_c/' + corrupt_type + '/test_labels.npy'
            
            x = torch.tensor(np.load(data_path)/255.,dtype= torch.float32).view(10000,1,28,28).to(device)
            y = torch.tensor(np.load(label_path),dtype=torch.float32).view(10000).to(device)
            with torch.no_grad():
                if freq:
                    x = batch_dct2(x, dct_matrix).unsqueeze(1)

                y_hat = model(x)
                y = y.long()
                # total_correct = (y_hat.argmax(dim = 1) == y).sum().item()
                batch_acc = accuracy(y_hat, y, topk=(1,5))
                total_correct = batch_acc[0].sum().item()*x.shape[0]/100
                total_correct_5 = batch_acc[1].sum().item()*x.shape[0]/100
        
        elif dataset == 'cifar10':
            data_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/CIFAR-10-C/' + corrupt_type + '.npy'
            label_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/CIFAR-10-C/labels.npy'
            x = torch.tensor(np.transpose(np.load(data_path), (0, 3, 1, 2))/255.,dtype= torch.float32)[(severity-1)*10000:severity*10000].to(device)
            y = torch.tensor(np.load(label_path),dtype=torch.float32)[(severity-1)*10000:severity*10000].to(device)
            
            # mean = [0.49139968, 0.48215841, 0.44653091]
            # std = [0.24703223, 0.24348513, 0.26158784]
            # x[:,0,:,:] = (x[:,0,:,:]-mean[0])/std[0]
            # x[:,1,:,:] = (x[:,1,:,:]-mean[1])/std[1]
            # x[:,2,:,:] = (x[:,2,:,:]-mean[2])/std[2]

            for i in range(10):
                with torch.no_grad():
                    y_hat = model(x[1000*i:1000*(i+1)])
                    # total_correct += (y_hat.argmax(dim = 1) == y[1000*i:1000*(i+1)]).sum().item()
                    batch_acc = accuracy(y_hat, y[1000*i:1000*(i+1)], topk=(1,5))
                    total_correct += batch_acc[0].sum().item()*1000/100
                    total_correct_5 += batch_acc[1].sum().item()*1000/100

        elif dataset == 'cifar100':
            data_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/CIFAR-100-C/' + corrupt_type + '.npy'
            label_path = '/h/ama/workspace/ama-at-vector/freq-robust/data/CIFAR-100-C/labels.npy'
            x = torch.tensor(np.transpose(np.load(data_path), (0, 3, 1, 2))/255.,dtype= torch.float32)[(severity-1)*10000:severity*10000].to(device)
            y = torch.tensor(np.load(label_path),dtype=torch.float32)[(severity-1)*10000:severity*10000].to(device)
            
            # mean = [0.50707516, 0.48654887, 0.44091784]
            # std = [0.26733429, 0.25643846, 0.27615047]
            # x[:,0,:,:] = (x[:,0,:,:]-mean[0])/std[0]
            # x[:,1,:,:] = (x[:,1,:,:]-mean[1])/std[1]
            # x[:,2,:,:] = (x[:,2,:,:]-mean[2])/std[2]

            for i in range(10):
                with torch.no_grad():
                    y_hat = model(x[1000*i:1000*(i+1)])
                    # total_correct += (y_hat.argmax(dim = 1) == y[1000*i:1000*(i+1)]).sum().item()
                    batch_acc = accuracy(y_hat, y[1000*i:1000*(i+1)], topk=(1,5))
                    total_correct += batch_acc[0].sum().item()*1000/100
                    total_correct_5 += batch_acc[1].sum().item()*1000/100

        corrupt_acc = total_correct / len(y) * 100
        corrupt_acc_5 = total_correct_5 / len(y) * 100
        acc.append(corrupt_acc)
        acc_5.append(corrupt_acc_5)
        total_correct = 0
        total_correct_5 = 0
        
        del x,y 
    
    return acc, acc_5

def eval_CE(base_acc, f_acc):
    
    mCE = []
    rel_mCE = []
    for i in range(1,16):
        mCE.append((100-f_acc[i])/(100-base_acc[i]))
        rel_mCE.append(((100-f_acc[i])-(100-f_acc[0]))/((100-base_acc[i])-(100-base_acc[0])))

    return np.array(mCE).mean(), np.array(rel_mCE).mean()

def computeSensitivityMap(loader, model, eps, dim, numb_inputs, clip, device):
    sens_map = torch.zeros(dim,dim,device=device)

    with trange(dim*dim) as t:
        for i in range(dim):
            for j in range(dim):
                sens_map[i,j] = test_freq_sensitivity(loader, model, eps, i,j, dim, numb_inputs, clip, device)

                t.set_postfix(err = '{0:.2f}%'.format(sens_map[i,j]))
                t.update()
    return sens_map

def test_freq_sensitivity(loader, model, eps, x, y, size, numb_inputs, clip, device):
    total_correct = 0.
    total_correct_bayes = 0.
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
            y_hat = model(X)
            if clip:
                y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
                y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
            else:
                y_hat_pos = model((X+delta_pos))
                y_hat_neg = model((X+delta_neg))
                
        y = y.long()
#         batch_correct_pos = (y_hat_pos.argmax(dim = 1) == y)
#         batch_correct_neg = (y_hat_neg.argmax(dim = 1) == y)
        batch_correct_pos = (y_hat_pos.argmax(dim = 1) == y)
        batch_correct_neg = (y_hat_neg.argmax(dim = 1) == y)
        
        batch_correct = (batch_correct_neg*batch_correct_pos).sum().item()
        
        total_correct += batch_correct
        
        total_tested_input += X.shape[0]
        if total_tested_input>numb_inputs:
            break
        
    test_acc = (total_correct) / total_tested_input * 100

    return test_acc
