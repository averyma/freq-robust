import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import grad

from src.utils_general import seed_everything
from src.context import ctx_noparamgrad_and_eval

from models import PreActResNet18,c2

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from collections import defaultdict
import ipdb
import copy
import warnings
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
import matplotlib.pyplot as plt

from src.utils_dataset import load_dataset
from src.evaluation import test_AA

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--norm', default='L2')
parser.add_argument('--eps', default='0.1',type=float)

args = parser.parse_args()

dataset = args.dataset
_eps = args.eps
norm = args.norm

f = open("/scratch/ssd001/home/ama/workspace/ama-at-vector/freq-robust/notebook/TMLR/aa_eval/{}_{}_{}.txt".format(dataset,norm,_eps), "a")
f.write("{}\t{}\t{}\n".format(dataset,norm,_eps))

seed_everything(43)
train_loader, test_loader = load_dataset(dataset, 128, True, freq=False, test_shuffle = True)

if dataset == 'mnist':
    model_sgd_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-mnist-c2-sgd-0.1-4036/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-mnist-c2-sgd-0.1-8841/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-mnist-c2-sgd-0.1-12944/model/final_model.pt']
    model_adam_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-mnist-c2-adam-0.0005-20658/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-mnist-c2-adam-0.0005-14085/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-mnist-c2-adam-0.0005-6600/model/final_model.pt']
    model_rmsp_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-mnist-c2-rmsprop-0.0005-13138/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-mnist-c2-rmsprop-0.0005-22508/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-mnist-c2-rmsprop-0.0005-10665/model/final_model.pt']
    var = 0.1
    r_list = [9.23556484576784, 13.429772309525275, 16.573100530176493, 19.246660078812468, 21.650519659173664, 23.770439869682388, 25.719176010665375, 27.52522654685313, 30.425587675884607]
    model_sgd = c2()
    model_adam = c2()
    model_rmsp = c2()

elif dataset == 'fashionmnist':
    model_sgd_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-fashionmnist-c2-sgd-0.01-16733/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-fashionmnist-c2-sgd-0.01-5160/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-fashionmnist-c2-sgd-0.01-12377/model/final_model.pt']
    model_adam_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-fashionmnist-c2-adam-0.0005-20237/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-fashionmnist-c2-adam-0.0005-7021/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-07/20221207-fashionmnist-c2-adam-0.0005-2781/model/final_model.pt']
    model_rmsp_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-08/20221208-fashionmnist-c2-rmsprop-0.00005-2646/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-08/20221208-fashionmnist-c2-rmsprop-0.00005-13068/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-08/20221208-fashionmnist-c2-rmsprop-0.00005-11649/model/final_model.pt']
    var = 0.1
    r_list = [9.23556484576784, 13.429772309525275, 16.573100530176493, 19.246660078812468, 21.650519659173664, 23.770439869682388, 25.719176010665375, 27.52522654685313, 30.425587675884607]
    model_sgd = c2()
    model_adam = c2()
    model_rmsp = c2()
    
elif dataset == 'cifar10':
    num_classes = 10
    model_sgd_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-cifar10-sgd-0.2-26092/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-cifar10-sgd-0.2-26737/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-cifar10-sgd-0.2-26737/model/final_model.pt']
    model_adam_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-cifar10-adam-0.0002-32106/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-cifar10-adam-0.0002-30320/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-cifar10-adam-0.0002-13649/model/final_model.pt']
    
    model_rmsp_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-cifar10-rmsprop-0.0005-26922/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-cifar10-rmsprop-0.0005-4481/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-cifar10-rmsprop-0.0005-3710/model/final_model.pt']
    
    var = 0.01
    r_list = [10.781431922956449, 15.301968130724937, 19.04982421996594, 22.040185695902814, 24.75901837969084, 27.220978213474407, 29.441362859363448, 31.57831841836015, 34.71337554275011]
    model_sgd = PreActResNet18(dataset, num_classes, False, False)
    model_adam = PreActResNet18(dataset, num_classes, False, False)
    model_rmsp = PreActResNet18(dataset, num_classes, False, False)
    
elif dataset == 'cifar100':
    num_classes = 100
    model_sgd_path = ['/scratch/hdd001/home/ama/freq-robust/2022-08-10/20220810-cifar100-sgd-0.3-19570/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-08-10/20220810-cifar100-sgd-0.3-32033/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-08-10/20220810-cifar100-sgd-0.3-18853/model/final_model.pt']
    model_adam_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-cifar100-preactresnet50-adam-0.0005-8754/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-cifar100-preactresnet50-adam-0.0005-5056/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-cifar100-preactresnet50-adam-0.0005-741/model/final_model.pt']
    model_rmsp_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-cifar100-preactresnet50-rmsprop-0.0005-17210/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-cifar100-preactresnet50-rmsprop-0.0005-12464/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-cifar100-preactresnet50-rmsprop-0.0005-12783/model/final_model.pt']

    var = 0.01
    r_list = [10.781431922956449, 15.301968130724937, 19.04982421996594, 22.040185695902814, 24.75901837969084, 27.220978213474407, 29.441362859363448, 31.57831841836015, 34.71337554275011]
    model_sgd = PreActResNet18(dataset, num_classes, False, False)
    model_adam = PreActResNet18(dataset, num_classes, False, False)
    model_rmsp = PreActResNet18(dataset, num_classes, False, False)
    
elif dataset == 'svhn':
    num_classes = 10
    model_sgd_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-svhn-sgd-0.2-5074/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-svhn-sgd-0.2-2812/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-svhn-sgd-0.2-24803/model/final_model.pt']
    model_adam_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-svhn-adam-0.0002-18636/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-svhn-adam-0.0002-14539/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-svhn-adam-0.0002-27724/model/final_model.pt']     
    model_rmsp_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-svhn-rmsprop-0.0002-5933/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-svhn-rmsprop-0.0002-2389/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-svhn-rmsprop-0.0002-29169/model/final_model.pt']
    var = 0.1
    r_list = [10.781431922956449, 15.301968130724937, 19.04982421996594, 22.040185695902814, 24.75901837969084, 27.220978213474407, 29.441362859363448, 31.57831841836015, 34.71337554275011]
    model_sgd = PreActResNet18(dataset, num_classes, False, False)
    model_adam = PreActResNet18(dataset, num_classes, False, False)
    model_rmsp = PreActResNet18(dataset, num_classes, False, False)

elif dataset == 'caltech':
    num_classes = 101
    model_sgd_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-caltech-preactresnet50-sgd-0.05-5936/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-caltech-preactresnet50-sgd-0.05-27380/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-caltech-preactresnet50-sgd-0.05-13158/model/final_model.pt']
    model_adam_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-caltech-preactresnet50-adam-0.0002-13765/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-caltech-preactresnet50-adam-0.0002-22154/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-caltech-preactresnet50-adam-0.0002-28879/model/final_model.pt']
    model_rmsp_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-caltech-preactresnet50-rmsprop-0.001-21244/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-caltech-preactresnet50-rmsprop-0.001-29384/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-14/20221214-caltech-preactresnet50-rmsprop-0.001-21692/model/final_model.pt']
    
    var = 0.2
    r_list = [79.99591867969022, 113.48107898359288, 138.12383764880263, 159.77959526368392, 178.8104125616099, 195.53446412645394, 211.45609907581306, 226.35093083884985, 250.1009872575449]
    model_sgd = PreActResNet18(dataset, num_classes, False, False)
    model_adam = PreActResNet18(dataset, num_classes, False, False)
    model_rmsp = PreActResNet18(dataset, num_classes, False, False)

elif dataset == 'imagenette':
    num_classes = 10
    model_sgd_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-imagenette-sgd-0.1-16528/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-imagenette-sgd-0.1-10875/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-imagenette-sgd-0.1-10824/model/final_model.pt']
    model_adam_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-imagenette-adam-0.0002-32508/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-imagenette-adam-0.0002-329/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-imagenette-adam-0.0002-3167/model/final_model.pt']
    model_rmsp_path = ['/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-imagenette-rmsprop-0.0002-9496/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-imagenette-rmsprop-0.0002-31471/model/final_model.pt',
                     '/scratch/hdd001/home/ama/freq-robust/2022-12-13/20221213-imagenette-rmsprop-0.0002-30967/model/final_model.pt']

    var = 0.01
    r_list = [79.99591867969022, 113.48107898359288, 138.12383764880263, 159.77959526368392, 178.8104125616099, 195.53446412645394, 211.45609907581306, 226.35093083884985, 250.1009872575449]
    model_sgd = PreActResNet18(dataset, num_classes, False, False)
    model_adam = PreActResNet18(dataset, num_classes, False, False)
    model_rmsp = PreActResNet18(dataset, num_classes, False, False)

# #sgd
model_sgd.load_state_dict(torch.load(model_sgd_path[0]))
#adam
model_adam.load_state_dict(torch.load(model_adam_path[0]))
#rmsp
model_rmsp.load_state_dict(torch.load(model_rmsp_path[0]))

model_sgd.to(_device)
model_adam.to(_device)
model_rmsp.to(_device)
print('model loaded')

aa_acc_sgd = []
aa_acc_adam = []
aa_acc_rmsp = []
aa_acc_sgd_top5 = []
aa_acc_adam_top5 = []
aa_acc_rmsp_top5 = []

for _m in range(3):
    model_sgd.load_state_dict(torch.load(model_sgd_path[_m]))
    aa_acc = test_AA(test_loader, model_sgd, norm, eps=_eps, visualization_only = False)
    aa_acc_sgd.append(aa_acc[0])
    aa_acc_sgd_top5.append(aa_acc[1])
    
    model_adam.load_state_dict(torch.load(model_adam_path[_m]))
    aa_acc = test_AA(test_loader, model_adam, norm, eps=_eps, visualization_only = False)
    aa_acc_adam.append(aa_acc[0])
    aa_acc_adam_top5.append(aa_acc[1])
    
    model_rmsp.load_state_dict(torch.load(model_rmsp_path[_m]))
    aa_acc = test_AA(test_loader, model_rmsp, norm, eps=_eps, visualization_only = False)
    aa_acc_rmsp.append(aa_acc[0])
    aa_acc_rmsp_top5.append(aa_acc[1])
    

f.write("**************** accuracy (top1) ****************\n")
f.write('SGD\t{:.2f}\n'.format(np.array(aa_acc_sgd).mean()))
f.write('ADAM\t{:.2f}\n'.format(np.array(aa_acc_adam).mean()))
f.write('RmsProp\t{:.2f}\n'.format(np.array(aa_acc_rmsp).mean()))

f.write("**************** accuracy (top5) ****************\n")
f.write('SGD\t{:.2f}\n'.format(np.array(aa_acc_sgd_top5).mean()))
f.write('ADAM\t{:.2f}\n'.format(np.array(aa_acc_adam_top5).mean()))
f.write('RmsProp\t{:.2f}\n'.format(np.array(aa_acc_rmsp_top5).mean()))

# f.write("Now the file has more content!")
f.close()


train_loader, test_loader = load_dataset(dataset, 128, True, freq=False, test_shuffle = False)
sgd_AA = test_AA(test_loader, model_sgd, norm, eps=_eps, visualization_only = True)
adam_AA = test_AA(test_loader, model_adam, norm, eps=_eps, visualization_only = True)
rmsp_AA = test_AA(test_loader, model_rmsp, norm, eps=_eps, visualization_only = True)

if dataset == "mnist":
    print_dataset = 'MNIST'
elif dataset == "fashionmnist":
    print_dataset = 'FashionMNIST'
elif dataset == "cifar10":
    print_dataset = 'CIFAR10'
elif dataset == "cifar100":
    print_dataset = 'CIFAR100'
elif dataset == "svhn":
    print_dataset = 'SVHN'
elif dataset == "caltech":
    print_dataset = 'Caltech101'
elif dataset == "imagenette":
    print_dataset = 'Imagenette'
elif dataset == "tiny":
    print_dataset = 'TinyImageNet'
else:
    raise NotImplemented('Incorrect dataset specified!')


fix, axs = plt.subplots(nrows = 2, ncols=3, figsize=(7, 4))

for i in range(3):
    if i == 0:
        img = sgd_AA
    elif i == 1:
        img = adam_AA
    elif i == 2:
        img = rmsp_AA
        
    if dataset in ['mnist', 'fashionmnist']:
        im = axs[0,i].imshow(img[0].squeeze(),cmap='gray')
        im = axs[1,i].imshow(img[1].squeeze(),cmap='gray')
    else:
        im = axs[0,i].imshow(img[0].squeeze().permute(1,2,0).cpu().numpy())
        im = axs[1,i].imshow(img[1].squeeze().permute(1,2,0).cpu().numpy())
    axs[0,i].set_xticks([], [])
    axs[0,i].set_yticks([], [])
    axs[1,i].set_xticks([], [])
    axs[1,i].set_yticks([], [])
    
axs[0,0].set_title('Perturbed inputs based \non SGD-trained model', fontsize=12)
axs[0,1].set_title('Perturbed inputs based \non Adam-trained model', fontsize=12)
axs[0,2].set_title('Perturbed inputs based \non RMSProp-trained model', fontsize=12)
plt.tight_layout()

plt.savefig('/scratch/ssd001/home/ama/workspace/ama-at-vector/freq-robust/notebook/TMLR/figures/{}_AA_{}_{}.pdf'.format(print_dataset,norm,_eps), bbox_inches='tight')  