import os
import sys
import logging

import torch
import numpy as np

from src.attacks import pgd
from src.train import train_standard, train_adv, train_hp_filtered, train_amp_filtered
from src.evaluation import test_clean, test_pgd100, test_AA, eval_corrupt, eval_CE, test_gaussian, CORRUPTIONS_CIFAR10, CORRUPTIONS_MNIST
from src.args import get_args
from src.utils_dataset import load_dataset
from src.utils_log import metaLogger, rotateCheckpoint, wandbLogger, delCheckpoint
from src.utils_freq import mask_radial
from src.utils_plot import visualize_two_layer, visualize_attack
from src.utils_general import seed_everything, get_model, get_optim
import ipdb

def train(args, epoch, logger, loader, model, opt, device):

    """perform one epoch of training."""
    if args.method == "standard":
        train_log = train_standard(logger, epoch, loader, model, opt, device)

    elif args.method == "remove_high_freq":

        if args.dataset in ['mnist', 'fashionmnist']:
            _dim = 28
        elif args.dataset in ['cifar10', 'cifar100', 'svhn']:
            _dim = 32
        elif args.dataset in ['caltech', 'imagenette']:
            _dim = 224

        spacing = _dim*np.sqrt(2)/2000
        candidate_radius = np.arange(_dim*np.sqrt(2), 0, -spacing)
        for _r in candidate_radius:
            freq_mask = mask_radial(_dim, _r)
            included_freq_ratio = freq_mask.sum()/_dim/_dim*100
            if included_freq_ratio < args.threshold:
                break
        freq_mask = torch.tensor(freq_mask, dtype = torch.float32, device=device)
        train_log = train_hp_filtered(logger, epoch, loader, model, opt, freq_mask, device)

    elif args.method == "remove_low_amp":
        train_log = train_amp_filtered(logger, epoch, loader, model, opt, args.threshold, args.dataset, device)

    elif args.method[:3] == "adv":
        if 'l2' in args.method:
            norm = 'l2'
        elif 'linf' in args.method:
            norm = 'linf'
        else:
            print('norm not specified, will use default l2 adversarial training')
            norm = 'l2'
        train_log = train_adv(logger, epoch, loader, args.pgd_eps, args.pgd_steps, model, opt, device, norm)

    else:
        raise  NotImplementedError("Training method not implemented!")

    w_l2_norm = 0
    for param in model.parameters():
        w_l2_norm += ((param)**2).sum()
    w_l2_norm = torch.sqrt(w_l2_norm).item()

    logger.add_scalar("train/acc_ep", train_log[0], epoch+1)
    logger.add_scalar("train/loss_ep", train_log[1], epoch+1)
    logger.add_scalar("weight norm", w_l2_norm, epoch+1)
    logging.info(
        "Epoch: [{0}]\t"
        "Loss: {loss:.6f}\t"
        "Accuracy: {acc:.2f}\t"
        "Weight Norm: {norm:.2f}".format(
            epoch+1,
            loss=train_log[1],
            acc=train_log[0],
            norm=w_l2_norm))

    return train_log

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args()
    logger = metaLogger(args)
    logging.basicConfig(
        filename=args.j_dir+ "/log/log.txt",
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    seed_everything(args.seed)
    train_loader, test_loader = load_dataset(args.dataset, args.batch_size, args.standard_DA, args.freq)

    if args.dataset == 'mnist':
        var_list = [0.01, 0.05, 0.1]
        l2_eps_list = [0.5, 0.7, 1.0]
        linf_eps_list = [0.05, 0.07, 0.1]
    elif args.dataset == 'fashionmnist':
        var_list = [0.001, 0.005, 0.01]
        l2_eps_list = [0.1, 0.5, 0.7]
        linf_eps_list = [0.01, 0.03, 0.05]
    elif args.dataset == 'cifar10':
        var_list = [0.001, 0.005, 0.007]
        l2_eps_list = [0.1, 0.2, 0.3]
        linf_eps_list = [1./255., 2./255., 4./255.]
    elif args.dataset == 'cifar100':
        var_list = [0.001, 0.005, 0.007]
        l2_eps_list = [0.1, 0.2, 0.3]
        linf_eps_list = [1./255., 2./255., 4./255.]
    elif args.dataset == 'svhn':
        var_list = [0.001, 0.003, 0.005]
        l2_eps_list = [0.1, 0.2, 0.3]
        linf_eps_list = [1./255., 2./255., 4./255.]
    elif args.dataset == 'caltech':
        var_list = [0.01, 0.05, 0.1]
        l2_eps_list = [0.5, 1.0, 1.5]
        linf_eps_list = [1./255., 2./255., 4./255.]
    elif args.dataset == 'imagenette':
        var_list = [0.01, 0.05, 0.1]
        l2_eps_list = [0.5, 1.0, 1.5]
        linf_eps_list = [1./255., 2./255., 4./255.]

    eval_threshold = 50

    model = get_model(args)
    print('model initialized')
    model.to(device)
    opt, lr_scheduler = get_optim(model, args)
    ckpt_epoch = 0

    ckpt_dir = args.j_dir+"/"+str(args.j_id)+"/"
    ckpt_location = os.path.join(ckpt_dir, "custom_ckpt_"+logger.ckpt_status+".pth")
    if os.path.exists(ckpt_location):
        ckpt = torch.load(ckpt_location)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        ckpt_epoch = ckpt["epoch"]
        if lr_scheduler:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        print("LOADED CHECKPOINT")

    actual_trained_epoch = args.epoch

    for _epoch in range(ckpt_epoch, args.epoch):
        train_log = train(args, _epoch, logger, train_loader, model, opt, device)
        if lr_scheduler:
            lr_scheduler.step()

        # evaluation on testset
        test_log = test_clean(test_loader, model, device)
        logger.add_scalar("test/top1_acc", test_log[0], _epoch+1)
        logger.add_scalar("test/top5_acc", test_log[2], _epoch+1)
        logger.add_scalar("test/loss", test_log[1], _epoch+1)
        logging.info(
            "Test set: Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                loss=test_log[1],
                acc=test_log[0]))

        # at the end of training, save after test set evaluation, since Autoattack sometimes fail
        if (_epoch+1) == args.epoch:
            torch.save(model.state_dict(), args.j_dir+"/model/final_model.pt")

        if (_epoch+1) % args.ckpt_freq == 0:
            rotateCheckpoint(ckpt_dir, "custom_ckpt", model, opt, _epoch, lr_scheduler)
            logger.save_log()

        # half way thru training and test accuracy still below 20%
        if np.isnan(train_log[1]) or ((_epoch+1) > int(args.epoch/2) and test_log[0] < 20): 
            actual_trained_epoch = _epoch+1
            torch.save(model.state_dict(), args.j_dir+"/model/final_model.pt")
            break # break the training for-loop

        # evaluation on my implementation of the PGD100 attack
        # ONLY if the test accuracy is above 60%
        # if ((_epoch+1) % 20 == 0 and args.eval_PGD) or (_epoch+1) == args.epoch:
        # if ((_epoch+1) % eval_freq == 0) or (_epoch+1) == args.epoch:
            # if args.dataset in ['mnist', 'fashionmnist']:
                # if (_epoch+1) == args.epoch:
                    # num_test = None
                # else:
                    # num_test = 1000

                # for _eps in l2_eps_list:
                    # pgd_log = test_pgd100(test_loader, model, _eps, device, norm='l2', clip = args.pgd_clip, num_test = num_test)
                    # logger.add_scalar("pgd100(L2)_"+str(_eps)+"/top1_acc", pgd_log[0], _epoch+1)
                    # logger.add_scalar("pgd100(L2)_"+str(_eps)+"/top5_acc", pgd_log[2], _epoch+1)
                    # logger.add_scalar("pgd100(L2)_"+str(_eps)+"/delta_2norm", pgd_log[1], _epoch+1)

                # for _eps in linf_eps_list:
                    # pgd_log = test_pgd100(test_loader, model, _eps, device, norm='linf', clip = args.pgd_clip, num_test = num_test)
                    # logger.add_scalar("pgd100(Linf)_"+str(_eps)+"/top1_acc", pgd_log[0], _epoch+1)
                    # logger.add_scalar("pgd100(Linf)_"+str(_eps)+"/top5_acc", pgd_log[2], _epoch+1)
                    # logger.add_scalar("pgd100(Linf)_"+str(_eps)+"/delta_infnorm", pgd_log[1], _epoch+1)

        # # if ((_epoch+1) % 20 == 0) or (_epoch+1) == args.epoch:
            # for _var in var_list:
                # gau_acc = test_gaussian(test_loader, model, _var, device)
                # logger.add_scalar("gau_"+str(_var)+"/top1_acc", gau_acc[0], _epoch+1)
                # logger.add_scalar("gau_"+str(_var)+"/top5_acc", gau_acc[2], _epoch+1)

        # if ((_epoch+1)%10 == 0 and args.eval_AA) or (_epoch+1) == args.epoch:
        # if (_epoch+1) == args.epoch and test_log[0] > eval_threshold:
            # for _eps in l2_eps_list:
                # AA_acc = test_AA(test_loader, model, norm="L2", eps=_eps)
                # logger.add_scalar("AA(L2)_"+str(_eps)+"/top1_acc", AA_acc[0], _epoch+1)
                # logger.add_scalar("AA(L2)_"+str(_eps)+"/top5_acc", AA_acc[1], _epoch+1)

            # for _eps in linf_eps_list:
                # AA_acc = test_AA(test_loader, model, norm="Linf", eps=_eps)
                # logger.add_scalar("AA(Linf)_"+str(_eps)+"/top1_acc", AA_acc[0], _epoch+1)
                # logger.add_scalar("AA(Linf)_"+str(_eps)+"/top5_acc", AA_acc[1], _epoch+1)

            # for _var in var_list:
                # gau_acc = test_gaussian(test_loader, model, _var, device)
                # logger.add_scalar("gau_"+str(_var)+"/top1_acc", gau_acc[0], _epoch+1)
                # logger.add_scalar("gau_"+str(_var)+"/top5_acc", gau_acc[2], _epoch+1)


        # if (_epoch+1) == args.epoch and test_log[0] > eval_threshold:
            # if args.dataset == 'mnist':
                # corrupt_acc = eval_corrupt(model, args.dataset, args.freq, None, device)
                # for corruption, _corrupt_acc in zip(CORRUPTIONS_MNIST, corrupt_acc):
                    # logger.add_scalar(corruption, _corrupt_acc[0], _epoch+1)
                # logger.add_scalar('mCC top1_acc', np.array(corrupt_acc[0]).mean(), _epoch+1)
                # logger.add_scalar('mCC top5_acc', np.array(corrupt_acc[1]).mean(), _epoch+1)

            # if args.dataset in ['cifar10', 'cifar100']:
                # for _severity in [1, 3, 5]:
                    # corrupt_acc = eval_corrupt(model, args.dataset, False, _severity, device)
                    # for corruption, _corrupt_acc in zip(CORRUPTIONS_CIFAR10, corrupt_acc):
                        # logger.add_scalar(corruption+'-'+str(_severity), _corrupt_acc[0], _epoch+1)
                    # logger.add_scalar('mCC-'+str(_severity)+' top1_acc', np.array(corrupt_acc[0]).mean(), _epoch+1)
                    # logger.add_scalar('mCC-'+str(_severity)+' top5_acc', np.array(corrupt_acc[1]).mean(), _epoch+1)

            # if args.eval_CC:
                # if args.method != 'standard':
                    # base_model = get_model(args, device)
                    # base_model.load_state_dict(torch.load(args.base_model, map_location=device))
                    # model.to(device)
                    # print("\n ***  base model loaded: "+ args.base_model + " *** \n")
                    # base_corrupt_acc = eval_corrupt(base_model, args.dataset, device)

                    # mCE, rel_mCE = eval_CE(base_corrupt_acc, corrupt_acc)
                    # logger.add_scalar('mCE', mCE, _epoch+1)
                    # logger.add_scalar('rel mCE', rel_mCE, _epoch+1)

            # visualize model weight
            # if args.arch == 'two_layer':
                # fig = visualize_two_layer(model, init_weight)
                # logger.add_figure("weight", fig)


        if (_epoch+1) == args.epoch and test_log[0] > eval_threshold and args.eval_TMLR:
            for _eps in l2_eps_list:
                AA_acc = test_AA(test_loader, model, norm="L2", eps=_eps)
                logger.add_scalar("AA(L2)_"+str(_eps)+"/top1_acc", AA_acc[0], _epoch+1)
                logger.add_scalar("AA(L2)_"+str(_eps)+"/top5_acc", AA_acc[1], _epoch+1)

            for _eps in linf_eps_list:
                AA_acc = test_AA(test_loader, model, norm="Linf", eps=_eps)
                logger.add_scalar("AA(Linf)_"+str(_eps)+"/top1_acc", AA_acc[0], _epoch+1)
                logger.add_scalar("AA(Linf)_"+str(_eps)+"/top5_acc", AA_acc[1], _epoch+1)

            for _var in var_list:
                gau_acc = test_gaussian(test_loader, model, _var, device)
                logger.add_scalar("gau_"+str(_var)+"/top1_acc", gau_acc[0], _epoch+1)
                logger.add_scalar("gau_"+str(_var)+"/top5_acc", gau_acc[2], _epoch+1)


    # upload runs to wandb:
    if args.enable_wandb:
        save_wandb_retry = 0
        save_wandb_successful = False
        while not save_wandb_successful and save_wandb_retry < 5:
            print('Uploading runs to wandb...')
            try:
                wandb_logger = wandbLogger(args)
                wandb_logger.upload(logger, actual_trained_epoch)
            except:
                save_wandb_retry += 1
                print('Retry {} times'.format(save_wandb_retry))
            else:
                save_wandb_successful = True

        if not save_wandb_successful:
            print('Failed at uploading runs to wandb.')

    logger.save_log(is_final_result=True)

    # delete slurm checkpoints
    delCheckpoint(args.j_dir, args.j_id)

if __name__ == "__main__":
    main()
