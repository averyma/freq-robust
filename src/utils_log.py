import time
import os
from collections import defaultdict
import torch
from src.save_function import checkpoint_save
# from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from collections import OrderedDict
import wandb
import ipdb
import numpy as np

# def rotateCheckpoint(ckpt_dir, ckpt_name, model, opt, epoch, lr_scheduler, init_weight):
def rotateCheckpoint(ckpt_dir, ckpt_name, model, opt, epoch, lr_scheduler):
    ckpt_curr = os.path.join(ckpt_dir, ckpt_name+"_curr.pth")
    ckpt_prev = os.path.join(ckpt_dir, ckpt_name+"_prev.pth")

    # no existing ckpt
    if not (os.path.exists(ckpt_curr) or os.path.exists(ckpt_prev)):
        saveCheckpoint(ckpt_dir,
                       ckpt_name+"_curr.pth",
                       model,
                       opt,
                       epoch,
                       lr_scheduler)
                       # lr_scheduler,
                       # init_weight)

    elif os.path.exists(ckpt_curr):
        # overwrite ckpt_prev with ckpt_curr
        cmd = "cp -r {} {}".format(ckpt_curr, ckpt_prev)
        os.system(cmd)
        saveCheckpoint(ckpt_dir,
                       ckpt_name+"_curr.pth",
                       model,
                       opt,
                       epoch,
                       lr_scheduler)
                       # lr_scheduler,
                       # init_weight)

# def saveCheckpoint(ckpt_dir, ckpt_name, model, opt, epoch, lr_scheduler, init_weight):
def saveCheckpoint(ckpt_dir, ckpt_name, model, opt, epoch, lr_scheduler):
    if lr_scheduler:
        checkpoint_save({"state_dict": model.state_dict(),
                         "optimizer": opt.state_dict(),
                         "epoch": epoch+1,
                         "lr_scheduler":lr_scheduler.state_dict()},
                        ckpt_dir, ckpt_name)
    else:
        checkpoint_save({"state_dict": model.state_dict(),
                         "optimizer": opt.state_dict(),
                         "epoch": epoch+1},
                         ckpt_dir, ckpt_name)

    print("SAVED CHECKPOINT")

class metaLogger(object):
    def __init__(self, args, flush_sec=5):
        self.log_path = args.j_dir+"/log/"
        self.ckpt_status = self.get_ckpt_status(args.j_dir, args.j_id)
        self.log_dict = self.load_log(self.log_path)
        # self.enable_wandb = args.enable_wandb

        # if self.enable_wandb:
            # self.wandb_log = wandb.init(name=args.j_dir.split("/")[-1],
                                        # project=args.wandb_project,
                                        # dir=args.j_dir,
                                        # id=str(args.j_id),
                                        # resume=True)
            # self.wandb_log.config.update(args)

    def get_ckpt_status(self, j_dir, j_id):
        
        status = "curr"
        ckpt_dir = j_dir+"/"+str(j_id)+"/"
        ckpt_location_prev = os.path.join(ckpt_dir, "ckpt_prev.pth")
        ckpt_location_curr = os.path.join(ckpt_dir, "ckpt_curr.pth")
        if os.path.exists(ckpt_location_curr) and os.path.exists(ckpt_location_prev):
            try:
                torch.load(ckpt_location_curr)
            except TypeError:
                status = "prev"
        else:
            status = "curr"
        return status
    
    def load_log(self, log_path):
        if self.ckpt_status == "curr":
            try:
                log_dict = torch.load(log_path + "/log_curr.pth")
            except FileNotFoundError:
                log_dict = defaultdict(lambda: list())
            except TypeError:
                log_dict = torch.load(log_path + "/log_prev.pth")
                self.ckpt_status = "prev"
        else:
            log_dict = torch.load(log_path + "/log_prev.pth")
        return log_dict

    def add_scalar(self, name, val, step):
        # self.writer.add_scalar(name, val, step)
        try:
            self.log_dict[name] += [(time.time(), int(step), float(val))]
        except KeyError:
            self.log_dict[name] = [(time.time(), int(step), float(val))]

        # if self.enable_wandb:
            # if "_itr" in name:
                # self.wandb_log.log({"iteration": step, name: float(val)})
            # else:
                # self.wandb_log.log({"epoch": step, name: float(val)})

    def add_scalars(self, name, val_dict, step):
        # self.writer.add_scalars(name, val_dict, step)
        for key, val in val_dict.items():
            self.log_dict[name+key] += [(time.time(), int(step), float(val))]

    def add_figure(self, name, val):
        # self.writer.add_figure(name, val, step)
        path = self.log_path + "/" + name + ".png"
        val.savefig(path)
        # if self.enable_wandb:
            # self.wandb_log.log({name: wandb.Image(path)})
            
    def save_log(self, is_final_result=False):
        try:
            os.makedirs(self.log_path)
        except os.error:
            pass

        if not is_final_result:
            log_prev = os.path.join(self.log_path, "log_prev.pth")
            log_curr = os.path.join(self.log_path, "log_curr.pth")

            # no existing logs
            if not (os.path.exists(log_curr) or os.path.exists(log_prev)):
                pass
            elif os.path.exists(log_curr):
                # overwrite log_prev with log_curr
                cmd = "cp -r {} {}".format(log_curr, log_prev)
                os.system(cmd)

        log_final = os.path.join(self.log_path, "log_final.pth")
        saved_path = log_final if is_final_result else log_curr

        load_successfully = False
        save_counter = 0
        while not load_successfully and save_counter < 10:
            torch.save(dict(self.log_dict), saved_path)
            try:
                torch.load(saved_path)
            except:
                print('Failed to load log!')
                save_counter += 1
                print('{}th attempt at saving log'.format(save_counter))
            else:
                print('Log saved and verified!')
                load_successfully = True
    # def save_log(self):
        # try:
            # os.makedirs(self.log_path)
        # except os.error:
            # pass

        # log_curr = os.path.join(self.log_path, "log_curr.pth")
        # log_prev = os.path.join(self.log_path, "log_prev.pth")

        # # no existing logs
        # if not (os.path.exists(log_curr) or os.path.exists(log_prev)):
            # torch.save(dict(self.log_dict), log_curr)

        # elif os.path.exists(log_curr):
            # # overwrite log_prev with log_curr
            # cmd = "cp -r {} {}".format(log_curr, log_prev)
            # os.system(cmd)
            # torch.save(dict(self.log_dict), log_curr)

class wandbLogger(object):
    def __init__(self, args):
        self.log_path = args.j_dir+"/log/"
        self.wandb_log = wandb.init(name=args.j_dir.split("/")[-1],
                                    project=args.wandb_project,
                                    dir=args.j_dir,
                                    id=str(args.j_id),
                                    resume=True)
        self.wandb_log.config.update(args)

    def upload(self, logger, epoch):
        all_keys = [*logger.log_dict.keys()]
        for _epoch in range(epoch):
            commit = ((_epoch+1)==epoch)
            wandb_dict = OrderedDict()
            wandb_dict['epoch'] = _epoch+1

            for key in all_keys:
                if (_epoch+1) in np.array(logger.log_dict[key])[:, 1]:
                    idx = (_epoch+1) == np.array(logger.log_dict[key])[:, 1]
                    wandb_dict[key] = np.array(logger.log_dict[key])[idx, 2]
            self.wandb_log.log(wandb_dict, step=_epoch+1, commit=commit)

    # def add_figure(self, name, val):
        # path = self.log_path + "/" + name + ".png"
        # self.wandb_log.log({name: wandb.Image(path)})

def delCheckpoint(j_dir, j_id):
    ckpt_curr = os.path.join(j_dir, str(j_id), "custom_ckpt_curr.pth")
    ckpt_prev = os.path.join(j_dir, str(j_id), "custom_ckpt_prev.pth")
    for ckpt_path in [ckpt_curr, ckpt_prev]:
        if os.path.isfile(ckpt_path):
            print("ckpt {} is deleted".format(ckpt_path))
            cmd = "rm {}".format(ckpt_path)
            os.system(cmd)

