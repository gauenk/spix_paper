# -- basic --
import os
import sys
import time
import math
import glob
import copy
import logging
import importlib
import argparse, yaml
from tqdm import tqdm
dcopy = copy.deepcopy
from pathlib import Path
from easydict import EasyDict as edict
import numpy as np

# -- pytorch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from einops import rearrange,repeat

# -- helper --
from dev_basics.utils.timer import ExpTimer

# -- project imports --
from spix_paper.data import load_data
from spix_paper.losses import load_loss
from spix_paper.models import load_model
import spix_paper.trte_utils as utils
import spix_paper.utils as base_utils
from spix_paper import metrics
from spix_paper import isp


# -- fill missing with defaults --
tr_defs = {"dim":12,"qk_dim":6,"mlp_dim":6,"stoken_size":[8],"block_num":1,
        "heads":1,"M":0.,"use_local":False,"use_inter":False,
        "use_intra":True,"use_ffn":False,"use_nat":False,"nat_ksize":9,
        "affinity_softmax":1.,"topk":100,"intra_version":"v1",
        "data_path":"./data/","data_augment":False,
        "patch_size":128,"data_repeat":1,
        "gpu_ids":"[1]","num_workers":4,
        "model":"model","model_name":"simple",
        "decays":[],"gamma":0.5,"lr":0.0002,"resume":None,
        "log_name":"default_log","exp_name":"default_exp",
        "epochs":50,"log_every":100,"test_every":1,"batch_size":8,"colors":3,
        "base_path":"output/default_basepath/train/",
        "resume_uuid":None,"resume_flag":True,
        "spatial_chunk_size":256,"spatial_chunk_overlap":0.25,
        "gradient_clip":0.,"spix_loss_target":"seg",
        "resume_weights_only":False,
        "save_every_n_epochs":5,"noise_type":"gaussian"}

def run(cfg):

    cfg = base_utils.extract_defaults(cfg,tr_defs)
    if cfg.mname == "empty": return None

    # -- select active gpu devices --
    base_utils.seed_everything(cfg.seed)
    # utils.set_gpus(cfg.gpu_ids)
    torch.set_num_threads(cfg.num_workers)
    device = torch.device('cuda')

    # -- model, loss, data --
    model = load_model(cfg)
    loss_fxn = load_loss(cfg)
    train_dataloader, valid_dataloaders = load_data(cfg)

    # -- noise function --
    sigma = base_utils.optional(cfg,'sigma',0.)
    ntype = base_utils.optional(cfg,'noise_type',"gaussian")
    def pre_process(x):
        if ntype == "gaussian":
            return x + (sigma/255.)*th.randn_like(x),{}
        elif ntype == "isp":
            return isp.run_unprocess(x)
        else:
            raise ValueError(f"Uknown noise type [{ntype}]")
    def post_process(deno,noise_info):
        if ntype == "gaussian":
            return deno
        elif ntype == "isp":
            keys = ['red_gain','blue_gain','cam2rgb']
            args = [noise_info[k] for k in keys]
            return isp.run_process(deno,*args)
        else:
            raise ValueError(f"Uknown noise type [{ntype}]")

    # -- info & parallel --
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('#Params : {:<.4f} [K]'.format(num_parameters / 10 ** 3))
    # exit()
    model = nn.DataParallel(model).to(device)

    # -- optim and sched --
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = MultiStepLR(optimizer, milestones=cfg.decays, gamma=cfg.gamma)

    # -- resume training --
    uuid,base_path = cfg.uuid,cfg.base_path
    resume_uuid,resume_flag = cfg.resume_uuid,cfg.resume_flag
    chkpt_root = utils.get_ckpt_root(uuid,base_path,resume_uuid,resume_flag)
    chkpt,chkpt_fn = utils.get_checkpoint(chkpt_root)
    start_epoch = utils.load_checkpoint(chkpt,model,optimizer,
                                        cfg.resume_weights_only,skip_module=False)
    if cfg.resume_flag and start_epoch > 0:
        print("Resuming from ",chkpt_fn)
        print('select {}, resume training from epoch {}.'.format(chkpt_fn, start_epoch))

    # -- init stat dict --
    stat_dict = utils.init_stat_dict(chkpt,reset=cfg.resume_weights_only)

    # -- setup logging --
    log_path,log_name,chkpt_path = utils.init_logging(cfg,cfg.uuid,cfg.base_path)
    time.sleep(3) # sleep 3 seconds
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    print(model)
    sys.stdout.flush()

    # -----------------------------
    #
    #       Training Loop
    #
    # -----------------------------

    info = {}
    buff = utils.init_metrics_buffer()
    timer_start = time.time()
    for epoch in range(start_epoch, cfg.nepochs+1):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        opt_lr = scheduler.get_last_lr()
        print('##==========={}-training, Epoch: {}, lr: {} =============##'\
              .format('fp32', epoch, opt_lr))
        th.manual_seed(int(cfg.seed)+epoch)
        for iter, (img,seg) in enumerate(train_dataloader):

            # -- timing --
            timer = ExpTimer(False)

            # -- unpack --
            optimizer.zero_grad()
            # img = batch['img']
            # seg = utils.optional(batch,'seg',None)
            img, seg = img.to(device)/255., seg[:,None].to(device)

            # -- optional noise --
            noisy,ninfo = pre_process(img)

            # -- forward --
            timer.sync_start("fwd")
            output = model(noisy,ninfo)
            timer.sync_stop("fwd")

            # -- unpack --
            deno = base_utils.optional(output,'deno',None)
            sims = base_utils.optional(output,'sims',None)

            # -- optionally post-process --
            deno = post_process(deno,ninfo)

            # -- loss --
            timer.sync_start("loss")
            loss = loss_fxn(255*img,seg,deno=255*deno,sims=sims)
            timer.sync_stop("loss")

            timer.sync_start("bwd")
            loss.backward()
            # print("normz: ",utils.grad_norm(model))
            if cfg.gradient_clip > 0:
                clip = cfg.gradient_clip
                th.nn.utils.clip_grad_norm_(model.parameters(),clip)
            timer.sync_stop("bwd")

            timer.sync_start("step")
            optimizer.step()
            timer.sync_stop("step")

            epoch_loss += float(loss)

            # -- update buffer --
            utils.update_metrics_buffer(buff,img,seg,deno,sims)

            # -- logging --
            if (iter + 1) % cfg.log_every == 0:
                cur_steps = (iter + 1) * cfg.batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(cfg.nepochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter + 1)
                stat_dict['losses'].append(avg_loss)

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end

                # print(timer)
                print('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}'.\
                      format(cur_epoch, cur_steps, total_steps, avg_loss, duration))
            sys.stdout.flush()

        # -- compute metrics --
        utils.update_agg(info,buff,epoch)
        utils.view_buff(buff)
        buff = utils.init_metrics_buffer()

        # -- epoch loop --
        if epoch % cfg.save_every_n_epochs == 0:

            # -- save --
            model_str = '%s-epoch=%02d.ckpt'%(cfg.uuid,epoch-1) # "start at 0"
            saved_model_path = os.path.join(chkpt_path,model_str)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'stat_dict': stat_dict
            }, saved_model_path)
            torch.set_grad_enabled(True)

            # -- state name --
            stat_dict_name = os.path.join(log_path, 'stat_dict_%d.yml' % epoch)
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)

        # -- update scheduler --
        scheduler.step()

    return info

