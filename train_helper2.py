# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""
import json
# import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


class Opt:
    # evolve超参进化次数，需要大量重复运算，默认不需要就设0
    def __init__(self, weights, device, data, imgsz=640, rect=False, resume=False, nosave=False, noval=True,
                 noautoanchor=True, noplots=True, evolve=0, bucket='', cache='ram', image_weights=False,
                 multi_scale=False, single_cls=False, optimizer='SGD', sync_bn=False, workers=0,
                 project='runs/train', name='train', exist_ok=False, quad=False, cos_lr=False,
                 label_smoothing=0.0, patience=100, freeze=[0, ], save_period=-1, seed=0,
                 local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias='latest',
                 batch_size=16, epochs=100, cfg="", hyp='data/hyps/hyp.scratch-low.yaml', save_dir="") -> None:
        super().__init__()
        self.weights = weights
        self.cfg = cfg
        self.data = data
        self.hyp = hyp
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.rect = rect
        self.resume = resume
        self.nosave = nosave
        self.noval = noval
        self.noautoanchor = noautoanchor
        self.noplots = noplots
        self.evolve = evolve
        self.bucket = bucket
        self.cache = cache
        self.image_weights = image_weights
        self.device = device
        self.multi_scale = multi_scale
        self.single_cls = single_cls
        self.optimizer = optimizer
        self.sync_bn = sync_bn
        self.workers = workers
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.quad = quad
        self.cos_lr = cos_lr
        self.label_smoothing = label_smoothing
        self.patience = patience
        self.freeze = freeze
        self.save_period = save_period
        self.seed = seed
        self.local_rank = local_rank
        self.entity = entity
        self.upload_dataset = upload_dataset
        self.bbox_interval = bbox_interval
        self.artifact_alias = artifact_alias
        self.save_dir = save_dir


class TrainingHelper:

    def __init__(self, opt, lvl=1, epoch_start=0) -> None:
        super().__init__()
        self.opt = opt
        # self.weight = None
        # mP, mR, mAP@50, mAP@50-95, average_loss, maps,
        # mP（mean Precision）：平均精确度，是指在不同阈值下测量的平均精确度。精确度是正确预测的正样本数除以总的正样本数的比例。mP是针对不同类别计算的。
        # mR（mean Recall）：平均召回率，是指在不同阈值下测量的平均召回率。召回率是正确预测的正样本数除以总的正样本数的比例。mR也是针对不同类别计算的。
        # mAP@50（mean Average Precision at 50 IoU）：平均IoU为50时的平均精确度。IoU（Intersection over Union）是预测边界框与实际边界框的交集与并集的比率，50表示阈值为50%的IoU。
        # mAP@50-95（mean Average Precision at IoU from 50 to 95）：在不同IoU阈值范围内计算的平均精确度。通常会计算从50%到95%的不同IoU阈值下的平均精确度。
        # average_loss：模型在验证集上的平均损失值。损失值是模型预测与真实标签之间的差异度量。
        # maps（ ）：所有类别的平均精确度，通常用来总结整个目标检测系统的性能。
        # 这些指标用于评估模型在验证集上的性能，以确定模型是否足够准确，以便在实际应用中进行目标检测任务。更高的精确度和召回率，以及更高的平均精确度通常表示模型性能更好。
        # 得分，由0.1*mAP50 + 0.9*nAp50-95 得到
        self.fi = -1
        self.val_loader = None
        self.imgsz = None
        self.data_dict = None
        self.train_loader = None
        self.dataset = None
        self.start_epoch = 0
        self.epochs = 0
        self.model = None
        self.maps = None
        self.nc = None
        self.device = None
        self.nb = None
        self.optimizer = None
        self.nw = None
        self.nbs = None
        self.batch_size = None
        self.hyp = None
        self.lf = None
        self.gs = None
        self.amp = None
        self.compute_loss = None
        self.scaler = None
        self.ema = None
        self.best_fitness = None
        self.last_opt_step = None
        self.accumulate = None
        self.single_cls = None
        self.save_dir = None
        self.epoch_now = epoch_start
        self.lvl = lvl
        self.init()

    # 执行训练，更新对象的weight, fi, val等参数，并保存对应的模型
    def init(self, callbacks=Callbacks()):
        opt = self.opt
        # Checks
        # if RANK in {-1, 0}:
        #     print_args(vars(opt))
        #     check_git_status()
        #     check_requirements(ROOT / 'requirements.txt')

        # Resume (from specified or most recent last.pt)
        if opt.resume and not check_comet_resume(opt) and not opt.evolve:
            last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
            opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
            opt_data = opt.data  # original dataset
            if opt_yaml.is_file():
                with open(opt_yaml, errors='ignore') as f:
                    d = yaml.safe_load(f)
            else:
                d = torch.load(last, map_location='cpu')['opt']
            # opt = argparse.Namespace(**d)  # replace
            opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
            if is_url(opt_data):
                opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
        else:
            opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
                check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(
                    opt.project)  # checks
            assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
            if opt.evolve:
                if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                    opt.project = str(ROOT / 'runs/evolve')
                opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
            if opt.name == 'cfg':
                opt.name = Path(opt.cfg).stem  # use model.yaml as name
            opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

        # DDP mode
        self.device = select_device(opt.device, batch_size=opt.batch_size)
        if LOCAL_RANK != -1:
            msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
            assert not opt.image_weights, f'--image-weights {msg}'
            assert not opt.evolve, f'--evolve {msg}'
            assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
            assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            torch.cuda.set_device(LOCAL_RANK)
            self.device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

        # Evolve hyperparameters (optional)
        else:
            # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
            meta = {
                'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

            with open(opt.hyp, errors='ignore') as f:
                self.hyp = yaml.safe_load(f)  # load hyps dict
                if 'anchors' not in self.hyp:  # anchors commented in hyp.yaml
                    self.hyp['anchors'] = 3
            if opt.noautoanchor:
                del self.hyp['anchors'], meta['anchors']
            opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
            if opt.bucket:
                # download evolve.csv if exists
                subprocess.run([
                    'gsutil',
                    'cp',
                    f'gs://{opt.bucket}/evolve.csv',
                    str(evolve_csv), ])

        self.save_dir, self.epochs, self.batch_size, weights, self.single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
            Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
                opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
        callbacks.run('on_pretrain_routine_start')

        # Directories
        w = self.save_dir / 'weights'  # weights dir
        (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
        last, best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(self.hyp, str):
            with open(self.hyp, errors='ignore') as f:
                self.hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in self.hyp.items()))
        opt.hyp = self.hyp.copy()  # for saving hyps to checkpoints

        # Loggers
        data_dict = None
        if RANK in {-1, 0}:
            loggers = Loggers(self.save_dir, weights, opt, self.hyp, LOGGER)  # loggers instance

            # Register actions
            for k in methods(loggers):
                callbacks.register_action(k, callback=getattr(loggers, k))

            # Process custom dataset artifact link
            data_dict = loggers.remote_dataset
            if resume:  # If resuming runs from remote artifact
                self.weights, self.epochs, self.hyp, self.batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Config
        plots = not evolve and not opt.noplots  # create plots
        cuda = self.device.type != 'cpu'
        init_seeds(opt.seed + 1 + RANK, deterministic=True)
        with torch_distributed_zero_first(LOCAL_RANK):
            self.data_dict = data_dict or check_dataset(data)  # check if None
        train_path, val_path = self.data_dict['train'], self.data_dict['val']
        self.nc = 1 if self.single_cls else int(self.data_dict['nc'])  # number of classes
        names = {0: 'item'} if self.single_cls and len(self.data_dict['names']) != 1 else self.data_dict[
            'names']  # class names
        is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

        # Model
        check_suffix(weights, '.pt')  # check weights
        pretrained = weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            self.model = Model(cfg or ckpt['model'].yaml, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(
                self.device)  # create
            exclude = ['anchor'] if (cfg or self.hyp.get('anchors')) and not resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from {weights}')  # report
        else:
            self.model = Model(cfg, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
        self.amp = check_amp(self.model)  # check AMP

        # Freeze
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze):
                LOGGER.info(f'freezing {k}')
                v.requires_grad = False

        # Image size
        self.gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.imgsz = check_img_size(opt.imgsz, self.gs, floor=self.gs * 2)  # verify imgsz is gs-multiple

        # Batch size
        if RANK == -1 and self.batch_size == -1:  # single-GPU only, estimate best batch size
            self.batch_size = check_train_batch_size(self.model, self.imgsz, self.amp)
            loggers.on_params_update({'batch_size': self.batch_size})

        # Optimizer
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.batch_size * self.accumulate / self.nbs  # scale weight_decay
        self.optimizer = smart_optimizer(self.model, opt.optimizer, self.hyp['lr0'], self.hyp['momentum'],
                                         self.hyp['weight_decay'])

        # Scheduler
        if opt.cos_lr:
            self.lf = one_cycle(1, self.hyp['lrf'], self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        scheduler = lr_scheduler.LambdaLR(self.optimizer,
                                          lr_lambda=self.lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        self.ema = ModelEMA(self.model) if RANK in {-1, 0} else None

        # Resume
        # best_fitness, self.start_epoch = 0.0, 0
        if pretrained:
            if resume:
                self.best_fitness, self.start_epoch, self.epochs = smart_resume(ckpt, self.optimizer, self.ema, weights,
                                                                                self.epochs,
                                                                                resume)
            del ckpt, csd

        # DP mode
        if cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning(
                'WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                'See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started.'
            )
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if opt.sync_bn and cuda and RANK != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            LOGGER.info('Using SyncBatchNorm()')
        if self.train_loader == None:
            # Trainloader
            self.train_loader, self.dataset = create_dataloader(train_path,
                                                                self.imgsz,
                                                                self.batch_size // WORLD_SIZE,
                                                                self.gs,
                                                                self.single_cls,
                                                                hyp=self.hyp,
                                                                augment=True,
                                                                cache=None if opt.cache == 'val' else opt.cache,
                                                                rect=opt.rect,
                                                                rank=LOCAL_RANK,
                                                                workers=workers,
                                                                image_weights=opt.image_weights,
                                                                quad=opt.quad,
                                                                prefix=colorstr('train: '),
                                                                shuffle=True,
                                                                seed=opt.seed)
        labels = np.concatenate(self.dataset.labels, 0)
        mlc = int(labels[:, 0].max())  # max label class
        assert mlc < self.nc, f'Label class {mlc} exceeds nc={self.nc} in {data}. Possible class labels are 0-{self.nc - 1}'

        # Process 0
        if RANK in {-1, 0}:
            if not resume:
                if not opt.noautoanchor:
                    check_anchors(self.dataset, model=self.model, thr=self.hyp['anchor_t'],
                                  imgsz=self.imgsz)  # run AutoAnchor
                self.model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end', labels, names)

        # DDP mode
        if cuda and RANK != -1:
            self.model = smart_DDP(self.model)

        # Model attributes
        nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        self.hyp['box'] *= 3 / nl  # scale to layers
        self.hyp['cls'] *= self.nc / 80 * 3 / nl  # scale to classes and layers
        self.hyp['obj'] *= (self.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.hyp['label_smoothing'] = opt.label_smoothing
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(
            self.device) * self.nc  # attach class weights
        self.model.names = names

        # Start training
        t0 = time.time()
        self.nb = len(self.train_loader)  # number of batches
        self.nw = max(round(self.hyp['warmup_epochs'] * self.nb),
                      100)  # number of warmup iterations, max(3 epochs, 100 iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        self.last_opt_step = -1
        self.maps = np.zeros(self.nc)  # mAP per class
        scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        stopper, stop = EarlyStopping(patience=opt.patience), False
        self.compute_loss = ComputeLoss(self.model)  # init loss class
        callbacks.run('on_train_start')
        LOGGER.info(f'Image sizes {self.imgsz} train, {self.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        tmp_val_result = []

        self.val_loader = create_dataloader(val_path,
                                            self.imgsz,
                                            self.opt.batch_size // WORLD_SIZE * 2,
                                            self.gs,
                                            self.single_cls,
                                            hyp=self.hyp,
                                            cache=None if noval else opt.cache,
                                            rect=True,
                                            rank=-1,
                                            workers=workers * 2,
                                            pad=0.5,
                                            prefix=colorstr('val: '))[0]

    def train(self, callbacks=Callbacks()):
        self.compute_loss = ComputeLoss(self.model)  # init loss class
        # EMA
        self.ema = ModelEMA(self.model) if RANK in {-1, 0} else None
        # for epoch in range(self.start_epoch,
        #                    self.epochs):  # epoch ------------------------------------------------------------------
        # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步
        for epoch in range(self.epoch_now, self.lvl + self.epoch_now):
            callbacks.run('on_train_epoch_start')
            self.model.train()

            # Update image weights (optional, single-GPU only)
            if self.opt.image_weights:
                cw = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2 / self.nc  # class weights
                iw = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=cw)  # image weights
                self.dataset.indices = random.choices(range(self.dataset.n), weights=iw,
                                                      k=self.dataset.n)  # rand weighted idx

            # Update mosaic border (optional)
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(3, device=self.device)  # mean losses
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            LOGGER.info(
                ('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
            if RANK in {-1, 0}:
                pbar = tqdm(pbar, total=self.nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
            self.optimizer.zero_grad()
            for i, (
                    imgs, targets, paths,
                    _) in pbar:  # batch -------------------------------------------------------------
                callbacks.run('on_train_batch_start')
                ni = i + self.nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= self.nw:
                    xi = [0, self.nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [self.hyp['warmup_bias_lr'] if j == 0 else 0.0,
                                             x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

                # Multi-scale
                if self.opt.multi_scale:
                    sz = random.randrange(int(self.imgsz * 0.5),
                                          int(self.imgsz * 1.5) + self.gs) // self.gs * self.gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / self.gs) * self.gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    pred = self.model(imgs)  # forward
                    loss, loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if self.opt.quad:
                        loss *= 4.1

                # Backward
                self.scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - self.last_opt_step >= self.accumulate:
                    self.scaler.unscale_(self.optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
                    self.scaler.step(self.optimizer)  # optimizer.step
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)
                    self.last_opt_step = ni

                # Log
                if RANK in {-1, 0}:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                         (f'{epoch}/{self.epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                    callbacks.run('on_train_batch_end', self.model, ni, imgs, targets, paths, list(mloss))
                    if callbacks.stop_training:
                        return
                # end batch ------------------------------------------------------------------------------------------------

            # if RANK in {-1, 0}:
            #     # mAP 评估
            #     self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            #     results, maps, _ = validate.run(self.data_dict,
            #                                     batch_size=self.batch_size // WORLD_SIZE * 2,
            #                                     imgsz=self.imgsz,
            #                                     half=self.amp,
            #                                     model=self.ema.ema,
            #                                     single_cls=self.single_cls,
            #                                     dataloader=self.val_loader,
            #                                     save_dir=self.save_dir,
            #                                     plots=False,
            #                                     callbacks=callbacks,
            #                                     compute_loss=self.compute_loss)
            self.epoch_now += 1

    def get_ckpt(self):
        ckpt = {
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'opt': vars(self.opt),
            'git': GIT_INFO,  # {remote, branch, commit} if a git repo
            'date': datetime.now().isoformat()}

        return ckpt

    def val(self, callbacks=Callbacks()):
        self.compute_loss = ComputeLoss(self.model)  # init loss class
        # EMA
        self.ema = ModelEMA(self.model) if RANK in {-1, 0} else None
        results, maps, _ = validate.run(self.data_dict,
                                        batch_size=self.opt.batch_size // WORLD_SIZE * 2,
                                        imgsz=self.imgsz,
                                        half=self.amp,
                                        model=self.ema.ema,
                                        single_cls=self.opt.single_cls,
                                        dataloader=self.val_loader,
                                        save_dir=Path(self.opt.save_dir),
                                        plots=False,
                                        callbacks=callbacks,
                                        compute_loss=self.compute_loss)

        return results


#
if __name__ == '__main__':
    # helper = None
    # lvl = [(2, 640)for i in range(10)]
    lvl = [(2, 240), (2, 240), (2, 240), (2, 480), (2, 480), (2, 480), (2, 640), (2, 640), (2, 640), (2, 240)]

    for _type in [ "crossing"]:
        clients = []
        for i in [j for j in range(10)]:
            opt = Opt(
                weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yolov5n.pt',
                device='0',
                # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
                data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\{}\\client{}.yml'.format(_type, i),
                epochs=150,
                imgsz=lvl[i][1],
            )
            helper = TrainingHelper(opt, 1)
            tmp = []
            if not os.path.exists("./base_models4/{}/client{}/".format(_type, i)):
                os.makedirs("./base_models4/{}/client{}/".format(_type, i))
            for j in range(150):
                helper.train()
                torch.save(helper.get_ckpt(), "./base_models4/{}/client{}/epoch{}.pt".format(_type, i, j))
                mp, mr, map50, _, _, _, _ = helper.val()
                tmp.append((mp, mr, map50))
            clients.append(tmp)
        with open("./base_models4/{}/val".format(_type), "w") as file:
            file.write(json.dumps(clients))

    # for _type in ['main_road']:
    #     val = []
    #     for i in [1, 7, 9]:
    #         for p in [240, 480, 640]:
    #             tmp = []
    #             # if not os.path.exists("./motivation/{}/client{}".format(_type, i)):
    #             #     os.makedirs("./motivation/{}/client{}/".format(_type, i))
    #             opt = Opt(
    #                 weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yolov5n.pt',
    #                 device='0',
    #                 # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
    #                 data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\{}\\client{}.yml'.format(_type, i),
    #                 epochs=225,
    #                 imgsz=p,
    #             )
    #             helper = TrainingHelper(opt, 1)
    #             for j in range(225):
    #                 helper.train()
    #                 # torch.save(helper.get_ckpt(), "./motivation/{}/client{}/epoch{}.pt".format(_type, i, j))
    #                 mp, mr, map50, _, _, _, _ = helper.val()
    #                 tmp.append((mp, mr, map50))
    #             val.append(tmp)
    #         with open("./motivation/{}_val".format(_type) + "_" + str(i), "w") as file:
    #             file.write(json.dumps(val))

    # val = []
    # for i in [4]:
    #     for p in [240, 480, 640]:
    #         tmp = []
    #         # if not os.path.exists("./motivation/{}/client{}".format(_type, i)):
    #         #     os.makedirs("./motivation/{}/client{}/".format(_type, i))
    #         opt = Opt(
    #             weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yolov5n.pt',
    #             device='0',
    #             # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
    #             data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\total.yml',
    #             epochs=225,
    #             imgsz=p,
    #         )
    #         helper = TrainingHelper(opt, 1)
    #         for j in range(225):
    #             helper.train()
    #             # torch.save(helper.get_ckpt(), "./motivation/{}/client{}/epoch{}.pt".format(_type, i, j))
    #             mp, mr, map50, _, _, _, _ = helper.val()
    #             tmp.append((mp, mr, map50))
    #         val.append(tmp)
    #     with open("./motivation/total_val", "w") as file:
    #         file.write(json.dumps(val))

    # a = torch.load("./init_model/model0.pt")
    # b = torch.load("./yolov5n.pt")
    # print(1)

#     helper = TrainingHelper(opt)
#     for i in range(5):
#         helper.train()
#     helper.val()
#     opt = Opt(
#         weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yolov5n.pt',
#         device='0',
#         # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
#         data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\client4.yml', epochs=100,
#         imgsz=352,
#     )
#     helper2 = TrainingHelper(opt)
#     helper2.model = helper.model
#     helper2.val()

#     for i in range(100):
#         t = time.time()
#         helper.model_val(model)
#         print(time.time() - t)

#     opt = Opt(weights=ROOT / 'runs\\train\\train7\\weights\\best.pt', device='0', data=ROOT / 'data\\tank.v5i.yolov8\\data.yaml', epochs=1,
#               )
#     helper = TrainingHelper(opt)
#     helper.main()
#     a = helper.weight["model"].state_dict()
#     for key, value in a.items():
#         print(2)
#     print(1)
