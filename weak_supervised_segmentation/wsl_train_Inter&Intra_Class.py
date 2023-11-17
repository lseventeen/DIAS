import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import time
import math
import torch
from loguru import logger
from tqdm import tqdm
from utils.helpers import to_cuda
from utils.metrics import AverageMeter, get_metrics
import wandb
import torch.distributed as dist
import argparse
from loguru import logger
from data import build_train_loader
from utils.helpers import seed_torch
from losses.losses import *
from datetime import datetime
import wandb
from configs.config import get_config
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def intra_class_variance(prob, img):
    mean_std = torch.std(torch.min(img, dim=1, keepdim=True)[
                         1].float() * prob, dim=[2, 3])
    return mean_std.mean()


def inter_class_variance(prob, img):
    mean_std = torch.std(torch.mean(torch.min(img, dim=1, keepdim=True)[
                         1].float() * prob, dim=[2, 3]), dim=1)
    return mean_std.mean()


class Trainer:
    def __init__(self, config, train_loader, val_loader, model,is_2d,  optimizer, lr_scheduler):
        self.config = config

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.model = model
        self.is_2d = is_2d
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_steps = len(self.train_loader)
        if self._get_rank() == 0:
            self.checkpoint_dir = os.path.join(
                config.SAVE_DIR, config.EXPERIMENT_ID)

            os.makedirs(self.checkpoint_dir)
          # MONITORING
        self.improved = True
        self.not_improved_count = 0
        self.mnt_best = -math.inf if self.config.TRAIN.MNT_MODE == 'max' else math.inf
        self.loss = DC_and_CE_loss({}, {}, ignore_label=255)

    def train(self):

        for epoch in range(1, self.config.TRAIN.EPOCHS+1):

            if self.config.DIS:
                self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.config.TRAIN.VAL_NUM_EPOCHS == 0:
                results = self._valid_epoch(epoch)
                if self._get_rank() == 0:
                    logger.info(f'## Info for epoch {epoch} ## ')
                    for k, v in results.items():
                        logger.info(f'{str(k):15s}: {v}')
                    if self.config.TRAIN.MNT_MODE != 'off' and epoch >= 10:
                        try:
                            if self.config.TRAIN.MNT_MODE == 'min':
                                self.improved = (
                                    results[self.config.TRAIN.MNT_METRIC] <= self.mnt_best)
                            else:
                                self.improved = (
                                    results[self.config.TRAIN.MNT_METRIC] >= self.mnt_best)
                        except KeyError:
                            logger.warning(
                                f'The metrics being tracked ({self.config.TRAIN.MNT_METRIC}) has not been calculated. Training stops.')
                            break

                        if self.improved:
                            self.mnt_best = results[self.config.TRAIN.MNT_METRIC]
                            self.not_improved_count = 0
                        else:
                            self.not_improved_count += 1
                        if self.not_improved_count >= self.config.TRAIN.EARLY_STOPPING:
                            logger.info(
                                f'\nPerformance didn\'t improve for {self.config.TRAIN.EARLY_STOPPING} epochs')
                            logger.warning('Training Stoped')
                            break

            # SAVE CHECKPOINT
            if self._get_rank() == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

    def _train_epoch(self, epoch):
        wrt_mode = "train"
        self.model.train()

        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        tic = time.time()

        for idx, (img, gt) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            if not self.is_2d:
                img = img.unsqueeze(1)
            img = to_cuda(img)
            gt = to_cuda(gt)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config.AMP):
                pre = self.model(img)
                outputs_soft = torch.softmax(pre, dim=1)
                consistency_loss = inter_class_variance(
                    outputs_soft, img) - intra_class_variance(outputs_soft,  img)
                loss = self.loss(pre, gt)+0.01*consistency_loss
            if self.config.AMP:
                self.scaler.scale(loss).backward()
                if self.config.TRAIN.DO_BACKPROP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:

                loss.backward()
                if self.config.TRAIN.DO_BACKPROP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()
            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)

            self._metrics_update(
                *get_metrics(torch.softmax(pre[:, :self.config.DATASET.NUM_CLASSES], dim=1).cpu().detach().numpy()[:, 1, :, :], gt.cpu().detach().numpy()).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
            tic = time.time()
            self.lr_scheduler.step_update(epoch * self.num_steps + idx)
        if self._get_rank() == 0:
            wandb.log({f'{wrt_mode}/loss': self.total_loss.average}, step=epoch)
            for k, v in list(self._metrics_ave().items())[:-1]:
                wandb.log({f'{wrt_mode}/{k}': v}, step=epoch)
            for i, opt_group in enumerate(self.optimizer.param_groups):
                wandb.log(
                    {f'{wrt_mode}/Learning_rate_{i}': opt_group['lr']}, step=epoch)

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for idx, (img, gt) in enumerate(tbar):
                img = to_cuda(img)
                gt = to_cuda(gt)

                with torch.cuda.amp.autocast(enabled=self.config.AMP):

                    predict = self.model(img)
                    loss = self.loss(predict, gt)

                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(torch.softmax(predict[:, :self.config.DATASET.NUM_CLASSES], dim=1).cpu().detach().numpy()[:, 1, :, :], gt.cpu().detach().numpy()).values())
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))

        if self._get_rank() == 0:

            wandb.log({f'{wrt_mode}/loss': self.total_loss.average}, step=epoch)
            for k, v in list(self._metrics_ave().items())[:-1]:
                wandb.log({f'{wrt_mode}/{k}': v}, step=epoch)

        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def _save_checkpoint(self, epoch, save_best=True):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'final_checkpoint.pth')
        logger.info(f'Saving a checkpoint: {filename}')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, 'best_model.pth')
            logger.info(f"Saving current best: {filename}")
            torch.save(state, filename)

        return filename

    def _get_rank(self):
        """get gpu id in distribution training."""
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()
        self.VC = AverageMeter()

    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }


def parse_option():
    parser = argparse.ArgumentParser("CVSS_training")
    parser.add_argument('--cfg', type=str, metavar="FILE",
                        help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument("--tag", help='tag of experiment')
    parser.add_argument("-wm", "--wandb_mode", default="online")
    parser.add_argument("-mt", "--model_type", default="FR_UNet")
    parser.add_argument("-st", "--scribble_type", default="scribble")
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help="batch size for single GPU")
    parser.add_argument('-ed', '--enable_distributed', help="training without DDP",
                        required=False, action="store_true")
    parser.add_argument('-ws', '--world_size', type=int,
                        help="process number for DDP")
    args = parser.parse_args()
    config = get_config(args)

    return args, config


def main(config):
    if config.DIS:
        mp.spawn(main_worker,
                 args=(config,),
                 nprocs=config.WORLD_SIZE,)
    else:
        main_worker(0, config)


def main_worker(local_rank, config):
    if local_rank == 0:
        config.defrost()
        config.EXPERIMENT_ID = f"{config.SCRIBBLE_TYPE}_{config.WANDB.TAG}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        config.freeze()
        wandb.init(project=config.WANDB.PROJECT,
                   name=config.EXPERIMENT_ID, config=config, mode=config.WANDB.MODE)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    torch.cuda.set_device(local_rank)
    if config.DIS:
        dist.init_process_group(
            "nccl", init_method='env://', rank=local_rank, world_size=config.WORLD_SIZE)
    seed = config.SEED + local_rank
    seed_torch(seed)
    cudnn.benchmark = True

    train_loader, val_loader = build_train_loader(config)
    model,is_2d = build_model(config)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    if config.DIS:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    logger.info(f'\n{model}\n')

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    trainer = Trainer(config=config,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      model=model.cuda(),
                      is_2d=is_2d,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10000"
    _, config = parse_option()

    main(config)
