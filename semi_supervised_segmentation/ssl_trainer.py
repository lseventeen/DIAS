import os
import time
import math
import torch
from loguru import logger
from tqdm import tqdm
from utils.helpers import to_cuda
from utils.metrics import AverageMeter, get_metrics, get_metrics
import wandb
import torch.distributed as dist
from trainer import Trainer
gl_epoch = 0


class SSL_Trainer(Trainer):
    def __init__(self, config, train_loader, val_loader, model,is_2d, loss, optimizer, lr_scheduler, tag):
        self.config = config
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss = loss
        self.model = model
        self.is_2d = is_2d
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_steps = len(self.train_loader)
        self.tag = tag
        if self._get_rank() == 0:
            self.checkpoint_dir = os.path.join(
                config.SAVE_DIR, config.EXPERIMENT_ID, tag)

            os.makedirs(self.checkpoint_dir)
          # MONITORING
        self.improved = True
        self.not_improved_count = 0
        self.mnt_best = -math.inf if self.config.TRAIN.MNT_MODE == 'max' else math.inf



    def _train_epoch(self, epoch):
        wrt_mode = self.tag+"_train"
        self.model.train()

        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        tic = time.time()

        for idx, (img, gt) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            img = to_cuda(img)
            gt = to_cuda(gt)
            if not self.is_2d:
                img = img.unsqueeze(1)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config.AMP):
                pre = self.model(img)
                loss = self.loss(pre, gt)
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
                *get_metrics(torch.softmax(pre, dim=1).cpu().detach().numpy()[:, 1, :, :], gt.cpu().detach().numpy()).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} |DSC {:.4f}  Acc {:.4f}  Sen {:.4f} Spe {:.4f}  IOU {:.4f} AUC {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
            tic = time.time()
            self.lr_scheduler.step_update(epoch * self.num_steps + idx)
        if self._get_rank() == 0:
            wandb.log({f'{wrt_mode}/loss': self.total_loss.average})
            for k, v in list(self._metrics_ave().items())[:-1]:
                wandb.log({f'{wrt_mode}/{k}': v})
            for i, opt_group in enumerate(self.optimizer.param_groups):
                wandb.log({f'{wrt_mode}/Learning_rate_{i}': opt_group['lr']})

    def _valid_epoch(self, epoch):
        logger.info(f'\n###### {self.tag} EVALUATION ######')
        self.model.eval()
        wrt_mode = self.tag+"_val"
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for idx, (img, gt) in enumerate(tbar):
                img = to_cuda(img)
                gt = to_cuda(gt)
                if not self.is_2d:
                    img = img.unsqueeze(1)
                with torch.cuda.amp.autocast(enabled=self.config.AMP):

                    predict = self.model(img)
                    loss = self.loss(predict, gt)

                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(torch.softmax(predict, dim=1).cpu().detach().numpy()[:, 1, :, :], gt.cpu().detach().numpy()).values())
                tbar.set_description(
                'EVAL ({})  | Loss: {:.4f} |DSC {:.4f}  Acc {:.4f}  Sen {:.4f} Spe {:.4f}  IOU {:.4f} AUC {:.4f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values()))

        if self._get_rank() == 0:

            wandb.log({f'{wrt_mode}/loss': self.total_loss.average})
            for k, v in list(self._metrics_ave().items())[:-1]:
                wandb.log({f'{wrt_mode}/{k}': v})

        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log



