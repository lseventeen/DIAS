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


class Trainer:
    def __init__(self, config, train_loader, val_loader, model, is_2d, loss, optimizer, lr_scheduler):
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
        if self._get_rank() == 0:
            self.checkpoint_dir = os.path.join(
                config.SAVE_DIR, config.EXPERIMENT_ID)

            os.makedirs(self.checkpoint_dir)
          # MONITORING
        self.improved = True
        self.not_improved_count = 0
        self.mnt_best = -math.inf if self.config.TRAIN.MNT_MODE == 'max' else math.inf

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
        return self.checkpoint_dir

    def _train_epoch(self, epoch):
        wrt_mode = "train"
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
            self._update_metrics(
                *get_metrics(torch.softmax(pre, dim=1).cpu().detach().numpy()[:, 1, :, :], gt.cpu().detach().numpy()).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} |DSC {:.4f}  Acc {:.4f}  Sen {:.4f} Spe {:.4f}  IOU {:.4f} AUC {:.4f} clDice {:.4f}|B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.mean, *self._get_metrics_mean().values(), self.batch_time.mean, self.data_time.mean))
            tic = time.time()
            self.lr_scheduler.step_update(epoch * self.num_steps + idx)
        if self._get_rank() == 0:
            wandb.log({f'{wrt_mode}/loss': self.total_loss.mean}, step=epoch)
            for k, v in list(self._get_metrics_mean().items())[:-1]:
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
                if not self.is_2d:
                    img = img.unsqueeze(1)
                with torch.cuda.amp.autocast(enabled=self.config.AMP):

                    predict = self.model(img)
                    loss = self.loss(predict, gt)

                self.total_loss.update(loss.item())
                self._update_metrics(
                    *get_metrics(torch.softmax(predict, dim=1).cpu().detach().numpy()[:, 1, :, :], gt.cpu().detach().numpy()).values())
                tbar.set_description(
                'EVAL ({})  | Loss: {:.4f} |DSC {:.4f}  Acc {:.4f}  Sen {:.4f} Spe {:.4f}  IOU {:.4f} AUC {:.4f} |'.format(
                    epoch, self.total_loss.mean, *self._get_metrics_mean().values()))

        if self._get_rank() == 0:

            wandb.log({f'{wrt_mode}/loss': self.total_loss.mean}, step=epoch)
            for k, v in list(self._get_metrics_mean().items())[:-1]:
                wandb.log({f'{wrt_mode}/{k}': v}, step=epoch)

        log = {
            'val_loss': self.total_loss.mean,
            **self._get_metrics_mean()
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
        self.DSC = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.iou = AverageMeter()
        self.VC = AverageMeter()
        self.cldice = AverageMeter()

    def _update_metrics(self, DSC, acc, sen, spe, iou,auc, cldice):
        self.DSC.update(DSC)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.iou.update(iou)
        self.auc.update(auc)
        self.cldice.update(cldice)

    def _get_metrics_mean(self):

        return {
            
            "DSC": self.DSC.mean,
            "Acc": self.acc.mean,
            "Sen": self.sen.mean,
            "Spe": self.spe.mean,
            "IOU": self.iou.mean,
            "AUC": self.auc.mean,
            "cldice": self.cldice.mean,
        }
    def _get_metrics_std(self):

        return {
            
            "DSC": self.DSC.std,
            "Acc": self.acc.std,
            "Sen": self.sen.std,
            "Spe": self.spe.std,
            "IOU": self.iou.std,
            "AUC": self.auc.std,
            "cldice": self.cldice.std,
        }


