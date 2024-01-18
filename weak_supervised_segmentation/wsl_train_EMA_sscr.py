import os
import time
import math
import torch
from loguru import logger
from tqdm import tqdm
from utils.helpers import to_cuda
from utils.metrics import AverageMeter, get_metrics
import wandb
import torch.distributed as dist
import losses.losses as losses
import argparse
from loguru import logger
from data import build_train_loader,build_PLC_train_loader
from utils.helpers import seed_torch
from losses.losses import *
from datetime import datetime
import wandb
from configs.config import get_config
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
import os
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss,MSELoss
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis

class Trainer:
    def __init__(self, config, train_loader, val_loader, model, optimizer,lr_scheduler):
        self.config = config

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.model = model
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
        self.pce_loss = CrossEntropyLoss(ignore_index=255)
        self.ce_loss = CrossEntropyLoss()
        self.con_loss = MSELoss()
        self.pseudo_weight = config.TRAIN.PSEUDO_LOSS_WIGHT
        self.consistency_weight = config.TRAIN.CONSISTENCY_LOSS_WIGHT
        # self.1- Func.cosine_similarity(outputs_cut["pred_masks"], mixed_output, dim=1).mean()

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
                                self.improved = (results[self.config.TRAIN.MNT_METRIC] <= self.mnt_best )
                            else:
                                self.improved = ( results[self.config.TRAIN.MNT_METRIC] >= self.mnt_best )
                        except KeyError:
                            logger.warning(f'The metrics being tracked ({self.config.TRAIN.MNT_METRIC}) has not been calculated. Training stops.')
                            break

                        if self.improved:
                            self.mnt_best = results[self.config.TRAIN.MNT_METRIC]
                            self.not_improved_count = 0
                        else:
                            self.not_improved_count += 1
                        if self.not_improved_count >= self.config.TRAIN.EARLY_STOPPING:
                            logger.info(f'\nPerformance didn\'t improve for {self.config.TRAIN.EARLY_STOPPING} epochs')
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

        for idx, (img1,img2, gt) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            img1 = to_cuda(img1)
            img2 = to_cuda(img2)
            gt = to_cuda(gt.squeeze(1))
            self.optimizer.zero_grad()
          
            with torch.cuda.amp.autocast(enabled=self.config.AMP):
                pre1 = self.model(img1)
                # pre2 = self.model(img2)
                # ent_loss = losses.entropy_loss(torch.softmax(pre, dim=1), C=3)
                pce_loss1 = self.pce_loss(pre1, gt)
                # pce_loss2 = self.pce_loss(pre2, gt)
                # x = torch.softmax(pre1[:,:-1,:,:], dim=1)
                # pseudo_gt1 = torch.argmax(torch.softmax(pre1[:,:-1,:,:], dim=1),dim=1, keepdim=False)
                # pseudo_gt2 = torch.argmax(torch.softmax(pre2[:,:-1,:,:], dim=1),dim=1, keepdim=False)
                # pseudo_gt1[gt==0]=0
                # pseudo_gt1[gt==1]=1
                # pseudo_gt2[gt==0]=0
                # pseudo_gt2[gt==1]=1
                # for i in pseudo_gt1 
                # ce_loss1 = self.ce_loss(pre1, pseudo_gt1)
                # ce_loss1 = self.ce_loss(pre1, pseudo_gt2)
                # ce_loss2 = self.ce_loss(pre2, pseudo_gt1)
                # ce_loss1 = self.ce_loss(pre1, pseudo_gt1)
                # ce_loss2 = self.ce_loss(pre2, pseudo_gt2)
                # self.consistency_loss = self.con_loss(pseudo_gt1.float(),pseudo_gt2.float())
                # self.consistency_loss = self.con_loss(pre1,pre2)

                # loss = pce_loss1+pce_loss2+self.pseudo_weight*(ce_loss1+ce_loss2)+self.consistency_weight*(self.consistency_loss)
                # loss = pce_loss1+pce_loss2+self.pseudo_weight*(ce_loss1+ce_loss2)
                # loss = pce_loss1+ce_loss1
                loss = pce_loss1
                

                # loss = loss + 0.1*ent_loss
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
            
            # if self.config.TRAIN.MODE == "scrawl" or self.config.TRAIN.MODE =="largeVessel" or self.config.TRAIN.MODE =="centerline":
                # pre = pre[:,0:2,:,:]
            self._metrics_update(
                *get_metrics(torch.softmax(pre1[:,:self.config.DATASET.NUM_CLASSES], dim=1).cpu().detach().numpy()[:,1,:,:], gt.cpu().detach().numpy()).values())
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
            tic = time.time()
            self.lr_scheduler.step_update(epoch * self.num_steps + idx)
           
        if self._get_rank()==0:
            # self.writer.add_scalar(
            #     f'{wrt_mode}/loss', self.total_loss.average, epoch)
            wandb.log({f'{wrt_mode}/loss': self.total_loss.average},step=epoch)
            for k, v in list(self._metrics_ave().items())[:-1]:
                wandb.log({f'{wrt_mode}/{k}': v}, step=epoch)
            for i, opt_group in enumerate(self.optimizer.param_groups):
                wandb.log({f'{wrt_mode}/Learning_rate_{i}': opt_group['lr']}, step=epoch)
      

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():
            for idx, (img, gt) in enumerate(tbar):
                img = to_cuda(img)
                gt = to_cuda(gt.squeeze(1))
                
                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                
                    
                    predict = self.model(img)
                    loss = self.ce_loss(predict, gt)
            
                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(torch.softmax(predict[:,:self.config.DATASET.NUM_CLASSES], dim=1).cpu().detach().numpy()[:,1,:,:], gt.cpu().detach().numpy()).values())
                tbar.set_description(
                    'EVAL ({})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |'.format(
                        epoch, self.total_loss.average, *self._metrics_ave().values()))
               
        if self._get_rank()==0:
           
            wandb.log({f'{wrt_mode}/loss': self.total_loss.average},step=epoch)
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
        filename = os.path.join(self.checkpoint_dir,'final_checkpoint.pth')
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
        self.CCC = AverageMeter()

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
    parser.add_argument("-wm", "--wandb_mode", default="offline")
    parser.add_argument("-mt", "--model_type",default="FR_UNet")
    parser.add_argument("-st", "--scribble_type",default="scribble")
    parser.add_argument('-bs', '--batch-size', type=int,default=64,
                        help="batch size for single GPU")
    parser.add_argument('-ed', '--enable_distributed', help="training with DDP",
                        required=False, action="store_true")
    parser.add_argument('-ws', '--world_size', type=int,
                        help="process number for DDP")
    parser.add_argument('-plw', '--pseudo_loss_weight', type=float, default=1,
                        help="process number for DDP")
    parser.add_argument('-clw', '--consistency_loss_weight', type=float, default=1,
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
    # logging.basicConfig(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    #     torch.multiprocessing.set_start_method('fork', force=True)
    torch.cuda.set_device(local_rank)
    if config.DIS:
        dist.init_process_group(
            "nccl", init_method='env://', rank=local_rank, world_size=config.WORLD_SIZE)
    seed = config.SEED + local_rank
    seed_torch(seed)
    cudnn.benchmark = True

    train_loader, val_loader = build_PLC_train_loader(config)
    model = build_model(config)
   
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
                     
                      optimizer=optimizer,
                    
                      lr_scheduler=lr_scheduler,
                     

                      )
    trainer.train()


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10000"
    _, config = parse_option()

    main(config)
