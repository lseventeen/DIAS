import argparse
from loguru import logger
from data import build_train_single_loader, build_train_all_loader, build_val_loader, build_inference_loader
from trainer import Trainer
from utils.helpers import seed_torch
from losses import DC_and_CE_loss
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
from utils.helpers import load_checkpoint
from inference import Inference


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
    parser.add_argument("-mt", "--model_type", default="FR_UNet")
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help="batch size for single GPU")
    parser.add_argument('-ed', '--enable_distributed', help="training without DDP",
                        required=False, action="store_true")
    parser.add_argument('-nl', '--num_label',
                        help="number of label data: (1-60)", default=1)
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

    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    torch.cuda.set_device(local_rank)
    if config.DIS:
        dist.init_process_group(
            "nccl", init_method='env://', rank=local_rank, world_size=config.WORLD_SIZE)
    seed = config.SEED + local_rank
    seed_torch(seed)
    cudnn.benchmark = True

    model = build_model(config)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    if config.DIS:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    logger.info(f'\n{model}\n')
    loss = DC_and_CE_loss({}, {})
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(
        config, optimizer, config.DATASET.NUM_EACH_EPOCH//config.DATALOADER.BATCH_SIZE)
    if local_rank == 0:
        config.defrost()
        config.EXPERIMENT_ID = f"{config.WANDB.TAG}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        config.freeze()
        wandb.init(project=config.WANDB.PROJECT, name=config.EXPERIMENT_ID,
                   config=config, mode=config.WANDB.MODE)

    tag = "ite_1_teacher"
    train_label_loader = build_train_single_loader(config)
    val_loader = build_val_loader(config)
    trainer = Trainer(config=config,
                      train_loader=train_label_loader,
                      val_loader=val_loader,
                      model=model.cuda(),
                      loss=loss,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      tag=tag,
                      )
    checkpoint_dir = trainer.train()

    for i in range(1, config.ITE+1):
        save_dir = "pseudo_label" + "/"+config.EXPERIMENT_ID + "/" + tag
        test_loader = build_inference_loader(config)
        model_checkpoint = load_checkpoint(checkpoint_dir, False)
        model.load_state_dict({k.replace('module.', ''): v for k,
                               v in model_checkpoint['state_dict'].items()})
        predict = Inference(config=config,
                            test_loader=test_loader,
                            model=model.eval().cuda(),
                            save_dir=save_dir,
                            )
        predict.predict()

        tag = f"ite_{i}_student"
        train_label_loader = build_train_all_loader(config, save_dir)
        val_loader = build_val_loader(config)
        trainer = Trainer(config=config,
                          train_loader=train_label_loader,
                          val_loader=val_loader,
                          model=model.cuda(),
                          loss=loss,
                          optimizer=optimizer,
                          lr_scheduler=lr_scheduler,
                          tag=tag,
                          )
        checkpoint_dir = trainer.train()


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10000"
    _, config = parse_option()

    main(config)
