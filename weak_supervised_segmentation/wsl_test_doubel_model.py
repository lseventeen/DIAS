import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from loguru import logger
from data import build_test_loader
from utils.helpers import load_checkpoint
from losses.losses import *
from configs.config import get_val_config
from models.build import build_wsl_model
import numpy as np
import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm
from weak_supervised_segmentation.wsl_train_sscr import Trainer
from utils.helpers import dir_exists, to_cuda, recompone_overlap
from utils.metrics import AverageMeter, get_metrics, count_connect_component, get_color
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd


class Tester(Trainer):
    def __init__(self, config, test_loader, model1, model2,is_2d, save_dir, model_name):
        self.config = config
        self.test_loader = test_loader
        self.is_2d = is_2d
        self.model1 = model1
        self.model2 = model2
        self.model_name = model_name
        self.save_path = join("save_results", save_dir)
        self.labels_path = config.DATASET.TEST_LABEL_PATH
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE
        dir_exists(self.save_path)

        cudnn.benchmark = True

    def test(self):
        self.model1.eval()
        self.model2.eval()
        
        self._reset_metrics()
        self.VC=AverageMeter()
        gts = self.get_labels()
        
        tbar = tqdm(self.test_loader, ncols=150)
        
        pres = []
        with torch.no_grad():

            for img, _ in tbar:
                img = to_cuda(img)
                if not self.is_2d:
                    img = img.unsqueeze(1)
                   
                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    pre1 = self.model1(img)
                    pre2 = self.model2(img)

                pre1 = torch.softmax(pre1[:, :self.config.DATASET.NUM_CLASSES], dim=1)[
                    :, 1, :, :]
                pre2 = torch.softmax(pre2[:, :self.config.DATASET.NUM_CLASSES], dim=1)[
                    :, 1, :, :]
                # pre = (pre1+pre2)/2
                pre = pre1
                pres.extend(pre)

        pres = torch.stack(pres, 0).cpu()

        H, W = gts[0].shape
        num_data = len(gts)
        pad_h = self.stride - (H - self.patch_size[0]) % self.stride
        pad_w = self.stride - (W - self.patch_size[1]) % self.stride
        new_h = H + pad_h
        new_w = W + pad_w
        pres = recompone_overlap(np.expand_dims(pres.cpu().detach().numpy(
        ), axis=1), new_h, new_w, self.stride, self.stride)  # predictions

        predict = pres[:, 0, 0:H, 0:W]
        predict_b = np.where(predict >= 0.5, 1, 0)
        for j in range(num_data):

            cv2.imwrite(self.save_path + f"/gt{j}.png", np.uint8(gts[j]*255))
            cv2.imwrite(self.save_path +
                        f"/pre{j}.png", np.uint8(predict[j]*255))
            cv2.imwrite(self.save_path +
                        f"/pre_b{j}.png", np.uint8(predict_b[j]*255))
            cv2.imwrite(self.save_path +
                        f"/color_b{j}.png", get_color(predict_b[j], gts[j]))
            self._update_metrics(*get_metrics(predict[j], gts[j],run_clDice= True).values())
            self.VC.update(count_connect_component(predict_b[j], gts[j]))

        
                
            # tic = time.time()    

        mean_data = list(self._get_metrics_mean().values())
        std_data = list(self._get_metrics_std().values())
        mean_data.append(self.VC.mean)
        std_data.append(self.VC.std)
        columns = list(self._get_metrics_mean().keys())
        columns.append("VC")


        formatted_data = [f"{mean}$\pm${std}" for mean, std in zip(mean_data, std_data)]

        # 创建一个字典，用于构造DataFrame
        data_dict = {col: [val] for col, val in zip(columns, formatted_data)}

        # 创建DataFrame
        df = pd.DataFrame(data_dict)
        df.to_csv(join(self.save_path, f"{self.model_name}_result.cvs"))
        for k, v in self._get_metrics_mean().items():
            logger.info(f'{str(k):5s}: {v}')

        for k, v in self._get_metrics_std().items():
            logger.info(f'{str(k):5s}: {v}')
        
        logger.info(f'VC_mean: {self.VC.mean}')
       
        logger.info(f'VC_std: {self.VC.std}')

    def get_labels(self):
        labels = subfiles(self.labels_path, join=False, suffix='png')
        label_list = []
        for i in range(len(labels)):
            gt = cv2.imread(os.path.join(
                self.labels_path, f'label_s{i}.png'), 0)
            gt = np.array(gt/255)
            label_list.append(gt)
        return label_list


def parse_option():
    parser = argparse.ArgumentParser("DIAS_WSL_test")
    parser.add_argument('--cfg', type=str, metavar="FILE",
                        help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('-mp', '--model_path', type=str,
                        default=None, help='path to model.pth')
    args = parser.parse_args()
    config = get_val_config(args)

    return args, config


def main(config):
    save_dir = config.MODEL_PATH.split(
        '/')[-2]+"/"+config.MODEL_PATH.split('/')[-1]
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    test_loader = build_test_loader(config)

    model_checkpoint = load_checkpoint(config.MODEL_PATH, False)
    config_chk = model_checkpoint["config"]
    model_name = config_chk.MODEL.TYPE
    model1,is_2d = build_wsl_model(config_chk)
    model2,_ = build_wsl_model(config_chk)
    model1.load_state_dict({k.replace('module.', ''): v for k,
                            v in model_checkpoint['state_dict1'].items()})
    model2.load_state_dict({k.replace('module.', ''): v for k,
                            v in model_checkpoint['state_dict2'].items()})
    logger.info(f'\n{model1}\n')
    tester = Tester(config=config,
                    test_loader=test_loader,
                    model1=model1.eval().cuda(),
                    model2=model2.eval().cuda(),
                    is_2d= is_2d,
                    save_dir=save_dir,
                    model_name=model_name)
    tester.test()


if __name__ == '__main__':

    _, config = parse_option()
    main(config)
