import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from loguru import logger
from data import build_test_loader
from tester import Tester
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
from utils.helpers import dir_exists, to_cuda,recompone_overlap
from utils.metrics import get_metrics, get_metrics, count_connect_component,get_color
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd

class Tester(Trainer):
    def __init__(self,config, test_loader, model,is_2d, save_dir, model_name):
        self.config = config
        self.test_loader = test_loader
        self.model = model
        self.is_2d = is_2d
        self.model_name = model_name
        self.save_path = "save_results/" + save_dir
        self.labels_path = config.DATASET.TEST_LABEL_PATH
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE
        dir_exists(self.save_path)
       
        
        cudnn.benchmark = True

    def test(self):
        self.model.eval()
        self._reset_metrics()
        gts = self.get_labels()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        pres = []
        with torch.no_grad():
            
            for i, (img, _) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = to_cuda(img)
                if not self.is_2d:
                    img = img.unsqueeze(1)
                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    pre = self.model(img)
            
                self.batch_time.update(time.time() - tic)
                pre = torch.softmax(pre[0][:,:self.config.DATASET.NUM_CLASSES], dim=1)[:,1,:,:]
        
                pres.extend(pre)
                tbar.set_description(
                    'TEST ({}) |  |B {:.2f} D {:.2f} |'.format(i, self.batch_time.average, self.data_time.average))

        pres = torch.stack(pres, 0).cpu()

        H,W = gts[0].shape
        num_data = len(gts)
        pad_h = self.stride - (H - self.patch_size[0]) % self.stride
        pad_w = self.stride - (W - self.patch_size[1]) % self.stride
        new_h = H + pad_h
        new_w = W + pad_w
        pres = recompone_overlap(np.expand_dims(pres.cpu().detach().numpy(),axis=1), new_h, new_w, self.stride, self.stride)  # predictions

        predict = pres[:,0,0:H,0:W]
        predict_b = np.where(predict >= 0.5, 1, 0)
        for j in range(num_data):
            
            cv2.imwrite(self.save_path + f"/gt{j}.png", np.uint8(gts[j]*255))
            cv2.imwrite(self.save_path + f"/pre{j}.png", np.uint8(predict[j]*255))    
            cv2.imwrite(self.save_path + f"/pre_b{j}.png", np.uint8(predict_b[j]*255))
            cv2.imwrite(self.save_path + f"/color_b{j}.png", get_color(predict_b[j],gts[j]))
            self._metrics_update(*get_metrics(predict[j], gts[j]).values())
            self.VC.update(count_connect_component(predict_b[j], gts[j]))

            tic = time.time()
        data = list(self._metrics_ave().values())
        data.append(self.VC.average)
        columns = list(self._metrics_ave().keys())
        columns.append("VC")
        df = pd.DataFrame(data=np.array(data).reshape(1, len(columns)), index=[
                          self.save_path.split("/")[-1]], columns=columns)
        df.to_csv(join(self.save_path, "result.cvs"))
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     VC:  {self.VC.average}')
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')

    def get_labels(self):
        labels = subfiles(self.labels_path, join=False, suffix='png')
        label_list = []
        for i in range(len(labels)):
            gt = cv2.imread(os.path.join(self.labels_path, f'label_s{i}.png'), 0)
            gt = np.array(gt/255)
            label_list.append(gt)
        return label_list




def parse_option():
    parser = argparse.ArgumentParser("CVSS_test")
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

    model_checkpoint = load_checkpoint(config.MODEL_PATH, True)
    config_chk = model_checkpoint["config"]
    model_name = config_chk.MODEL.TYPE
    model,is_2d = build_wsl_model(config_chk)
    model.load_state_dict({k.replace('module.', ''): v for k,
                          v in model_checkpoint['state_dict'].items()})
    
    # average_state_dict = {}
    # model_state_dict1 = model_checkpoint['state_dict1']
    # model_state_dict2 = model_checkpoint['state_dict2']
    # for key in model_state_dict1:
    #     if key in model_state_dict2:
    #         average_state_dict[key] = (model_state_dict1[key] + model_state_dict2[key]) / 2
    # model.load_state_dict(average_state_dict)
    logger.info(f'\n{model}\n')
    tester = Tester(config=config,
                    test_loader=test_loader,
                    model=model.eval().cuda(),
                    is_2d = is_2d,
                    save_dir=save_dir,
                    model_name=model_name)
    tester.test()


if __name__ == '__main__':

    _, config = parse_option()
    main(config)
