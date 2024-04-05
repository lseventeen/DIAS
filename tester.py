import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files,to_cuda,recompone_overlap
from utils.metrics import get_metrics, get_metrics, count_connect_component,get_color,AverageMeter
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd

class Tester(Trainer):
    def __init__(self,config, test_loader, model, save_dir, is_2d,  model_name):
        # super(Trainer, self).__init__()
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
        remove_files(self.save_path)
        
        cudnn.benchmark = True

    def test(self):
        
        self.model.eval()
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
                # with torch.cuda.amp.autocast(enabled=False):
                    pre = self.model(img)
            
               
                pre = torch.softmax(pre, dim=1)[:,1,:,:]
        
                pres.extend(pre)
        

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
        # df = pd.DataFrame(data=np.array(data).reshape(1, len(columns)), index=[self.model_name], columns = columns)
       
        # logger.info(f"###### TEST EVALUATION ######")
        # logger.info(f'test time:  {self.batch_time.average}')
        # logger.info(f'     VC:  {self.VC.average}')
  
        

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
            gt = cv2.imread(os.path.join(self.labels_path, f'label_s{i}.png'), 0)
            gt = np.array(gt/255)
            label_list.append(gt)
        return label_list
    



   
        