import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, to_cuda,recompone_overlap
from batchgenerators.utilities.file_and_folder_operations import *


class Inference(Trainer):
    def __init__(self,config, test_loader, model, save_dir):
        # super(Trainer, self).__init__()
        self.config = config
        self.test_loader = test_loader
        self.model = model
        self.save_path = save_dir
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE
        dir_exists(self.save_path)
        remove_files(self.save_path)
        
        cudnn.benchmark = True

    def predict(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, ncols=150)
        pres = []
        with torch.no_grad():
            
            for i, img in enumerate(tbar):
                img = to_cuda(img)
                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    pre = self.model(img)
                pre = torch.softmax(pre, dim=1)[:,1,:,:]
                pres.extend(pre)
               
        pres = torch.stack(pres, 0).cpu()

        H,W = 800,800
        num_data = 60
        pad_h = self.stride - (H - self.patch_size[0]) % self.stride
        pad_w = self.stride - (W - self.patch_size[1]) % self.stride
        new_h = H + pad_h
        new_w = W + pad_w
        pres = recompone_overlap(np.expand_dims(pres.cpu().detach().numpy(),axis=1), new_h, new_w, self.stride, self.stride)  # predictions
        predict = pres[:,0,0:H,0:W]
        predict_b = np.where(predict >= 0.5, 1, 0)

        for j in range(num_data):
            
            cv2.imwrite(self.save_path + f"/label_s{j}.png", np.uint8(predict_b[j]*255))
           
        logger.info(f"###### Inference finish ######")
       




   
        