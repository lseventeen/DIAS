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

    model_checkpoint = load_checkpoint(config.MODEL_PATH, False)
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
