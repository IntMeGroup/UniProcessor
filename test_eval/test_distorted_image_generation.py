import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)

import torch.nn as nn
import torch
import torch.nn.functional as F

from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import test_eval.utils as utils
from pdb import set_trace as stx

from ir_data_uni.distortion_bank import *
from PIL import Image

parser = argparse.ArgumentParser(description='Generating distorted images')

parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Gaussian_Color_Denoising/', type=str, help='Directory for results')

args = parser.parse_args()

datasets = ['CBSD68'] # ['CBSD68', 'Kodak', 'McMaster','Urban100', 'imagenet_val_1k']
levels = ['heavy', 'middle', 'slight']#, 'Kodak', 'McMaster','Urban100']

for dataset in datasets:
    inp_dir = os.path.join(args.input_dir, dataset)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))

    with torch.no_grad():
        for choice in range(1,33):
            for level in levels:
                # result_dir_tmp = os.path.join(args.result_dir, dataset, str(choice))
                result_dir_tmp_gt = os.path.join(args.result_dir, dataset, 'gt')
                result_dir_tmp_input = os.path.join(args.result_dir, dataset, level, str(choice)+'_input')
                # os.makedirs(result_dir_tmp, exist_ok=True)
                os.makedirs(result_dir_tmp_gt, exist_ok=True)
                os.makedirs(result_dir_tmp_input, exist_ok=True)
                for file_ in tqdm(files):
                    img = cv2.imread(file_)
                    if dataset == 'imagenet_val_1k':
                        img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
                    if level == 'heavy':
                        inp_img, label, level, text_condition = add_distortions_test_heavy(choice,img)
                    elif level == 'middle':
                        inp_img, label, level, text_condition = add_distortions_test_middle(choice,img)
                    elif level == 'slight':
                        inp_img, label, level, text_condition = add_distortions_test_slight(choice,img)
                    else:
                        inp_img, label, level, text_condition = add_distortions_random(choice,img)

                    inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    inp_img = inp_img.astype(np.float32) / 255.
                    img = img.astype(np.float32) / 255.
                    save_file = os.path.join(result_dir_tmp_gt, os.path.split(file_)[-1])
                    utils.save_img(save_file, img_as_ubyte(img))
                    save_file = os.path.join(result_dir_tmp_input, os.path.split(file_)[-1])
                    utils.save_img(save_file, img_as_ubyte(np.clip(inp_img, 0, 1)))
