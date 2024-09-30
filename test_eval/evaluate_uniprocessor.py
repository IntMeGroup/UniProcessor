import os
import sys
# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)

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

from skimage import img_as_ubyte
# from basicsr.models.archs.restormer_arch import Restormer
import cv2
# import utils
import test_eval.utils as utils
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx

import lpips
alex = lpips.LPIPS(net='alex').cuda()

path_all_distorted_images = 'datasets/uni/'
path_all_restorted_images = 'results/'
method = 'UniProcessor_blip_uni'
# method = 'uni_baselines/PromptIR'
# method = 'uni_baselines/restormer'
# method = 'uni_baselines/uformer'
# method = 'uni_baselines/swinir'
# method = 'uni_baselines/MPRNet'
# method = 'uni_baselines/drunet'
# method = 'uni_baselines/uformer2'


datasets = ['CBSD68'] # ['CBSD68', 'Kodak', 'McMaster','Urban100', 'imagenet_val_1k']

levels = ['heavy', 'middle', 'slight'] # ['heavy', 'middle', 'slight']

for dataset in datasets:

    for level in levels:
        gt_path = os.path.join(path_all_distorted_images, dataset, 'gt')
        files = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.tif')))
        
        psnr_all, mae_all, ssim_all, pips_all = [], [], [], []
        for choice in range(1,33):
            restorted_path = os.path.join(path_all_restorted_images, method, dataset, level, str(choice))
            print(restorted_path)

            psnr, mae, ssim, pips = [], [], [], []
            for file_ in tqdm(files):
                

                gt_img = np.float32(utils.load_img(os.path.join(gt_path, os.path.split(file_)[-1])))/255.
                restorted_img = np.float32(utils.load_img(os.path.join(restorted_path, os.path.split(file_)[-1])))/255.

                psnr.append(utils.PSNR(gt_img, restorted_img))
                mae.append(utils.MAE(gt_img, restorted_img))
                ssim.append(utils.SSIM(gt_img, restorted_img))

                gt_img_ = torch.from_numpy(gt_img).unsqueeze(0).permute(0,3,1,2).cuda()
                restorted_img_ = torch.from_numpy(restorted_img).unsqueeze(0).permute(0,3,1,2).cuda()
                pips.append(alex(gt_img_, restorted_img_, normalize=True).item())

            print(np.mean(psnr))
            # print(np.mean(mae))
            # print(np.mean(ssim))
            # print(np.mean(pips))

            psnr_all.append(np.mean(psnr))
            mae_all.append(np.mean(mae))
            ssim_all.append(np.mean(ssim))
            pips_all.append(np.mean(pips))

        print('------')
        print(level)
        print('psnr')
        print(psnr_all)
        exec('{}_psnr_all = {}'.format(level, psnr_all))
        print('mae')
        print(mae_all)
        exec('{}_mae_all = {}'.format(level, mae_all))
        print('ssim')
        print(ssim_all)
        exec('{}_ssim_all = {}'.format(level, ssim_all))
        print('lpips')
        print(pips_all)
        exec('{}_pips_all = {}'.format(level, pips_all))
        print('------')

    print('------heavy------')
    print('psnr')
    print(heavy_psnr_all)
    print('mae')
    print(heavy_mae_all)
    print('ssim')
    print(heavy_ssim_all)
    print('lpips')
    print(heavy_pips_all)
    print('------heavy------')
    print('------middle------')
    print('psnr')
    print(middle_psnr_all)
    print('mae')
    print(middle_mae_all)
    print('ssim')
    print(middle_ssim_all)
    print('lpips')
    print(middle_pips_all)
    print('------middle------')
    print('------slight------')
    print('psnr')
    print(slight_psnr_all)
    print('mae')
    print(slight_mae_all)
    print('ssim')
    print(slight_ssim_all)
    print('lpips')
    print(slight_pips_all)
    print('------slight------')