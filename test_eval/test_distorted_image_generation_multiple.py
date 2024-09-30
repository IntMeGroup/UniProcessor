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

parser = argparse.ArgumentParser(description='Generating distorted images with multiple distortions')

parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Gaussian_Color_Denoising/', type=str, help='Directory for results')

args = parser.parse_args()

datasets = ['CBSD68'] # ['CBSD68', 'Kodak', 'McMaster','Urban100', 'imagenet_val_1k']

for dataset in datasets:
    inp_dir = os.path.join(args.input_dir, dataset)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))

    with torch.no_grad():
        choice1 = 15    # over bright
        choice2 = 10    # noise
        result_dir_tmp_input1 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple1_'+str(choice1)+'_input')
        result_dir_tmp_input12 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple1_'+str(choice1)+str(choice2)+'_input')
        os.makedirs(result_dir_tmp_input1, exist_ok=True)
        os.makedirs(result_dir_tmp_input12, exist_ok=True)
        PSNR_all = []
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = cv2.imread(file_)
            if dataset == 'imagenet_val_1k':
                img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            # inp_img1, label, level, text_condition = add_distortions_test_heavy(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_heavy(choice2,inp_img1)
            inp_img1, label, level, text_condition = add_distortions_test_middle(choice1,img)
            inp_img, label, level, text_condition = add_distortions_test_middle(choice2,inp_img1)
            # inp_img1, label, level, text_condition = add_distortions_test_slight(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_slight(choice2,inp_img1)
            inp_img1 = cv2.cvtColor(inp_img1, cv2.COLOR_BGR2RGB)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            inp_img1 = inp_img1.astype(np.float32) / 255.
            inp_img = inp_img.astype(np.float32) / 255.
            save_file = os.path.join(result_dir_tmp_input1, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img1, 0, 1)))
            save_file = os.path.join(result_dir_tmp_input12, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img, 0, 1)))

        choice1 = 16    # over dark
        choice2 = 20    # resize nearest
        result_dir_tmp_input1 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple2_'+str(choice1)+'_input')
        result_dir_tmp_input12 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple2_'+str(choice1)+str(choice2)+'_input')
        os.makedirs(result_dir_tmp_input1, exist_ok=True)
        os.makedirs(result_dir_tmp_input12, exist_ok=True)
        PSNR_all = []
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = cv2.imread(file_)
            if dataset == 'imagenet_val_1k':
                img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            # inp_img1, label, level, text_condition = add_distortions_test_heavy(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_heavy(choice2,inp_img1)
            inp_img1, label, level, text_condition = add_distortions_test_middle(choice1,img)
            inp_img, label, level, text_condition = add_distortions_test_middle(choice2,inp_img1)
            # inp_img1, label, level, text_condition = add_distortions_test_slight(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_slight(choice2,inp_img1)
            inp_img1 = cv2.cvtColor(inp_img1, cv2.COLOR_BGR2RGB)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            inp_img1 = inp_img1.astype(np.float32) / 255.
            inp_img = inp_img.astype(np.float32) / 255.
            save_file = os.path.join(result_dir_tmp_input1, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img1, 0, 1)))
            save_file = os.path.join(result_dir_tmp_input12, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img, 0, 1)))

        choice1 = 8    # color saturate
        choice2 = 20    # resize nearest
        result_dir_tmp_input1 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple3_'+str(choice1)+'_input')
        result_dir_tmp_input12 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple3_'+str(choice1)+str(choice2)+'_input')
        os.makedirs(result_dir_tmp_input1, exist_ok=True)
        os.makedirs(result_dir_tmp_input12, exist_ok=True)
        PSNR_all = []
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = cv2.imread(file_)
            if dataset == 'imagenet_val_1k':
                img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            # inp_img1, label, level, text_condition = add_distortions_test_heavy(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_heavy(choice2,inp_img1)
            inp_img1, label, level, text_condition = add_distortions_test_middle(choice1,img)
            inp_img, label, level, text_condition = add_distortions_test_middle(choice2,inp_img1)
            # inp_img1, label, level, text_condition = add_distortions_test_slight(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_slight(choice2,inp_img1)
            inp_img1 = cv2.cvtColor(inp_img1, cv2.COLOR_BGR2RGB)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            inp_img1 = inp_img1.astype(np.float32) / 255.
            inp_img = inp_img.astype(np.float32) / 255.
            save_file = os.path.join(result_dir_tmp_input1, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img1, 0, 1)))
            save_file = os.path.join(result_dir_tmp_input12, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img, 0, 1)))
        
        choice1 = 31    # rain streak
        choice2 = 16    # over dark
        result_dir_tmp_input1 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple4_'+str(choice1)+'_input')
        result_dir_tmp_input12 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple4_'+str(choice1)+str(choice2)+'_input')
        os.makedirs(result_dir_tmp_input1, exist_ok=True)
        os.makedirs(result_dir_tmp_input12, exist_ok=True)
        PSNR_all = []
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = cv2.imread(file_)
            if dataset == 'imagenet_val_1k':
                img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            # inp_img1, label, level, text_condition = add_distortions_test_heavy(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_heavy(choice2,inp_img1)
            inp_img1, label, level, text_condition = add_distortions_test_middle(choice1,img)
            inp_img, label, level, text_condition = add_distortions_test_middle(choice2,inp_img1)
            # inp_img1, label, level, text_condition = add_distortions_test_slight(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_slight(choice2,inp_img1)
            inp_img1 = cv2.cvtColor(inp_img1, cv2.COLOR_BGR2RGB)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            inp_img1 = inp_img1.astype(np.float32) / 255.
            inp_img = inp_img.astype(np.float32) / 255.
            save_file = os.path.join(result_dir_tmp_input1, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img1, 0, 1)))
            save_file = os.path.join(result_dir_tmp_input12, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img, 0, 1)))
        
        choice1 = 8    # color saturate
        choice2 = 12    # impulse noise
        result_dir_tmp_input1 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple5_'+str(choice1)+'_input')
        result_dir_tmp_input12 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple5_'+str(choice1)+str(choice2)+'_input')
        os.makedirs(result_dir_tmp_input1, exist_ok=True)
        os.makedirs(result_dir_tmp_input12, exist_ok=True)
        PSNR_all = []
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = cv2.imread(file_)
            if dataset == 'imagenet_val_1k':
                img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            # inp_img1, label, level, text_condition = add_distortions_test_heavy(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_heavy(choice2,inp_img1)
            inp_img1, label, level, text_condition = add_distortions_test_middle(choice1,img)
            inp_img, label, level, text_condition = add_distortions_test_middle(choice2,inp_img1)
            # inp_img1, label, level, text_condition = add_distortions_test_slight(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_slight(choice2,inp_img1)
            inp_img1 = cv2.cvtColor(inp_img1, cv2.COLOR_BGR2RGB)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            inp_img1 = inp_img1.astype(np.float32) / 255.
            inp_img = inp_img.astype(np.float32) / 255.
            save_file = os.path.join(result_dir_tmp_input1, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img1, 0, 1)))
            save_file = os.path.join(result_dir_tmp_input12, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img, 0, 1)))

        choice1 = 16    # over dark
        choice2 = 10    # gaussian noise
        result_dir_tmp_input1 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple6_'+str(choice1)+'_input')
        result_dir_tmp_input12 = os.path.join(args.result_dir, dataset, 'multiple', 'multiple6_'+str(choice1)+str(choice2)+'_input')
        os.makedirs(result_dir_tmp_input1, exist_ok=True)
        os.makedirs(result_dir_tmp_input12, exist_ok=True)
        PSNR_all = []
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = cv2.imread(file_)
            if dataset == 'imagenet_val_1k':
                img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
            # inp_img1, label, level, text_condition = add_distortions_test_heavy(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_heavy(choice2,inp_img1)
            inp_img1, label, level, text_condition = add_distortions_test_middle(choice1,img)
            inp_img, label, level, text_condition = add_distortions_test_middle(choice2,inp_img1)
            # inp_img1, label, level, text_condition = add_distortions_test_slight(choice1,img)
            # inp_img, label, level, text_condition = add_distortions_test_slight(choice2,inp_img1)
            inp_img1 = cv2.cvtColor(inp_img1, cv2.COLOR_BGR2RGB)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            inp_img1 = inp_img1.astype(np.float32) / 255.
            inp_img = inp_img.astype(np.float32) / 255.
            save_file = os.path.join(result_dir_tmp_input1, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img1, 0, 1)))
            save_file = os.path.join(result_dir_tmp_input12, os.path.split(file_)[-1])
            utils.save_img(save_file, img_as_ubyte(np.clip(inp_img, 0, 1)))