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

# from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
# import utils
import test_eval.utils as utils
from pdb import set_trace as stx
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import models.models_uniprocessor_blip as models

parser = argparse.ArgumentParser(description='Testing using UniProcessor')

parser.add_argument('--arch', default='uniprocessor_blip_large', type=str, help='arch')
parser.add_argument('--data_path', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Gaussian_Color_Denoising/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/gaussian_color_denoising', type=str, help='Path to weights')
parser.add_argument('--input_size', default=224, type=int, help='input_size')

args = parser.parse_args()

import lightning.pytorch as pl
class UniProcessorModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.__dict__[args.arch]()
    def forward(self, inp_img, inp_img_blip, text_q, text_s, text_c, text_out):
        return self.model(inp_img, inp_img_blip, text_q, text_s, text_c, text_out)


from lavis.models import load_preprocess
from omegaconf import OmegaConf
from PIL import Image

####### Load UniProcessor #######
# checkpoint = torch.load(args.weights)
# model_restoration = models.__dict__[args.arch]()
# utils.load_checkpoint(model_restoration,checkpoint["state_dict"])
# model_restoration = UniProcessorModel().load_from_checkpoint(checkpoint_path=args.weights)
model_restoration = UniProcessorModel.load_from_checkpoint(checkpoint_path=args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration.eval()
##########################

import torch.nn.functional as F
def flip_pad(img_lq, factor):
    _, _, h, w = img_lq.size()
    H, W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    img_lq = F.pad(img_lq, (0,padw,0,padh), 'reflect')
    return img_lq, h, w
def depad(output, h, w):
    output = output[:, :, :h, :w]
    return output
def resize_up(img_lq, factor):
    _, _, h, w = img_lq.size()
    H, W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    img_lq = F.interpolate(img_lq, size=(H, W), mode='bicubic', align_corners=False)
    return img_lq, h, w
def resize_down(output, h, w):
    output = F.interpolate(output, size=(h, w), mode='bicubic', align_corners=False)
    return output
def center_crop(img_lqs, img_gts, lq_patch_size, scale=1):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    gt_patch_size = int(lq_patch_size * scale)
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). ')
    # randomly choose top and left coordinates for lq patch
    top = (h_lq - lq_patch_size) //2
    left = (w_lq - lq_patch_size) //2
    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]
    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_lqs, img_gts


def run(input, gt, inp_img_blip, text_q, text_s, text_c, text_out, write_path, img_name):
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    save_file = os.path.join(write_path, 'gt', img_name[0])
    utils.save_img(save_file, img_as_ubyte(gt.permute(0, 2, 3, 1).squeeze(0).numpy()))
    save_file = os.path.join(write_path, 'input', img_name[0])
    utils.save_img(save_file, img_as_ubyte(np.clip(input.permute(0, 2, 3, 1).squeeze(0).numpy(), 0, 1)))

    inp_img = input.cuda()
    inp_img_blip = inp_img_blip.cuda()

    if inp_img.shape[2] > 640:
        inp_img = resize_down(inp_img, 640, inp_img.shape[3]*640//inp_img.shape[2])
        gt = resize_down(gt, 640, inp_img.shape[3]*640//inp_img.shape[2])
    elif inp_img.shape[3] > 640:
        inp_img = resize_down(inp_img, inp_img.shape[2]*640//inp_img.shape[3], 640)
        gt = resize_down(gt, inp_img.shape[2]*640//inp_img.shape[3], 640)

    inp_img, h, w = flip_pad(inp_img, 16)

    restored = model_restoration(inp_img, inp_img_blip, text_q, text_s, text_c, text_out)
    # restored = model_restoration(input_)
    restored = depad(restored, h, w)
    restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    img = gt.permute(0, 2, 3, 1).squeeze(0).numpy()

    save_file = os.path.join(write_path, 'restorted', img_name[0])
    utils.save_img(save_file, img_as_ubyte(restored))

    # PSNR = utils.calculate_psnr(img*255., restored*255.)
    # SSIM = utils.calculate_ssim(img*255., restored*255.)
    PSNR = peak_signal_noise_ratio(img, restored, data_range=1)
    SSIM = structural_similarity(img, restored, data_range=1, multichannel=True)
    
    return PSNR,SSIM

from torch.utils.data import DataLoader

def test_Denoise(net, dataset, sigma=15):
    
    result_dir_tmp = os.path.join(args.result_dir, 'noise')
    os.makedirs(result_dir_tmp, exist_ok=True)
    os.makedirs(os.path.join(result_dir_tmp, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(result_dir_tmp, 'input'), exist_ok=True)
    os.makedirs(os.path.join(result_dir_tmp, 'restorted'), exist_ok=True)

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr_all = []
    ssim_all = []
    with torch.no_grad():
        for data in tqdm(testloader):
            psnr,ssim = run(data['input'], data['gt'], data['input_blip'], data['text_q'], data['text_s'], data['text_c'], data['text_out'], result_dir_tmp, data['img_name'])
            psnr_all.append(psnr)
            ssim_all.append(ssim)
        print(sum(psnr_all)/len(psnr_all))
        print(sum(ssim_all)/len(ssim_all))


def test_Derain_Dehaze(net, dataset, task="derain"):

    result_dir_tmp = os.path.join(args.result_dir, task)
    os.makedirs(result_dir_tmp, exist_ok=True)
    os.makedirs(os.path.join(result_dir_tmp, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(result_dir_tmp, 'input'), exist_ok=True)
    os.makedirs(os.path.join(result_dir_tmp, 'restorted'), exist_ok=True)

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr_all = []
    ssim_all = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            for data in tqdm(testloader):
                psnr,ssim = run(data['input'], data['gt'], data['input_blip'], data['text_q'], data['text_s'], data['text_c'], data['text_out'], result_dir_tmp, data['img_name'])
                psnr_all.append(psnr)
                ssim_all.append(ssim)
            print(sum(psnr_all)/len(psnr_all))
            print(sum(ssim_all)/len(ssim_all))


from ir_data_air import get_datasets as get_datasets

args.dataset = 'uni_air'
datasets = get_datasets.get_datasets(args)
noiseset = datasets['val']
rainset = datasets['val2']
hazeset = datasets['val3']





print('Start testing Sigma=15...')
test_Denoise(model_restoration, noiseset, sigma=15)

print('Start testing Sigma=25...')
test_Denoise(model_restoration, noiseset, sigma=25)

print('Start testing Sigma=50...')
test_Denoise(model_restoration, noiseset, sigma=50)


print('Start testing rain streak removal...')
test_Derain_Dehaze(model_restoration, rainset, task="derain")

print('Start testing SOTS...')
test_Derain_Dehaze(model_restoration, hazeset, task="dehaze")













