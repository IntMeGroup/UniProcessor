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

import models.models_uniprocessor_blip as models

parser = argparse.ArgumentParser(description='Testing using UniProcessor')

parser.add_argument('--arch', default='uniprocessor_blip_large', type=str, help='arch')
parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--input_dir2', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Gaussian_Color_Denoising/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/gaussian_color_denoising', type=str, help='Path to weights')

args = parser.parse_args()

import lightning.pytorch as pl
class UniProcessorModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.__dict__[args.arch]()
    def forward(self, inp_img, inp_img_blip, text_q, text_s, text_c, text_out):
        return self.model(inp_img, inp_img_blip, text_q, text_s, text_c, text_out)


from ir_data_uni.distortion_bank import *
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

factor = 8

datasets = ['CBSD68']#, 'Kodak', 'McMaster','Urban100']

levels = ['heavy', 'middle', 'slight']#, 'Kodak', 'McMaster','Urban100']

for dataset in datasets:
    inp_dir = os.path.join(args.input_dir, dataset)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))

    for level in levels:

        with torch.no_grad():
            for choice in range(1,33):
                result_dir_tmp = os.path.join(args.result_dir, dataset, level, str(choice))
                os.makedirs(result_dir_tmp, exist_ok=True)
                PSNR_all = []
                for file_ in tqdm(files):
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                    img = cv2.imread(file_)
                    inp_img, label, _, _ = add_distortions_test_heavy(choice,img)

                    img = cv2.imread(os.path.join(args.input_dir2, dataset, 'gt', os.path.split(file_)[-1]))
                    inp_img = cv2.imread(os.path.join(args.input_dir2, dataset, level, str(choice)+'_input', os.path.split(file_)[-1]))

                    inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    inp_img_blip = inp_img.copy()
                    inp_img = inp_img.astype(np.float32) / 255.
                    img = img.astype(np.float32) / 255.

                    text_q = ""
                    text_s = "This is a " + label
                    # text_c = "keep image unchanged."
                    text_c = "Remove the distortion in this image."
                    # text_c = "Remove the noise in this image."
                    # text_c = "keep the noise in this image."
                    # text_c = "Remove the blur in this image."   # seems not work
                    # text_c = "Clear up the noise in this image."
                    text_out = ""
                    cfg = OmegaConf.load('./models/lavis/lavis/configs/models/blip-diffusion/blip_diffusion_base.yaml')
                    if cfg is not None:
                        preprocess_cfg = cfg.preprocess

                        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
                    else:
                        vis_processors, txt_processors = None, None
                        logging.info(
                            f"""No default preprocess for model {name} ({model_type}).
                                This can happen if the model is not finetuned on downstream datasets,
                                or it is not intended for direct use without finetuning.
                            """
                        )
                    vis_preprocess = vis_processors
                    txt_preprocess = txt_processors
                    inp_img_blip = Image.fromarray(np.uint8(np.clip(inp_img_blip, 0, 1)*255))
                    # inp_img_blip = Image.fromarray(np.uint8(inp_img_blip))
                    inp_img_blip = vis_preprocess["eval"](inp_img_blip)
                    text_q = txt_preprocess["eval"](text_q)
                    text_s = txt_preprocess["eval"](text_s)
                    text_c = txt_preprocess["eval"](text_c)
                    text_out = txt_preprocess["eval"](text_out)

                    inp_img = torch.from_numpy(inp_img).permute(2,0,1)
                    inp_img = inp_img.unsqueeze(0).cuda()
                    inp_img, h, w = flip_pad(inp_img, 16)
                    inp_img_blip = inp_img_blip.unsqueeze(0).cuda()

                    restored = model_restoration(inp_img, inp_img_blip, text_q, text_s, text_c, text_out)
                    # restored = model_restoration(input_)
                    restored = depad(restored, h, w)
                    restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                    save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                    utils.save_img(save_file, img_as_ubyte(restored))

                    PSNR = utils.calculate_psnr(img*255., restored*255.)
                    PSNR_all.append(PSNR)

                avg_psnr = sum(PSNR_all)/len(PSNR_all)
                print('---')
                print(choice)
                print(avg_psnr)
            
