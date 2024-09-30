import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
from pdb import set_trace as stx

from . img_utils import is_image_file, load_img, padding, paired_random_crop, center_crop, random_augmentation, img2tensor

import cv2
import random

from .distortion_bank import *

from lavis.models import load_preprocess
from omegaconf import OmegaConf
from PIL import Image
from skimage import img_as_ubyte

class Dataset_UniDistortion(Dataset):
    def __init__(self, tar_dir, img_size, is_train=True, normalize=False, sigma_type='constant', sigma=15, val_ddp_expand=None):
        tar_files = sorted(os.listdir(tar_dir))

        self.tar_filenames = [os.path.join(tar_dir, x) for x in tar_files if is_image_file(x)]
        if val_ddp_expand is not None:
            self.tar_filenames = self.tar_filenames * val_ddp_expand

        self.is_train = is_train
        
        self.ps = img_size

        self.normalize = normalize
        self.mean = [0.5, 0.5, 0.5]
        self.std  = [0.5, 0.5, 0.5]

        self.sigma_type = sigma_type
        self.sigma = sigma

    def __len__(self):
        return len(self.tar_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ps = self.ps

        tar_path = self.tar_filenames[idx]
        img_name = os.path.basename(tar_path)

        tar_img = load_img(tar_path, float32=False)
        # tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
        inp_img = tar_img.copy()

        # augmentation for training
        if self.is_train:
            # padding
            inp_img, tar_img = padding(inp_img, tar_img, ps)
            # random crop
            inp_img, tar_img = paired_random_crop(inp_img, tar_img, ps)
            # flip, rotation augmentations
            inp_img, tar_img = random_augmentation(inp_img, tar_img)

            choice = random.randint(1, 34)   # imjpeg: quality_factor (8,96)
            inp_img, label, level,_ = add_distortions_random(choice, inp_img)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
            inp_img_blip = inp_img.copy()
            inp_img = inp_img.astype(np.float32) / 255.
            tar_img = tar_img.astype(np.float32) / 255.

            # HWC to CHW, numpy to tensor
            inp_img, tar_img = img2tensor([inp_img, tar_img],
                                        bgr2rgb=False,
                                        float32=True)
            

        else:            
            np.random.seed(seed=0)
            inp_img = inp_img.astype(np.float32) / 255.
            inp_img += np.random.normal(0, self.sigma/255.0, inp_img.shape)
            inp_img = inp_img * 255.
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
            inp_img_blip = inp_img.copy()
            inp_img = inp_img.astype(np.float32) / 255.
            tar_img = tar_img.astype(np.float32) / 255.
            inp_img, tar_img = img2tensor([inp_img, tar_img],
                            bgr2rgb=False,
                            float32=True)
        # normalize
        if self.normalize:
            inp_img = TF.normalize(inp_img, mean=self.mean, std=self.std, inplace=True)
            tar_img = TF.normalize(tar_img, mean=self.mean, std=self.std, inplace=True)

        data = {
            'input': inp_img,
            'gt': tar_img,
            'img_name': img_name
        }

        return data




class Dataset_UniDistortion_Blip(Dataset):
    def __init__(self, tar_dir, img_size, is_train=True, normalize=False, sigma_type='constant', sigma=25, val_ddp_expand=None):
        tar_files = sorted(os.listdir(tar_dir))

        self.tar_filenames = [os.path.join(tar_dir, x) for x in tar_files if is_image_file(x)]
        if val_ddp_expand is not None:
            self.tar_filenames = self.tar_filenames * val_ddp_expand

        self.is_train = is_train
        
        self.ps = img_size

        self.normalize = normalize
        self.mean = [0.5, 0.5, 0.5]
        self.std  = [0.5, 0.5, 0.5]

        self.sigma_type = sigma_type
        self.sigma = sigma

    def __len__(self):
        return len(self.tar_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ps = self.ps

        tar_path = self.tar_filenames[idx]
        img_name = os.path.basename(tar_path)

        tar_img = load_img(tar_path, float32=False)
        # tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
        inp_img = tar_img.copy()

        # augmentation for training
        if self.is_train:
            # padding
            inp_img, tar_img = padding(inp_img, tar_img, ps)
            # random crop
            inp_img, tar_img = paired_random_crop(inp_img, tar_img, ps)
            # flip, rotation augmentations
            inp_img, tar_img = random_augmentation(inp_img, tar_img)

            choice = random.randint(1, 34)   # imjpeg: quality_factor (8,96)
            inp_img, label, level,_ = add_distortions_random(choice, inp_img)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
            inp_img_blip = inp_img.copy()
            inp_img = inp_img.astype(np.float32) / 255.
            tar_img = tar_img.astype(np.float32) / 255.
            # cv2.imwrite("train_inp.png",cv2.cvtColor(img_as_ubyte(np.clip(inp_img, 0, 1)), cv2.COLOR_RGB2BGR))
            # cv2.imwrite("train_tar.png",cv2.cvtColor(img_as_ubyte(np.clip(tar_img, 0, 1)), cv2.COLOR_RGB2BGR))

            # HWC to CHW, numpy to tensor
            inp_img, tar_img = img2tensor([inp_img, tar_img],
                                        bgr2rgb=False,
                                        float32=True)

            text_q = ""
            text_s = "This is a " + label
            condition = random.randint(0, 10)
            if condition == 0:
                text_c = "keep image unchanged."
                tar_img = inp_img.clone()
            else:
                text_c = "Remove the distortion in this image."
            

        else:            
            np.random.seed(seed=0)
            inp_img = inp_img.astype(np.float32) / 255.
            inp_img += np.random.normal(0, self.sigma/255.0, inp_img.shape)
            inp_img = inp_img * 255.
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
            inp_img_blip = inp_img.copy()
            inp_img = inp_img.astype(np.float32) / 255.
            tar_img = tar_img.astype(np.float32) / 255.
            # cv2.imwrite("test_inp.png",cv2.cvtColor(img_as_ubyte(np.clip(inp_img, 0, 1)), cv2.COLOR_RGB2BGR))
            # cv2.imwrite("test_tar.png",cv2.cvtColor(img_as_ubyte(np.clip(tar_img, 0, 1)), cv2.COLOR_RGB2BGR))

            inp_img, tar_img = img2tensor([inp_img, tar_img],
                            bgr2rgb=False,
                            float32=True)
            label = "noise"

            text_q = ""
            text_s = "This is a low-quality image with " + label + " distortion."
            text_c = "Remove the distortion in this image."

        # normalize
        if self.normalize:
            inp_img = TF.normalize(inp_img, mean=self.mean, std=self.std, inplace=True)
            tar_img = TF.normalize(tar_img, mean=self.mean, std=self.std, inplace=True)

        # text_q = ""
        # text_s = "This is " + label
        # condition = random.randint(0, 10)
        # if condition == 0:
        #     text_c = "keep image unchanged."
        #     tar_img = inp_img.clone()
        # else:
        #     text_c = "Remove the distortion in this image."
        # print(text_s)
        # print(text_c)
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
        # inp_img_blip = Image.fromarray(np.uint8(np.clip(inp_img_blip, 0, 1)*255))
        inp_img_blip = Image.fromarray(np.uint8(inp_img_blip))
        inp_img_blip = vis_preprocess["eval"](inp_img_blip)
        text_q = txt_preprocess["eval"](text_q)
        text_s = txt_preprocess["eval"](text_s)
        text_c = txt_preprocess["eval"](text_c)
        text_out = txt_preprocess["eval"](text_out)

        data = {
            'input': inp_img,
            'gt': tar_img,
            'img_name': img_name,
            'input_blip': inp_img_blip,
            'text_q': text_q,
            'text_s': text_s,
            'text_c': text_c,
            'text_out': text_out
        }

        return data
