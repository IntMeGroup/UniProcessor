from torch.utils.data import Dataset, DataLoader, Subset
from . import datasets as datasets
import os

def get_datasets(args):

    if args.dataset == 'uni_DFWB':
        training_set = datasets.Dataset_UniDistortion(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','train','DFWB'), 
            img_size=args.input_size, is_train=True)
        val_set = datasets.Dataset_UniDistortion(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','test','CBSD68'), 
            img_size=args.input_size, is_train=False)
        val_set2 = datasets.Dataset_UniDistortion(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','test','Kodak'), 
            img_size=args.input_size, is_train=False)
        val_set3 = datasets.Dataset_UniDistortion(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','test','McMaster'), 
            img_size=args.input_size, is_train=False)
        val_set4 = datasets.Dataset_UniDistortion(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','test','Urban100'), 
            img_size=args.input_size, is_train=False)

    if args.dataset == 'uni_blip_DFWB':
        training_set = datasets.Dataset_UniDistortion_Blip(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','train','DFWB'), 
            img_size=args.input_size, is_train=True)
        val_set = datasets.Dataset_UniDistortion_Blip(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','test','CBSD68'), 
            img_size=args.input_size, is_train=False)
        val_set2 = datasets.Dataset_UniDistortion_Blip(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','test','Kodak'), 
            img_size=args.input_size, is_train=False)
        val_set3 = datasets.Dataset_UniDistortion_Blip(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','test','McMaster'), 
            img_size=args.input_size, is_train=False)
        val_set4 = datasets.Dataset_UniDistortion_Blip(
            tar_dir=os.path.join(args.data_path,'Denoising','Gaussian','test','Urban100'), 
            img_size=args.input_size, is_train=False)
    
    
    if args.dataset == 'uni_DFWB':
        all_datasets = {'train': training_set, 'val': val_set, 'val2': val_set2, 'val3': val_set3, 'val4': val_set4}
    
    
    if args.dataset == 'uni_blip_DFWB':
        all_datasets = {'train': training_set, 'val': val_set, 'val2': val_set2, 'val3': val_set3, 'val4': val_set4}



    return all_datasets