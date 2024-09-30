from torch.utils.data import Dataset, DataLoader, Subset
from . import dataset_utils as datasets
import os

def get_datasets(args):

    if args.dataset == 'uni_air':
        training_set = datasets.PromptTrainDataset(
            denoise_dir=os.path.join(args.data_path,'Denoising','Gaussian','BW'), 
            dehaze_dir=os.path.join(args.data_path,'Dehazing','OTS_ALPHA'), 
            derain_dir=os.path.join(args.data_path,'Deraining','RainTrainL'), 
            img_size=args.input_size)
        val_set = datasets.DenoiseTestDataset(
            path=os.path.join(args.data_path,'Denoising','Gaussian','test','CBSD68'))
        val_set_rain = datasets.DerainDehazeDataset(
            path=os.path.join(args.data_path,'Deraining','test','Rain100L'), task='derain')
        val_set_haze = datasets.DerainDehazeDataset(
            path=os.path.join(args.data_path,'Dehazing','test','outdoor'), task='dehaze')

    if args.dataset == 'uni_air':
        all_datasets = {'train': training_set, 'val': val_set, 'val2': val_set_rain, 'val3': val_set_haze}
        # all_datasets = {'train': training_set, 'val': val_set}



    return all_datasets