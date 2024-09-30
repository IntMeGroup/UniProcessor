import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)

import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ir_data_air import get_datasets as get_datasets
import models.models_uniprocessor_blip as models
from utils.schedulers import LinearWarmupCosineAnnealingLR
import utils.utils_ir as utils
import numpy as np
import wandb
# from options import options as opt
from train_options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger,CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.image import PeakSignalNoiseRatio

class UniProcessorModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        # self.model = models.__dict__['uniprocessor_blip_tiny']()
        self.model = models.__dict__['uniprocessor_blip_large']()
        self.loss_fn  = nn.L1Loss()
        self.opt = opt
        self.lr = opt.lr
        self.warmup_epochs = opt.warmup_epochs
        self.max_epochs = opt.epochs
        
        self.validation_step_outputs = []
        # self.validation_length = 1
        self.psnr = PeakSignalNoiseRatio()
    
    def forward(self,x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # ([clean_name, de_id], degrad_patch, clean_patch) = batch
        target = batch['gt'].cuda()
        input_ = batch['input'].cuda()
        input_blip = batch['input_blip'].cuda()
        text_q = batch['text_q']
        text_s = batch['text_s']
        text_c = batch['text_c']
        text_out = batch['text_out']
        # print(input_.shape)
        # print(text_s)

        if (self.opt.mixup_aug) and (self.current_epoch>self.warmup_epochs):
            target, input_ = utils.MixUp_AUG().aug(target, input_)

        # output = self.model(input_)
        output = self.model(input_, input_blip, text_q, text_s, text_c, text_out)

        loss = self.loss_fn(output,target)
        # Logging to TensorBoard (if installed) by default
        # if self.opt.logger == 'wblogger' or self.opt.logger == 'tblogger'
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        target = batch['gt'].cuda()
        input_ = batch['input'].cuda()
        input_blip = batch['input_blip'].cuda()
        text_q = batch['text_q']
        text_s = batch['text_s']
        text_c = batch['text_c']
        text_out = batch['text_out']

        input_, h, w = utils.flip_pad(input_, 16)
        # output = self.model(input_)
        output = self.model(input_, input_blip, text_q, text_s, text_c, text_out)
        output = output.clamp(0.0, 1.0)
        output = utils.depad(output, h, w)
        self.psnr.update(output, target)

        performance = utils.batch_PSNR(output, target, False)
        self.validation_step_outputs.append(performance)
        self.log("val psnr step", performance, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return performance

    def on_validation_epoch_end(self):
        psnr_all = self.psnr.compute()
        print(f"all psnr: {psnr_all}")
        self.log("psnr_all", psnr_all, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.psnr.reset()

        all_performance = torch.stack(self.validation_step_outputs)
        avg_performance = all_performance.mean()
        print('avg performance:')
        print(avg_performance)
        self.log("val psnr epoch", avg_performance, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.max_epochs)

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    # if opt.wblogger is not None:
    #     logger  = WandbLogger(save_dir=opt.output_path, project=opt.wblogger, name="UniProcessor-Train")
    # else:
    #     logger = TensorBoardLogger(save_dir = "logs/")
    if opt.logger == 'wblogger':
        logger = WandbLogger(save_dir=opt.output_dir, project=opt.wblogger, name="UniProcessor-Train")
    elif opt.logger == 'tblogger':
        logger = TensorBoardLogger(save_dir=opt.output_dir)
    else:
        logger = CSVLogger(save_dir=opt.output_dir, name=opt.log_name)

    datasets = get_datasets.get_datasets(opt)
    trainset = datasets['train']
    print(len(trainset))
    valset = datasets['val']
    print(len(valset))
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, save_last=True, every_n_epochs=20, save_top_k=-1)
    trainloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    valloader = DataLoader(dataset=valset, batch_size=1, pin_memory=False, shuffle=False, 
                             drop_last=False, num_workers=1)
    
    model = UniProcessorModel(opt)
    
    # trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, strategy="ddp_find_unused_parameters_true", logger=logger, callbacks=[checkpoint_callback])
    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, precision=16, strategy="ddp_find_unused_parameters_true", logger=logger, callbacks=[checkpoint_callback])
    if opt.resume:
        trainer.fit(model=model, ckpt_path=opt.pretrain_weights, train_dataloaders=trainloader, val_dataloaders=valloader)
    else:
        trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()