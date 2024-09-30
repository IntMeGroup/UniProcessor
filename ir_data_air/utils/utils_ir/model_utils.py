import torch
import torch.nn as nn
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_pretrain_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    model.module.load_state_dict(checkpoint["model"], False)

def load_pretrain_checkpoint_different_dim(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["model"]
    model_state_dict = model.module.state_dict()
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('keep the weight of this layer')
                state_dict[k] = model_state_dict[k]
    model.module.load_state_dict(state_dict, strict=False)

def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    # from model import Uformer, UNet
    # from models.models_generation import Uformer as MAE_Uformer
    # from models.models_generation_v2 import Uformer as MAE_Uformer_v2
    # from models.models_generation_v5 import Uformer as MAE_Uformer_v5
    from models_swin.models_generation_CSV1 import MaskedAutoencoderSwin as CSV1
    from models_swin.models_generation_CSV2 import MaskedAutoencoderSwin as CSV2
    from models_swin.models_generation_S1 import MaskedAutoencoderSwin as S1
    from models_swin.models_generation_S2 import MaskedAutoencoderSwin as S2
    from models_swin.models_generation_SV1 import MaskedAutoencoderSwin as SV1
    from models_swin.models_generation_SV2 import MaskedAutoencoderSwin as SV2
    from models_CSformer.models_generation import MaskedAutoencoderSwin as CSF
    from models_CSformer.models_generation_v2 import MaskedAutoencoderSwin as CSF_v2
    from models_CSformer.models_generation_2stage import MaskedAutoencoderSwin as CSF_v2_2stage
    

    arch = opt.arch

    print('You choose '+arch+'...')
    if arch == 'CSV1':
        model_restoration = CSV1(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], window_size=6,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)
    elif arch == 'CSV2':
        model_restoration = CSV2(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[16,32,64,128], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[256], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], window_size=6,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)
    elif arch == 'S1':
        model_restoration = S1(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], window_size=6,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)
    elif arch == 'S2':
        model_restoration = S2(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[16,32,64,128], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[256], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], window_size=6,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)
    elif arch == 'SV1':
        model_restoration = SV1(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], window_size=6,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)
    elif arch == 'SV2':
        model_restoration = SV2(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[16,32,64,128], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[256], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], window_size=6,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)

    elif arch == 'SV1_T':
        model_restoration = SV1(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)

    elif arch == 'SV1_B':
        model_restoration = SV1(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[48,96,192,384], depths_enc=[2,2,8,8], depths_dec=[2,2,8,8], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[768], depths_mid=[4], middle_num_heads=[16], middle_mlp_ratios=[2.66], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)



    elif arch == 'CSF_T':
        model_restoration = CSF(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.,2.,2.,2.], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)

    elif arch == 'CSF_T_V2':
        model_restoration = CSF_v2(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.,2.,2.,2.], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)

    elif arch == 'CSF_L_V2':
        model_restoration = CSF_v2(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[64,128,256,512], depths_enc=[1,2,4,8], depths_dec=[1,2,4,8], num_heads=[1,2,4,8], mlp_ratios=[2.,2.,2.,2.], 
                 middle_embed_dims=[1024], depths_mid=[8], middle_num_heads=[16], middle_mlp_ratios=[2.], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)
    
    elif arch == 'CSF_v2_2stage':
        model_restoration = CSF_v2_2stage(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.,2.,2.,2.], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)

    elif arch == 'CSF_L_V3':
        model_restoration = CSF_v2(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[64,128,256,512], depths_enc=[1,2,4,8], depths_dec=[1,2,4,8], num_heads=[1,2,4,8], mlp_ratios=[2.,2.,2.,2.], 
                 middle_embed_dims=[1024], depths_mid=[4], middle_num_heads=[16], middle_mlp_ratios=[2.], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)

    elif arch == 'CSF_L_V4':
        model_restoration = CSF_v2(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[64,128,256,512], depths_enc=[2,2,8,12], depths_dec=[2,2,2,2], num_heads=[2,4,8,16], mlp_ratios=[2.,2.,2.,2.], 
                 middle_embed_dims=[1024], depths_mid=[4], middle_num_heads=[32], middle_mlp_ratios=[2.], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)
    
    elif arch == 'CSF_L_V5':
        model_restoration = CSF_v2(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[48,96,192,384], depths_enc=[4,6,6,8], depths_dec=[4,6,6,8], num_heads=[1,3,6,12], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[768], depths_mid=[8], middle_num_heads=[24], middle_mlp_ratios=[2.66], window_size=8,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 shift_flag=True, modulator=True)

        

    # if arch == 'UNet':
    #     model_restoration = UNet(dim=opt.embed_dim)
    # elif arch == 'Uformer':
    #     model_restoration = Uformer(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    # elif arch == 'Uformer_T':
    #     model_restoration = Uformer(img_size=opt.train_ps,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    # elif arch == 'Uformer_S':
    #     model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    # elif arch == 'Uformer_S_noshift':
    #     model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',modulator=True,
    #         shift_flag=False)
    # elif arch == 'Uformer_B_fastleff':
    #     model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='fastleff',
    #         depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True)  
    # elif arch == 'Uformer_B':
    #     model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
    #         depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=opt.dd_in)  

    # elif arch == 'MAE_Uformer_T':
    #     model_restoration = MAE_Uformer(img_size=opt.train_ps,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    # elif arch == 'MAE_Uformer_B':
    #     model_restoration = MAE_Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
    #         depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=opt.dd_in)  

    # elif arch == 'MAE_UformerV2_T':
    #     model_restoration = MAE_Uformer_v2(img_size=opt.train_ps,embed_dim=16,win_size=7,token_projection='linear',token_mlp='leff',modulator=True)
    # elif arch == 'MAE_UformerV2_B':
    #     model_restoration = MAE_Uformer_v2(img_size=opt.train_ps,embed_dim=32,win_size=7,token_projection='linear',token_mlp='leff',
    #         depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=opt.dd_in)  

    # elif arch == 'MAE_UformerV5_T':
    #     model_restoration = MAE_Uformer_v5(img_size=opt.train_ps,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff',
    #     depths=[1, 1, 2, 2, 2, 2, 2, 1, 1],modulator=True)

    else:
        raise Exception("Arch error!")

    return model_restoration