import torch
import torch.nn as nn
import math

from .modules_uniprocessor import *

class UniProcessor(nn.Module):
    """ UniProcessor
    """
    def __init__(self, img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[48,96,192,384], depths_enc=[2,2,2,2], depths_dec=[2,2,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[768], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66],
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 norm_layer=LayerNorm2d):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        # stochastic depth
        dpr_enc = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_enc[:]))]
        dpr_mid = [drop_path_rate]*depths_mid[0]
        dpr_dec = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_dec[:]))]

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dims[0], kernel_size=3, stride=1)
        self.output_proj = OutputProj(in_channel=embed_dims[0], out_channel=out_chans, kernel_size=3, stride=1)
        
        # Encoder
        self.encoder_level1 = BasicLayerConvFormer(dim=embed_dims[0], depth=depths_enc[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], 
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_enc[sum(depths_enc[:0]):sum(depths_enc[:1])], norm_layer=norm_layer, 
                                             use_checkpoint=use_checkpoint)
        self.down1_2 = Downsample(embed_dims[0], embed_dims[1])
        self.encoder_level2 = BasicLayerConvFormer(dim=embed_dims[1], depth=depths_enc[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], 
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_enc[sum(depths_enc[:1]):sum(depths_enc[:2])], norm_layer=norm_layer, 
                                             use_checkpoint=use_checkpoint)
        self.down2_3 = Downsample(embed_dims[1], embed_dims[2])
        self.encoder_level3 = BasicLayerConvFormer(dim=embed_dims[2], depth=depths_enc[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], 
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_enc[sum(depths_enc[:2]):sum(depths_enc[:3])], norm_layer=norm_layer, 
                                             use_checkpoint=use_checkpoint)
        self.down3_4 = Downsample(embed_dims[2], embed_dims[3])
        self.encoder_level4 = BasicLayerConvFormer(dim=embed_dims[3], depth=depths_enc[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], 
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_enc[sum(depths_enc[:3]):sum(depths_enc[:4])], norm_layer=norm_layer, 
                                             use_checkpoint=use_checkpoint)
        self.down4_5 = Downsample(embed_dims[3], middle_embed_dims[0])

        # middle 1
        self.middle = BasicLayerTransFormer(dim=middle_embed_dims[0], depth=depths_mid[0], num_heads=middle_num_heads[0], mlp_ratio=middle_mlp_ratios[0], 
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_mid, norm_layer=norm_layer, 
                                             use_checkpoint=use_checkpoint)

        # Decoder
        self.up5_4 = Upsample(middle_embed_dims[0], embed_dims[3])
        self.reduce_chan_level4 = nn.Conv2d(middle_embed_dims[0], embed_dims[3], kernel_size=1, bias=False)
        self.decoder_level4 = BasicLayerConvFormer(dim=embed_dims[3], depth=depths_dec[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], 
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_enc[sum(depths_dec[:3]):sum(depths_dec[:4])], norm_layer=norm_layer, 
                                             use_checkpoint=use_checkpoint)
        self.up4_3 = Upsample(embed_dims[3], embed_dims[2])
        self.reduce_chan_level3 = nn.Conv2d(embed_dims[3], embed_dims[2], kernel_size=1, bias=False)
        self.decoder_level3 = BasicLayerConvFormer(dim=embed_dims[2], depth=depths_dec[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], 
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_enc[sum(depths_dec[:2]):sum(depths_dec[:3])], norm_layer=norm_layer, 
                                             use_checkpoint=use_checkpoint)
        self.up3_2 = Upsample(embed_dims[2], embed_dims[1])
        self.reduce_chan_level2 = nn.Conv2d(embed_dims[2], embed_dims[1], kernel_size=1, bias=False)
        self.decoder_level2 = BasicLayerConvFormer(dim=embed_dims[1], depth=depths_dec[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], 
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_enc[sum(depths_dec[:1]):sum(depths_dec[:2])], norm_layer=norm_layer, 
                                             use_checkpoint=use_checkpoint)
        self.up2_1 = Upsample(embed_dims[1], embed_dims[0])
        self.reduce_chan_level1 = nn.Conv2d(embed_dims[1], embed_dims[0], kernel_size=1, bias=False)
        self.decoder_level1 = BasicLayerConvFormer(dim=embed_dims[0], depth=depths_dec[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], 
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_enc[sum(depths_dec[:0]):sum(depths_dec[:1])], norm_layer=norm_layer, 
                                             use_checkpoint=use_checkpoint)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):

        # Input Projection
        x_embed = self.input_proj(x)
        
        #Encoder
        out_enc_level1 = self.encoder_level1(x_embed)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4 = self.encoder_level4(inp_enc_level4)

        # Middle
        latent = self.down4_5(out_enc_level4)
        latent = self.middle(latent)

        # Decoder
        inp_dec_level4 = self.up5_4(latent)
        inp_dec_level4 = torch.cat([inp_dec_level4, out_enc_level4], 1)
        inp_dec_level4 = self.reduce_chan_level4(inp_dec_level4)
        out_dec_level4 = self.decoder_level4(inp_dec_level4)

        inp_dec_level3 = self.up4_3(out_dec_level4)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # Output Projection
        y = self.output_proj(out_dec_level1)

        return x + y if self.in_chans == self.out_chans else y



def uniprocessor_tiny(**kwargs):
    model = UniProcessor(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[1,1,2,2], depths_dec=[1,1,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 norm_layer=LayerNorm2d, **kwargs)
    return model


def uniprocessor_large(**kwargs):
    model = UniProcessor(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[48,96,192,384], depths_enc=[4,6,6,8], depths_dec=[4,6,6,8], num_heads=[1,3,6,12], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[768], depths_mid=[8], middle_num_heads=[24], middle_mlp_ratios=[2.66], 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='',
                 norm_layer=LayerNorm2d, **kwargs)
    return model