import logging
import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'lavis'))
print(sys.path)
print(dir_name)

import torch
import torch.nn as nn

from lavis.models.blip2_models.blip2_vicuna_instruct import Blip2VicunaInstruct
from lavis.models.blip_diffusion_models.blip_diffusion import BlipDiffusion

from .models_uniprocessor_attn import UniProcessor, UniProcessorEnc, UniProcessorDec

class LowlevelBlipAttn(BlipDiffusion):
    def __init__(
            self,
            vit_model="clip_L",
            qformer_num_query_token=16,
            qformer_cross_attention_freq=1,
            qformer_pretrained_path=None,
            qformer_train=False,
            sd_pretrained_model_name_or_path="lavis/pretrained_weights/blip-diffusion",
            sd_train_text_encoder=False,
            controlnet_pretrained_model_name_or_path=None,
            vae_half_precision=False,
            proj_train=False,
        ):
        super().__init__(
            vit_model=vit_model,
            qformer_num_query_token=qformer_num_query_token,
            qformer_cross_attention_freq=qformer_cross_attention_freq,
            qformer_pretrained_path=qformer_pretrained_path,
            qformer_train=qformer_train,
            sd_pretrained_model_name_or_path=sd_pretrained_model_name_or_path,
            sd_train_text_encoder=sd_train_text_encoder,
            controlnet_pretrained_model_name_or_path=controlnet_pretrained_model_name_or_path,
            vae_half_precision=vae_half_precision,
            proj_train=proj_train,
        )

    def forward(self, img, text_s, text_c):
        # latents = self.vae.encode(samples["tgt_image"].half()).latent_dist.sample()
        # latents = latents * 0.18215

        # # Sample noise that we'll add to the latents
        # noise = torch.randn_like(latents)
        # bsz = latents.shape[0]
        # # Sample a random timestep for each image
        # timesteps = torch.randint(
        #     0,
        #     self.noise_scheduler.config.num_train_timesteps,
        #     (bsz,),
        #     device=latents.device,
        # )
        # timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        ctx_embeddings = self.forward_ctx_embeddings(
            input_image=img, text_input=text_s
        )

        # Get the text embedding for conditioning
        input_ids = self.tokenizer(
            text_c,
            # padding="do_not_pad",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)
        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=[self._CTX_BEGIN_POS] * input_ids.shape[0],
        )[0]

        # # Predict the noise residual
        # noise_pred = self.unet(
        #     noisy_latents.float(), timesteps, encoder_hidden_states
        # ).sample

        # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return encoder_hidden_states


class LowlevelBlip2VicunaInstruct(Blip2VicunaInstruct):
    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            llm_model="lavis/pretrained_weights/vicuna-7b-v1.1",
            prompt="",
            max_txt_len=128,
            max_output_txt_len=256,
            apply_lemmatizer=False,
            qformer_text_input=True,
        ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
        )
    
    def forward(self, img, text_in, text_out, latents):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = img
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        bs = image.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text_in,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            text_in,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in text_out],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        # return {"loss": loss}
        return loss



class UniProcessorBlip(nn.Module):
    """ UniProcessor
    """
    def __init__(self, img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[1,1,2,2], depths_dec=[1,1,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init=''):
        super().__init__()
        self.blip_attn = LowlevelBlipAttn(
            vit_model="clip_L",
            qformer_cross_attention_freq=1,
            qformer_num_query_token=16,
            qformer_train=False,
            sd_train_text_encoder=False,
            sd_pretrained_model_name_or_path="./models/lavis/pretrained_weights/blip-diffusion",
            controlnet_pretrained_model_name_or_path=None,
            vae_half_precision=False,
        )
        self.blip_attn.load_checkpoint_from_dir("./models/lavis/pretrained_weights/blip-diffusion")

        self.blip_instruct = LowlevelBlip2VicunaInstruct(
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            llm_model="./models/lavis/pretrained_weights/vicuna-7b-v1.1",
            prompt="",
            max_txt_len=128,
            max_output_txt_len=256,
            apply_lemmatizer=False,
            qformer_text_input=True,
        )
        self.blip_instruct.load_from_pretrained(url_or_filename="./models/lavis/pretrained_weights/instruct_blip_vicuna7b_trimmed.pth")

        # self.uniprocessor = UniProcessor(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
        #          embed_dims=[24,48,96,192], depths_enc=[1,1,2,2], depths_dec=[1,1,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
        #          middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], 
        #          qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='')
        
        self.uniprocessor_enc = UniProcessorEnc(img_size=img_size, patch_size=patch_size, in_chans=in_chans, out_chans=out_chans, 
                 embed_dims=embed_dims, depths_enc=depths_enc, depths_dec=depths_dec, num_heads=num_heads, mlp_ratios=mlp_ratios, 
                 middle_embed_dims=middle_embed_dims, depths_mid=depths_mid, middle_num_heads=middle_num_heads, middle_mlp_ratios=middle_mlp_ratios, 
                 qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint, weight_init=weight_init)
        self.uniprocessor_dec = UniProcessorDec(img_size=img_size, patch_size=patch_size, in_chans=in_chans, out_chans=out_chans, 
                 embed_dims=embed_dims, depths_enc=depths_enc, depths_dec=depths_dec, num_heads=num_heads, mlp_ratios=mlp_ratios, 
                 middle_embed_dims=middle_embed_dims, depths_mid=depths_mid, middle_num_heads=middle_num_heads, middle_mlp_ratios=middle_mlp_ratios, 
                 qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint, weight_init=weight_init)


    def forward(self, img, img_blip, text_q, text_s, text_c, text_out):
        """
        img: input img
        text_q: input text question:       Please describe the quality factor of this image in detail.
        text_s: input text subject:        This is a low-quality image with noise distortion.
        text_c: input text condition:      Remove the noise in this image.
        text_out: output text for text_q:  This is a low-quality image with noise distortion, the noise level is x.
        """
        enc_feats, latent = self.uniprocessor_enc(img)
        # loss_instruct = self.blip_instruct(img, text_q, text_out, latent) # whether train Q-former, whether use latents?
        control_hidden_states = self.blip_attn(img_blip, text_s, text_c)
        img_out = self.uniprocessor_dec(enc_feats, latent, control_hidden_states)
        return img_out


def uniprocessor_blip_tiny(**kwargs):
    model = UniProcessorBlip(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[24,48,96,192], depths_enc=[1,1,2,2], depths_dec=[1,1,2,2], num_heads=[1,2,4,8], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[384], depths_mid=[2], middle_num_heads=[16], middle_mlp_ratios=[2.66], 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='', **kwargs)
    return model

def uniprocessor_blip_large(**kwargs):
    model = UniProcessorBlip(img_size=192, patch_size=16, in_chans=3, out_chans=3, 
                 embed_dims=[48,96,192,384], depths_enc=[4,6,6,8], depths_dec=[4,6,6,8], num_heads=[1,3,6,12], mlp_ratios=[2.66,2.66,2.66,2.66], 
                 middle_embed_dims=[768], depths_mid=[8], middle_num_heads=[24], middle_mlp_ratios=[2.66], 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, use_checkpoint=False, weight_init='', **kwargs)
    return model