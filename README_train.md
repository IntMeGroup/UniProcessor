## Training

1. train model `models/model_uniprocessor_blip.py` on `ir_data_uni` config, we use 4 A6000 or 4 A100 for training.
```
DATA_DIR='datasets/Image_Restoration_Datasets/'
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train/train_uni_blip.py \
    --arch uniprocessor_large \
    --dataset uni_blip_DFWB \
    --data_path ${DATA_DIR} \
    --input_size 224 \
    --num_gpus 4 \
    --warmup_epochs 3 --epochs 202 --batch_size 5 \
    --output_dir 'output/UniProcessor_blip/logs' \
    --ckpt_dir 'output/UniProcessor_blip/ckpt' \
    --resume --pretrain_weights 'output/UniProcessor_blip/ckpt/last.ckpt'
```

2. train model `models/model_uniprocessor_blip.py` on `ir_data_air` config
```
DATA_DIR='datasets/Image_Restoration_Datasets/'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python train/train_air_blip.py \
    --dataset uni_air \
    --data_path ${DATA_DIR} \
    --input_size 224 \
    --num_gpus 6 \
    --warmup_epochs 3 --epochs 400 --batch_size 5 \
    --output_dir 'output/UniProcessor_blip_air/logs' \
    --ckpt_dir 'output/UniProcessor_blip_air/ckpt' \
    --resume --pretrain_weights 'output/UniProcessor_blip_air/ckpt/last.ckpt'
```