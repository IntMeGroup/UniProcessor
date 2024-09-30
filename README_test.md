## Testing

0. If you would like to generate your own distorted images, please follow the instrctions:
```
# for single distortion generation
DATA_DIR='datasets/Image_Restoration_Datasets/Denoising/Gaussian/test/'
python test_eval/test_distorted_image_generation.py \
    --input_dir ${DATA_DIR} \
    --result_dir 'datasets/uni'
```
```
# for multiple distortion generation
DATA_DIR='/media/amax/3f3218fe-d2fb-4962-bb1f-115b03978aba/huiyu/DATA/Image_Restoration_Datasets/Denoising/Gaussian/test/'
python test_eval/test_distorted_image_generation_multiple.py \
    --input_dir ${DATA_DIR} \
    --result_dir 'datasets/uni'
```

1. test our `uni` config
```
DATA_DIR='datasets/Image_Restoration_Datasets/Denoising/Gaussian/test/'
CUDA_VISIBLE_DEVICES=0 \
python test_eval/test_uni_blip.py \
    --arch uniprocessor_blip_large \
    --input_dir ${DATA_DIR} \
    --input_dir2 'datasets/uni/' \
    --result_dir 'results/UniProcessor_blip_uni/' \
    --weights 'output/UniProcessor_blip/ckpt/last.ckpt'
```
evaluate `uni` config
```
python test_eval/evaluate_uniprocessor.py
```

2. test our `uni` config with multiple distortions
```
DATA_DIR='datasets/Image_Restoration_Datasets/Denoising/Gaussian/test/'
CUDA_VISIBLE_DEVICES=0 \
python test_eval/test_uni_blip_multidistortion.py \
    --arch uniprocessor_blip_large \
    --input_dir ${DATA_DIR} \
    --input_dir2 'datasets/uni/' \
    --result_dir 'results/UniProcessor_blip_uni_multiple/' \
    --weights 'output/UniProcessor_blip/ckpt/last.ckpt'
```
evaluate `uni` config with multiple distortions
```
python test_eval/evaluate_uniprocessor_multiple.py
```

3. test `air` config
```
DATA_DIR='datasets/Image_Restoration_Datasets/'
CUDA_VISIBLE_DEVICES=0 \
python test_eval/test_air.py \
    --arch uniprocessor_blip_large \
    --data_path ${DATA_DIR} \
    --weights 'output/UniProcessor_blip_air/ckpt/last.ckpt' \
    --result_dir 'results/UniProcessor_blip_air/'
```
