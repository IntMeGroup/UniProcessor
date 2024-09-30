import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.models import load_preprocess
from omegaconf import OmegaConf
import models.models_uniprocessor_blip_test as models


if __name__ == "__main__":

    model = models.UniProcessorBlip()
    model = model.eval()
    model = model.to("cuda")


    # ###############################################################
    # test blip diffusion
    # ###############################################################

    torch.cuda.is_available()
    cfg = OmegaConf.load('models/lavis/lavis/configs/models/blip-diffusion/blip_diffusion_base.yaml')
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

    cond_subject = "dog"
    src_subject = "cat"
    tgt_subject = "dog"

    text_prompt = "sit on a chair, oil painting"

    cond_subject = txt_preprocess["eval"](cond_subject)
    src_subject = txt_preprocess["eval"](src_subject)
    tgt_subject = txt_preprocess["eval"](tgt_subject)
    text_prompt = [txt_preprocess["eval"](text_prompt)]

    cond_image = Image.open("models/lavis/projects/blip-diffusion/images/dog.png").convert("RGB")
    # display(cond_image.resize((256, 256)))
    cond_image.save('test_cond.png')

    cond_image = vis_preprocess["eval"](cond_image).unsqueeze(0).cuda()

    samples = {
        "cond_images": cond_image,
        "cond_subject": cond_subject,
        "src_subject": src_subject,
        "tgt_subject": tgt_subject,
        "prompt": text_prompt,
    }

    iter_seed = 88991
    guidance_scale = 7.5
    num_inference_steps = 50
    negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
    
    output = model.blip_attn.generate_then_edit(
        samples,
        seed=iter_seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
    )

    print("=" * 30)
    print("Before editing:")
    # display(output[0])
    output[0].save('test_before.png')

    print("After editing:")
    # display(output[1])
    output[1].save('test_after.png')


    # ###############################################################
    # test instruct blip
    # ###############################################################

    cfg = OmegaConf.load('models/lavis/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml')
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

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # load sample image
    raw_image = Image.open("models/lavis/docs/_static/Confusing-Pictures.jpg").convert("RGB")
    # display(raw_image.resize((596, 437)))
    raw_image.save('test_raw.png')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


    out_text = model.blip_instruct.generate({"image": image, "prompt": "What is unusual about this image?"})
    print('---')
    print(out_text)
    out_text = model.blip_instruct.generate({"image": image, "prompt": "Write a short description for the image."})
    print('---')
    print(out_text)
    out_text = model.blip_instruct.generate({"image": image, "prompt": "Write a detailed description."})
    print('---')
    print(out_text)
    out_text = model.blip_instruct.generate({"image": image, "prompt":"Describe the image in details."}, use_nucleus_sampling=True, top_p=0.9, temperature=1)
    print('---')
    print(out_text)
    out_text = model.blip_instruct.generate({"image": image, "prompt":"Is there any distortion in this image?"}, use_nucleus_sampling=True, top_p=0.9, temperature=1)
    print('---')
    print(out_text)
    out_text = model.blip_instruct.generate({"image": image, "prompt":"Describe the quality of this image."}, use_nucleus_sampling=True, top_p=0.9, temperature=1)
    print('---')
    print(out_text)