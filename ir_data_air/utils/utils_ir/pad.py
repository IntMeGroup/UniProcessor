import torch.nn.functional as F

def flip_pad(img_lq, factor):
    _, _, h, w = img_lq.size()
    H, W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    img_lq = F.pad(img_lq, (0,padw,0,padh), 'reflect')
    return img_lq, h, w

def depad(output, h, w):
    output = output[:, :, :h, :w]
    return output

def resize_up(img_lq, factor):
    _, _, h, w = img_lq.size()
    H, W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    img_lq = F.interpolate(img_lq, size=(H, W), mode='bicubic', align_corners=False)
    return img_lq, h, w

def resize_down(output, h, w):
    output = F.interpolate(output, size=(h, w), mode='bicubic', align_corners=False)
    return output