# borrows heavily from: https://github.com/avinabsaha/ReIQA/blob/main/datasets/iqa_distortions.py

import numpy as np
from PIL import Image, ImageFilter
import skimage.morphology
from scipy import ndimage
import random
from skimage import color,filters,io
from sklearn.preprocessing import normalize
import io
from scipy.interpolate import UnivariateSpline
import PIL
from scipy import interpolate
import skimage
import cv2

import warnings
warnings.filterwarnings("ignore")

import os


# ###############################################################
# functions
# ###############################################################

def curvefit (xx, coef):
    x = np.array([0,0.5,1])
    y = np.array([0,coef,1])
    tck = UnivariateSpline(x, y, k=2)
    return np.clip(tck(xx),0,1)

def mapmm(e):
    mina = 0.0
    maxa = 1.0
    minx = np.min(e)
    maxx = np.max(e)
    if minx<maxx : 
        e = (e-minx)/(maxx-minx)*(maxa-mina)+mina
    return e

def imwarpmap(im, shifts):
    sy, sx = shifts[:,:,0], shifts[:,:,1] 
    ## create mesh-grid for image shape
    [xx, yy] = np.meshgrid(range(0,shifts.shape[1]), range(0,shifts.shape[0]))
    ## check whether grey image or RGB
    shape = im.shape
    im_out = im
    if len(shape)>2:
        ch = shape[-1]
    else:
        ch = 1
    ## iterate function over each channel
    for i in range(ch):
        im_out[:,:,i] = ndimage.map_coordinates(im[:,:,i], [(yy-sy).ravel(), (xx-sx).ravel()], order = 3, mode = 'reflect').reshape(im.shape[:2])
    ## clip image between 0-255
    im_out = np.clip(im_out, 0, 255)
    return im_out

# for mosaic generation
def masks_CFA_Bayer(shape):
    pattern = "RGGB"
    channels = dict((channel, np.zeros(shape)) for channel in "RGB")
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1
    return tuple(channels[c].astype(bool) for c in "RGB")
def mosaic_CFA_Bayer(RGB):
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2])
    mask = np.concatenate(
        (R_m[..., np.newaxis], G_m[..., np.newaxis], B_m[..., np.newaxis]), axis=-1
    )
    # mask = tstack((R_m, G_m, B_m))
    mosaic = np.multiply(mask, RGB)  # mask*RGB
    CFA = mosaic.sum(2).astype(np.uint8)
    CFA4 = np.zeros((RGB.shape[0] // 2, RGB.shape[1] // 2, 4), dtype=np.uint8)
    CFA4[:, :, 0] = CFA[0::2, 0::2]
    CFA4[:, :, 1] = CFA[0::2, 1::2]
    CFA4[:, :, 2] = CFA[1::2, 0::2]
    CFA4[:, :, 3] = CFA[1::2, 1::2]
    return CFA, CFA4, mosaic, mask
def flip_pad(im, factor):
    # _, _, h, w = img_lq.size()
    h, w, _ = im.shape
    H, W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    im = cv2.copyMakeBorder(im, 0, padh, 0, padw, cv2.BORDER_REFLECT)
    return im, h, w
def depad(output, h, w):
    output = output[:h, :w, :]
    return output

# for painting generation
from enum import Enum
class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'
def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]
def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]

# for rain streak generation
def get_noise(img, value=10):
    # value controls the number of rain streaks
    noise = np.random.uniform(0, 256, img.shape[0:2])
    # control noise level and keep the largest noise
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0
    # first blurring for nosie
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, k)
    return noise
def rain_blur(noise, length=10, angle=0, w=1):
    '''
    motion blur for noise to simulate rain streaks
    noise: input noise image, shape = img.shape[0:2]
    length: length of rain streaks
    angle: angle of rain streaks
    w: width of rain streaks
    '''
    # since the diagonal array comes with a 45-degree inclination, it is positive counterclockwise, so an error of -45 degrees is added to ensure that it is positive at the beginning
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # generate a diagonal matrix
    k = cv2.warpAffine(dig, trans, (length, length))  # generate blur kernel
    k = cv2.GaussianBlur(k, (w, w), 0)  # gaussian blurs this rotated diagonal kernel so that the rain has width
    # k = k / length
    blurred = cv2.filter2D(noise, -1, k)  # filter
    # convert to 0-255
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
def alpha_rain(rain, img, beta=0.8):
    # beta controls rain streak weight
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel
    rain_result = img.copy()  # copy a mask
    rain = np.array(rain, dtype=np.float32)  # 32 bits are required to prevent the array from crossing the boundary
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    return rain_result
import sys
def crystallize(im, cnt):
    # Make output image same size
    res = np.zeros_like(im)
    h, w = im.shape[:2]
    # Generate some randomly placed crystal centres
    nx = np.random.randint(0,w,cnt,dtype=np.uint16)
    ny = np.random.randint(0,h,cnt,dtype=np.uint16)
    # Pick up colours at those locations from source image
    sRGB = []
    for i in range(cnt):
        sRGB.append(im[ny[i],nx[i]])
    # Iterate over image
    for y in range(h):
        for x in range(w):
            # Find nearest crystal centre...
            dmin = sys.float_info.max
            for i in range(cnt):
                d = (y-ny[i])*(y-ny[i]) + (x-nx[i])*(x-nx[i])
                if d < dmin:
                    dmin = d
                    j = i
            # ... and copy colour of original image to result
            res[y,x] = sRGB[j]
    return res

def snow_blur(noise, length=10, angle=0, w=1):
    '''
    motion blur for noise to simulate rain streaks
    noise: input noise image, shape = img.shape[0:2]
    length: length of rain streaks
    angle: angle of rain streaks
    w: width of rain streaks
    '''
    # since the diagonal array comes with a 45-degree inclination, it is positive counterclockwise, so an error of -45 degrees is added to ensure that it is positive at the beginning
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # generate a diagonal matrix
    k = cv2.warpAffine(dig, trans, (length, length))  # generate blur kernel
    k = cv2.GaussianBlur(k, (w, w), 0)  # gaussian blurs this rotated diagonal kernel so that the rain has width
    # k = k / length
    blurred = cv2.filter2D(noise, -1, k)  # filter
    # convert to 0-255
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
def alpha_snow(rain, img, beta=0.8):
    # beta controls rain streak weight
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel
    rain_result = img.copy()  # copy a mask
    rain = np.array(rain, dtype=np.float32)  # 32 bits are required to prevent the array from crossing the boundary
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    return rain_result
def get_noise_snow(img, value=10):
    '''
    # generate noise
    value= control amount
    '''
    noise = np.random.normal(0, 255*0.25, img.shape[0:2])
    # noise = np.random.uniform(0, 256, img.shape[0:2])
    # control noise level
    # v = value * 0.01
    # noise[np.where(noise < (256 - v))] = 0
    # resize
    size = noise.shape
    resized_noise = cv2.resize(noise, (int(size[1]/4),int(size[0]/4)), interpolation = cv2.INTER_NEAREST)
    noise = cv2.resize(resized_noise, (size[1],size[0]), cv2.INTER_NEAREST, interpolation = cv2.INTER_NEAREST)
    v = value * 0.01
    # noise[np.where(noise < (256 - v))] = 0
    noise[np.where(noise < 36)] = 0
    # blur noise
    # k = np.array([[0, 0.1, 0],
    #               [0.1, 8, 0.1],
    #               [0, 0.1, 0]])

    # noise = cv2.filter2D(noise, -1, k)
    # noise = cv2.GaussianBlur(noise,(0,0),1,1,cv2.BORDER_CONSTANT)
    return noise



# ###############################################################
# distortion generation
# ###############################################################

def imblurgauss(im, param):
    # Takes in cv2 Image and returns Gaussian Blurred cv2 Image
    sigma = param
    im_dist = cv2.GaussianBlur(im,(0,0),sigma,sigma,cv2.BORDER_CONSTANT)
    return im_dist

def imblurlens(im, param):
    # Takes cv2 Image and returns lens blurred image
    # MATLAB version https://github.com/alexandrovteam/IMS_quality/blob/master/codebase/fspecialIM.m
    radius = param
    im = np.array(im)
    crad  = int(np.ceil(radius-0.5))
    [x,y] = np.meshgrid(np.arange(-crad,crad+1,1), np.arange(-crad,crad+1,1), indexing='xy')
    maxxy = np.maximum(abs(x),abs(y))
    minxy = np.minimum(abs(x),abs(y))
    m1 = np.multiply((radius**2 <  (maxxy+0.5)**2 + (minxy-0.5)**2),(minxy-0.5)) + np.nan_to_num(np.multiply((radius**2 >= (maxxy+0.5)**2 + (minxy-0.5)**2), np.sqrt(radius**2 - (maxxy + 0.5)**2)),nan=0)
    m2 = np.multiply((radius**2 >  (maxxy-0.5)**2 + (minxy+0.5)**2),(minxy+0.5)) + np.nan_to_num(np.multiply((radius**2 <= (maxxy-0.5)**2 + (minxy+0.5)**2), np.sqrt(radius**2 - (maxxy - 0.5)**2)),nan=0)
    sgrid = np.multiply((radius**2*(0.5*(np.arcsin(m2/radius) - np.arcsin(m1/radius)) + 0.25*(np.sin(2*np.arcsin(m2/radius)) - np.sin(2*np.arcsin(m1/radius)))) - np.multiply((maxxy-0.5),(m2-m1)) + (m1-minxy+0.5)) ,((((radius**2 < (maxxy+0.5)**2 + (minxy+0.5)**2) & (radius**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)) | ((minxy==0)&(maxxy-0.5 < radius)&(maxxy+0.5>=radius)))))
    sgrid = sgrid + ((maxxy+0.5)**2 + (minxy+0.5)**2 < radius**2)
    sgrid[crad,crad] = min(np.pi*radius**2,np.pi/2)
    
    if ((crad>0) and (radius > crad-0.5) and (radius**2 < (crad-0.5)**2+0.25)) :
        m1  = np.sqrt(radius**2 - (crad - 0.5)**2)
        m1n = m1/radius
        sg0 = 2*(radius**2*(0.5*np.arcsin(m1n) + 0.25*np.sin(2*np.arcsin(m1n)))-m1*(crad-0.5))
        # sgrid[2*crad+1,crad+1] = sg0
        # sgrid[crad+1,2*crad+1] = sg0
        # sgrid[crad+1,1]        = sg0
        # sgrid[1,crad+1]        = sg0
        # sgrid[2*crad,crad+1]   = sgrid[2*crad,crad+1] - sg0
        # sgrid[crad+1,2*crad]   = sgrid[crad+1,2*crad] - sg0
        # sgrid[crad+1,2]        = sgrid[crad+1,2]      - sg0
        # sgrid[2,crad+1]        = sgrid[2,crad+1]      - sg0
        sgrid[2*crad,crad]    = sg0
        sgrid[crad,2*crad]    = sg0
        sgrid[crad,0]         = sg0
        sgrid[0,crad]         = sg0
        sgrid[2*crad-1,crad]  = sgrid[2*crad-1,crad] - sg0
        sgrid[crad,2*crad-1]  = sgrid[crad,2*crad-1] - sg0
        sgrid[crad,1]         = sgrid[crad,1]      - sg0
        sgrid[1,crad]         = sgrid[1,crad]      - sg0
    sgrid[crad,crad] = min(sgrid[crad,crad],1)
    h = sgrid/sgrid.sum()
    ndimage.convolve(im[:,:,0],  h, output = im[:,:,0], mode='nearest')
    ndimage.convolve(im[:,:,1],  h, output = im[:,:,1], mode='nearest')
    ndimage.convolve(im[:,:,2],  h, output = im[:,:,2], mode='nearest')
    return im

def imblurmotion2 (im, param):
    # MATLAB version https://github.com/alexandrovteam/IMS_quality/blob/master/codebase/fspecialIM.m
    im = np.array(im)
    radius = param
    length = max(1,radius)
    half = (length-1)/2
    phi = (random.randint(0,180))/(180)*np.pi
    print(phi)
    if phi == 90:    # since 90 not work
        phi = 0
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    xsign = np.sign(cosphi)
    linewdt = 1
    sx = np.fix(half*cosphi + linewdt*xsign - length*np.finfo(float).eps)
    sy = np.fix(half*sinphi + linewdt - length*np.finfo(float).eps)
    if sx > 0:
        end_sx = sx + 1
    else :
        end_sx = sx - 1 
    if sy >= 0:
        end_sy = sy + 1
    else :
        end_sy = sy - 1 
    [x,y] = np.meshgrid(np.arange(0,end_sx,xsign), np.arange(0,end_sy,1), indexing='xy')
    dist2line = (y*cosphi-x*sinphi) 
    rad = np.sqrt(x**2 + y**2)
    x2lastpix = half - abs((x[(rad >= half)&(abs(dist2line)<=linewdt)] + dist2line[(rad >= half)&(abs(dist2line)<=linewdt)]*sinphi)/cosphi)
    dist2line[(rad >= half)&(abs(dist2line)<=linewdt)] = np.sqrt(dist2line[(rad >= half)&(abs(dist2line)<=linewdt)]**2 + x2lastpix**2)
    dist2line = linewdt + np.finfo(float).eps - abs(dist2line)
    dist2line[dist2line<0] = 0 
    h1 = np.rot90(dist2line,2)
    h2 = np.zeros([h1.shape[0]*2-1,h1.shape[1]*2-1])
    h2[0:h1.shape[0],0:h1.shape[1]] = h1
    h2[h1.shape[0]-1:2*h1.shape[0]-1,h1.shape[1]-1:h1.shape[1]*2-1] = np.rot90(np.rot90(h1))
    h2 = h2/(h2.sum() + np.finfo(float).eps*length*length)
    if cosphi>0 :
        h2 = np.flipud(h2)
    ndimage.convolve(im[:,:,0],  h2, output = im[:,:,0], mode='nearest')
    ndimage.convolve(im[:,:,1],  h2, output = im[:,:,1], mode='nearest')
    ndimage.convolve(im[:,:,2],  h2, output = im[:,:,2], mode='nearest')
    return im

def imblurmotion(im, param):
    kernel_size = param
    phi = random.choice([0,90])/(180)*np.pi
    kernel = np.zeros((kernel_size, kernel_size))
    im = np.array(im)
    if phi == 0:
        kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    else :
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel/=kernel_size
    ndimage.convolve(im[:,:,0],  kernel, output = im[:,:,0], mode='nearest')
    ndimage.convolve(im[:,:,1],  kernel, output = im[:,:,1], mode='nearest')
    ndimage.convolve(im[:,:,2],  kernel, output = im[:,:,2], mode='nearest')
    return im

def imcolordiffuse(im, param):
    amount = param
    im = np.array(im)
    sigma = 1.5*amount + 2
    scaling = amount
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    lab = color.rgb2lab(im)
    l = lab[:,:,0]
    lab = filters.gaussian(lab, sigma=sigma, channel_axis=-1)* scaling
    lab[:,:,0] = l
    im = 255*color.lab2rgb(lab)
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    im = np.clip(im, 0, 255)
    return im

def imcolorshift(im, param):
    amount = param
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = np.float32(np.array(im)/255.0)
    im = im.astype(np.float32)/255.0
    # RGB to Gray
    x =  0.2989 * im[:,:,0] + 0.5870 * im[:,:,1]+ 0.1140 * im[:,:,2]
    dx = np.gradient(x,axis=0)
    dy = np.gradient(x,axis=1)
    e = np.hypot(dx, dy)  # magnitude
    e = filters.gaussian(e, sigma=4)
    e = mapmm(e)
    e = np.clip(e,0.1,1)
    e = mapmm(e)
    percdev = [1, 1]
    valuehi = np.percentile(e,100-percdev[1])
    valuelo = 1-np.percentile(1-e,100-percdev[0])
    e = np.clip(e,valuelo,valuehi)
    e = mapmm(e)
    channel = 1
    g = im[:,:,channel]
    amt_shift = np.uint8(np.round((normalize(np.random.random([1,2]), norm='l2', axis=1) * amount)))
    padding = np.multiply(int(np.max(amt_shift)),[1, 1])
    y = np.pad(g, padding, 'symmetric')
    y = np.roll(y, amt_shift.reshape(-1))
    sl = padding[0]
    g = y [sl:-sl,sl:-sl]
    J = im
    J[:,:,channel] = np.multiply(g,e) + np.multiply(J[:,:,channel],(1-e))
    J = J * 255.0
    # im = np.uint8(J)
    im = cv2.cvtColor(J.astype(np.uint8), cv2.COLOR_RGB2BGR)
    # im = Image.fromarray(np.uint8(J))
    return im

def imcolorsaturate(im, param):
    amount = param
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = np.array(im)
    hsvIm = color.rgb2hsv(im)
    hsvIm[:,:,1] = hsvIm[:,:,1] * amount
    im = color.hsv2rgb(hsvIm) * 255.0
    im = np.clip(im,0,255)
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    # im = Image.fromarray(np.uint8(im))
    return im

def imcolorsaturate2(im, param):
    amount = param
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    lab = color.rgb2lab(im)
    lab[:,:,1:] = lab[:,:,1:] * amount
    im = color.lab2rgb(lab) * 255.0
    im = np.clip(im,0,255)
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    # im = Image.fromarray(np.uint8(im))
    return im

def imcolorbrighten(im, param) :
    amount = param
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32)/255.0
    # im = np.float32(np.array(im)/255.0)
    lab = color.rgb2lab(im)
    L = lab[:,:,0]/100.0
    L_ = curvefit(L , 0.5 + 0.5*amount)
    lab[:,:,0] = L_*100.0
    J = curvefit(im, 0.5 + 0.5*amount)
    J = (2*J + np.clip(color.lab2rgb(lab),0,1) )/3.0
    J = np.clip(J * 255.0,0,255)
    im = cv2.cvtColor(J.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return im

def imcolordarken(im, param):
    ## convert 0-1
    im = im.astype(np.float32)/255.0
    # im = np.array(im).astype(np.float32)/255.0
    ## generate curve to fit based on amount
    x = [0, 0.5, 1]
    y = [0, 0.5, 1]
    y[1] = 0.5-param/2
    ## generate interpolating function and interpolate input
    cs = interpolate.UnivariateSpline(x, y, k=2)
    yy = cs(im)
    ## convert back to PIL image
    im_out = np.clip(yy, 0, 1)
    im_out = (im_out*255).astype(np.uint8) 
    return im_out

def imcompressjpeg(im, param):
    result, encimg = cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), param])
    im = cv2.imdecode(encimg, 1)
    return im
'''
def imcompressjpeg(im,level):
    levels = [70, 43, 36, 24, 7]
    amount = levels[level]
    imgByteArr = io.BytesIO()
    im.save(imgByteArr, format='JPEG',quality=amount)
    im1 = Image.open(imgByteArr)
    return im1
'''

def imnoisegauss(im, param):
    sigma = param
    im = im.astype(np.float32) / 255.
    im += np.random.normal(0, sigma/255., im.shape)
    im = im * 255.
    return im
'''
def imnoisegauss(im, level):
    levels = [0.001, 0.002, 0.003, 0.005, 0.01]
    im = np.float32(np.array(im)/255.0)
    row,col,ch= im.shape
    var = levels[level]
    mean = 0
    sigma = var**0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = im + gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy,0,255)
    return Image.fromarray(noisy.astype('uint8'))
'''

def imnoisecolormap(im, param):
    sigma = param
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ycbcr = color.rgb2ycbcr(im)
    ycbcr = ycbcr/ 255.
    ycbcr += np.random.normal(0, sigma/255., ycbcr.shape)
    im = color.ycbcr2rgb(ycbcr * 255.) * 255.
    im = np.clip(im,0,255)
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return im
'''
def imnoisecolormap(im, level):

    levels = [0.0001, 0.0005, 0.001, 0.002, 0.003]
    var = levels[level]
    mean = 0

    im = np.array(im)
    ycbcr = color.rgb2ycbcr(im)
    ycbcr = ycbcr/ 255.0

    row,col,ch= ycbcr.shape
    sigma = var**0.5
    gauss = np.array(ycbcr.shape)
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = ycbcr + gauss

    im_dist = color.ycbcr2rgb(noisy * 255.0) * 255.0
    im_dist = np.clip(im_dist,0,255)
    return Image.fromarray(im_dist.astype('uint8'))
'''

def imnoiseimpulse(im, param):
    prob = param
    im_dist = im.copy()
    black = np.array([0, 0, 0], dtype='uint8')
    white = np.array([255, 255, 255], dtype='uint8')
    probs = np.random.random(im_dist.shape[:2])
    im_dist[probs < (prob / 2)] = black
    im_dist[probs > 1 - (prob / 2)] = white
    return im_dist
'''
def imnoiseimpulse(im, level):
    levels = [0.001, 0.005, 0.01, 0.02, 0.03]
    prob = levels[level]
    im = np.array(im)
    output = im
    black = np.array([0, 0, 0], dtype='uint8')
    white = np.array([255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return Image.fromarray(output.astype('uint8'))
'''

def imnoisemultiplicative(im, param):
    sigma = param
    im = im.astype(np.float32) / 255.
    im = im + im * np.random.normal(0, sigma/255., im.shape)
    im = im * 255.
    im = np.clip(im,0,255).astype('uint8')
    return im
'''
def imnoisemultiplicative(im, level):

    levels = [0.001, 0.005, 0.01, 0.02, 0.05]

    im = np.float32(np.array(im)/255.0)

    row,col,ch= im.shape

    var = levels[level]
    mean = 0
    sigma = var**0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = im + im * gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy,0,255)

    return Image.fromarray(noisy.astype('uint8'))
'''

def imdenoise(im, param):
    sigma = param
    im = im.astype(np.float32) / 255.
    im += np.random.normal(0, sigma/255., im.shape)
    im = im * 255.
    filt_type = np.random.randint(0,2)
    if filt_type == 0:
        im = cv2.GaussianBlur(im,(0,0),3,3,cv2.BORDER_CONSTANT)
    elif filt_type == 1:
        im = cv2.GaussianBlur(im,(0,0),2,2,cv2.BORDER_CONSTANT)
    return im
'''
def imdenoise (im,level):
    levels = [0.001, 0.002, 0.003, 0.005, 0.01]
    im = np.float32(np.array(im)/255.0)
    row,col,ch= im.shape
    var = levels[level]
    mean = 0
    sigma = var**0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = im + gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy,0,255)
    im = Image.fromarray(noisy.astype('uint8'))
    filt_type = np.random.randint(0,2)
    if filt_type == 0:
        denoised = im.filter(ImageFilter.GaussianBlur(radius = 3)) 
    elif filt_type == 1:
        denoised = im.filter(ImageFilter.BoxBlur(radius = 2)) 
    return denoised
'''

def immeanshift(im, param):
    amount = param
    im = im.astype(np.float32) / 255.
    im = im + amount
    im = im * 255.0
    im = np.clip(im,0,255)
    im = im.astype(np.uint8)
    return im
       
def imresizedist2(im, param):
    amount = param
    w = int(im.shape[1])
    h = int(im.shape[0])
    scaled_w = int(w/amount)
    scaled_h = int(h/amount)
    im = Image.fromarray(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # resize image
    resized_image = im.resize((scaled_w,scaled_h), Image.BICUBIC)
    im = resized_image.resize((w,h), Image.BICUBIC)
    im = np.array(im)
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return im
def imresizedist(im, param):    # sames to generate different results compared to above
    amount = param
    w = int(im.shape[1])
    h = int(im.shape[0])
    scaled_w = int(w/amount)
    scaled_h = int(h/amount)
    # resize image
    im_resized = cv2.resize(im, (scaled_w, scaled_h), interpolation = cv2.INTER_CUBIC)
    im = cv2.resize(im_resized, (w, h), interpolation = cv2.INTER_CUBIC)
    return im

def imresizedist_bilinear2(im, param):
    amount = param
    w = int(im.shape[1])
    h = int(im.shape[0])
    scaled_w = int(w/amount)
    scaled_h = int(h/amount)
    im = Image.fromarray(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # resize image
    resized_image = im.resize((scaled_w,scaled_h), Image.BILINEAR)
    im = resized_image.resize((w,h), Image.BILINEAR)
    im = np.array(im)
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return im
def imresizedist_bilinear(im, param):   # sames to generate different results compared to above
    amount = param
    w = int(im.shape[1])
    h = int(im.shape[0])
    scaled_w = int(w/amount)
    scaled_h = int(h/amount)
    # resize image
    im_resized = cv2.resize(im, (scaled_w, scaled_h), interpolation = cv2.INTER_LINEAR)
    im = cv2.resize(im_resized, (w, h), interpolation = cv2.INTER_LINEAR)
    return im

def imresizedist_nearest2(im, param):
    amount = param
    w = int(im.shape[1])
    h = int(im.shape[0])
    scaled_w = int(w/amount)
    scaled_h = int(h/amount)
    im = Image.fromarray(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # resize image
    resized_image = im.resize((scaled_w,scaled_h), Image.NEAREST)
    im = resized_image.resize((w,h), Image.NEAREST)
    im = np.array(im)
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return im
def imresizedist_nearest(im, param):    # sames to generate different results compared to above
    amount = param
    w = int(im.shape[1])
    h = int(im.shape[0])
    scaled_w = int(w/amount)
    scaled_h = int(h/amount)
    # resize image
    im_resized = cv2.resize(im, (scaled_w, scaled_h), interpolation = cv2.INTER_NEAREST)
    im = cv2.resize(im_resized, (w, h), interpolation = cv2.INTER_NEAREST)
    return im

def imresizedist_lanczos2(im, param):
    amount = param
    w = int(im.shape[1])
    h = int(im.shape[0])
    scaled_w = int(w/amount)
    scaled_h = int(h/amount)
    im = Image.fromarray(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # resize image
    resized_image = im.resize((scaled_w,scaled_h), Image.LANCZOS)
    im = resized_image.resize((w,h), Image.LANCZOS)
    im = np.array(im)
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return im
def imresizedist_lanczos(im, param):
    amount = param
    w = int(im.shape[1])
    h = int(im.shape[0])
    scaled_w = int(w/amount)
    scaled_h = int(h/amount)
    # resize image
    im_resized = cv2.resize(im, (scaled_w, scaled_h), interpolation = cv2.INTER_LANCZOS4)
    im = cv2.resize(im_resized, (w, h), interpolation = cv2.INTER_LANCZOS4)
    return im

def imsharpenHi(im, param):
    ## param range to be use -> double from matlab
    ## convert PIL-RGB to LAB for operation in L space
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)    #.astype(np.float32) #(not work if using float32)
    LAB = color.rgb2lab(im)
    im_L = LAB[:,:,0]
    ## compute laplacians
    gy = np.gradient(im_L, axis=0)
    gx = np.gradient(im_L, axis=1)
    ggy = np.gradient(gy, axis=0)
    ggx = np.gradient(gx, axis=1)
    laplacian = ggx + ggy
    ## subtract blurred version from image to sharpen
    im_out = im_L - param*laplacian
    ## clip L space in 0-100
    im_out = np.clip(im_out, 0, 100)
    ## convert LAB to cv2-RGB 
    LAB[:,:,0] = im_out
    im_out = 255*color.lab2rgb(LAB)
    im_out = np.clip(im_out,0,255)
    im_out = cv2.cvtColor(im_out.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return im_out

def imcontrastc(im, param):
    ## convert 0-1
    im = im.astype(np.float32) / 255.
    ## generate curve to fit based on amount->param
    coef = [[0.3, 0.5, 0.7],[0.25-param/4, 0.5, 0.75+param/4]]
    defa = 0
    x = [0, 0, 0, 0, 1]
    x[1:-1] = coef[0] 
    y = [0, 0, 0, 0, 1]
    y[1:-1] = coef[1] 
    ## generate interpolating function and interpolate input
    cs = interpolate.UnivariateSpline(x, y)
    yy = cs(im)
    ## convert back to cv2 image
    im_out = np.clip(yy, 0, 1)
    im_out = (im_out*255).astype(np.uint8) 
    return im_out

def imcolorblock(im, param):
    ## convert 0-1
    im = im.astype(np.float32)
    # im_out = im.copy()
    h, w, _ = im.shape
    ## define patchsize
    patch_size = [32, 32]
    h_max = h - patch_size[0]
    w_max = w - patch_size[1]
    block = np.ones((patch_size[0], patch_size[1], 3))
    ## place the color blocks at random
    for i in range(0, param):
        color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        x = int(random.uniform(0, 1) * w_max)
        y = int(random.uniform(0, 1) * h_max)
        im[y:y+patch_size[0], x:x+patch_size[1],:] = color*block
    ## convert back to cv2 image
    im_out = im.astype(np.uint8) 
    return im_out

def impixelate(im, param):
    z = 0.95 - param**(0.6)
    w = int(im.shape[1])
    h = int(im.shape[0])
    scaled_w = int(z*w)
    scaled_h = int(z*h)
    im = Image.fromarray(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # resize image
    resized_image = im.resize((scaled_w,scaled_h), Image.NEAREST)
    im = resized_image.resize((w,h), Image.NEAREST)
    im = np.array(im)
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return im

def imnoneccentricity(im, param):
    ## convert 0-1
    im = im.astype(np.float32)
    h, w, _ = im.shape
    ## define patchsize
    patch_size = [16, 16]
    radius = 16
    h_min = radius
    w_min = radius
    h_max = h - patch_size[0] - radius
    w_max = w - patch_size[1] - radius
    block = np.ones((patch_size[0], patch_size[1], 3))
    ## place the color blocks at random
    for i in range(0, param):
        w_start = int(random.uniform(0, 1) * (w_max - w_min)) + w_min
        h_start = int(random.uniform(0, 1) * (h_max - h_min)) + h_min
        patch = im[h_start:h_start+patch_size[0],w_start:w_start+patch_size[1],:]
        rand_w_start = int((random.uniform(0, 1) - 0.5)*radius + w_start)
        rand_h_start = int((random.uniform(0, 1) - 0.5)*radius + h_start)
        im[rand_h_start:rand_h_start+patch_size[0],rand_w_start:rand_w_start+patch_size[1],:] = patch
    ## convert back to PIL image
    im_out = im.astype(np.uint8) 
    return im_out

def imjitter(im, param):
    ## convert 0-1
    im = im.astype(np.float32)
    h, w, _ = im.shape
    sz = [h,w,2]
    ## iterate image-warp over for 5 times
    J = im
    for i in range(0,5):
        ## generate random shift map
        shifts = np.random.randn(h,w,2)*param
        J = imwarpmap(J, shifts)
    ## convert back to PIL image
    im_out = (J).astype(np.uint8) 
    return im_out

def immosaic(im):
    # https://github.com/ofsoundof/GRL-Image-Restoration/blob/3123fcadfee2107d813c5e8fff58227f14c989bc/utils/utils_mosaic.py
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im, h, w = flip_pad(im, 2)
    # im = mosaic_CFA_Bayer(im)[1]
    im = mosaic_CFA_Bayer(im)[2]
    im = depad(im, h, w)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im

def impainting_irregular(im, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
    # https://github.com/advimman/lama
    im = im.astype(np.float32)/255.
    coef = 1
    cur_max_len = int(max(1, max_len * coef))
    cur_max_width = int(max(1, max_width * coef))
    cur_max_times = int(min_times + 1 + (max_times - min_times) * coef)
    mask = make_random_irregular_mask(im.shape[0:2], max_angle=max_angle, max_len=cur_max_len,
                                        max_width=cur_max_width, min_times=min_times, max_times=cur_max_times,
                                        draw_method=draw_method)
    mask = np.transpose(mask, (1, 2, 0))
    im = (im * (1-mask))*255.
    return im

def impainting_rectangle(im, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, ramp_kwargs=None):
    # https://github.com/advimman/lama
    im = im.astype(np.float32)/255.
    coef = 1
    cur_bbox_max_size = int(bbox_min_size + 1 + (bbox_max_size - bbox_min_size) * coef)
    cur_max_times = int(min_times + (max_times - min_times) * coef)
    mask = make_random_rectangle_mask(im.shape[0:2], margin=margin, bbox_min_size=bbox_min_size,
                                        bbox_max_size=cur_bbox_max_size, min_times=min_times,
                                        max_times=cur_max_times)
    mask = np.transpose(mask, (1, 2, 0))
    im = (im * (1-mask))*255.
    return im

def imrain(im, noise=500, rain_len=50, rain_angle=30, rain_thickness=3, alpha=0.7):
    # https://github.com/ShenZheng2000/Rain-Generation-Python
    noise = get_noise(im, value=noise)
    rain = rain_blur(noise, length=rain_len, angle=rain_angle, w=rain_thickness)
    im_out = alpha_rain(rain, im, beta=alpha)
    return im_out

def imsnow(im, noise=500, snow_len=50, snow_angle=30, snow_thickness=3, alpha=0.7):
    # noise = get_noise_snow(im, value=noise)
    # snow = rain_blur(noise, length=snow_len, angle=snow_angle, w=snow_thickness)
    # im_out = alpha_rain(snow, im, beta=alpha)
    # return im_out
    noise = get_noise(im, value=noise)
    snow = snow_blur(noise, length=snow_len, angle=snow_angle, w=snow_thickness)
    im_out = alpha_snow(snow, im, beta=alpha)
    return im_out


def add_distortions_random(choice, img):
    clean = False
    if choice == 1:
        if random.random() > 0.75:
            param = random.randint(5, 96)   # imjpeg: quality_factor (8,96)
        else:
            param = random.choice([10,20,30,40])   # imjpeg quality_factor: [10,20,30,40,50,60]
        img_dist = imcompressjpeg(img, param)
        label = "jpeg compression"
        level = param
        condition = "remove the jpeg compression artifact."
    elif choice == 2:
        param = random.uniform(0.1, 5)   # imblurgauss: sigma (0.1,5)
        img_dist = imblurgauss(img, param)
        label = "gaussian blur"
        level = param
        condition = "remove the blur distortion and make the image clear."
    elif choice == 3:
        param = random.uniform(1, 8)   # imblurlens: radius (1,8)
        img_dist = imblurlens(img, param)
        label = "lens blur"
        level = param
        condition = "remove the blur distortion and make the image clear."
    elif choice == 4:
        param = random.randint(12, 28)   # imblurmotion: kernel_size (12,28)
        img_dist = imblurmotion(img, param)
        label = "motion blur"
        level = param
        condition = "remove the blur distortion and make the image clear."
    elif choice == 5:
        param = random.uniform(1, 12)   # imcolordiffuse: amount (1,12)
        img_dist = imcolordiffuse(img, param)
        label = "color diffuse"
        level = param
        condition = "make the image color normal."
    elif choice == 6:
        param = random.uniform(1, 12)   # imcolorshift: amount (1,12)
        img_dist = imcolorshift(img, param)
        label = "color shift"
        level = param
        condition = "make the image color normal."
    elif choice == 7:
        param = random.uniform(0.4, -0.4)   # imcolorsaturate: amount (0.4,-0.4)
        img_dist = imcolorsaturate(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal."
    elif choice == 8:
        param = random.uniform(1, 9)   # imcolorsaturate2: amount (1,9)
        img_dist = imcolorsaturate2(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal."
    elif choice == 9:
        param = random.uniform(0, 50)   # imnoisegauss: amount (0, 50)
        img_dist = imnoisegauss(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear."
    elif choice == 10:
        param = random.uniform(0.0001, 0.003)   # imnoisecolormap: amount (0.0001, 0.003)
        param = (param**0.5)*255.
        img_dist = imnoisecolormap(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear."
    elif choice == 11:
        param = random.uniform(0.001, 0.03) # imnoiseimpulse: amount (0.001, 0.03)
        img_dist = imnoiseimpulse(img, param)
        label = "impulse noise"
        level = param
        condition = "remove the noise distortion and make the image clear."
    elif choice == 12:
        param = random.uniform(0.001, 0.05)   # imnoisemultiplicative: amount (0.001, 0.05)
        param = (param**0.5)*255.
        img_dist = imnoisemultiplicative(img, param)
        label = "multiplicative noise"
        level = param
        condition = "remove the noise distortion and make the image clear."
    elif choice == 13:
        param = random.uniform(0, 50) # imdenoise: amount (0, 50)
        img_dist = imdenoise(img, param)
        label = "denoise"
        level = param
        condition = "make the image clear."
    elif choice == 14:
        param = random.uniform(0.1, 1.1) # imcolorbrighten: amount (0.001, 0.03)
        img_dist = imcolorbrighten(img, param)
        label = "over bright"
        level = param
        condition = "reduce the light in this image and make the image light normal."
    elif choice == 15:
        param = random.uniform(0.05, 0.8) # imdenoise: amount (0.05, 0.8)
        img_dist = imcolordarken(img, param)
        label = "over dark"
        level = param
        condition = "enhance the light in this image and make the image light normal."
    elif choice == 16:
        param = random.uniform(0.15, -0.15) # imdenoise: amount (0.05, 0.8)
        img_dist = immeanshift(img, param)
        label = "mean shift"
        level = param
        condition = "make the image color and light normal."
    elif choice == 17:
        param = random.randint(2, 16)   # imresizedist2: kernel_size (2, 16)
        img_dist = imresizedist2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear."
    elif choice == 18:
        param = random.randint(2, 16)   # imresizedist_bilinear2: kernel_size (2, 16)
        img_dist = imresizedist_bilinear2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear."
    elif choice == 19:
        param = random.randint(2, 16)   # imresizedist_nearest2: kernel_size (2, 16)
        img_dist = imresizedist_nearest2(img, param)
        label = "resize"
        level = param
        condition = "remove the pixelate distortion and make the image clear."
    elif choice == 20:
        param = random.randint(2, 16)   # imresizedist_lanczos2: kernel_size (2, 16)
        img_dist = imresizedist_lanczos2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear."
    elif choice == 21:
        param = random.randint(1, 12) # imsharpenHi: amount (1, 12)
        img_dist = imsharpenHi(img, param)
        label = "over sharpening"
        level = param
        condition = "smooth image."
    elif choice == 22:
        param = random.uniform(0.3, -0.6) # imcontrastc: amount (0.05, 0.8)
        img_dist = imcontrastc(img, param)
        label = "contrast imbalance"
        level = param
        condition = "balance contrast."
    elif choice == 23:
        param = random.randint(2, 10) # imcolorblock: amount (2, 10)
        img_dist = imcolorblock(img, param)
        label = "color block"
        level = param
        condition = "remove color blocks."
    elif choice == 24:
        param = random.uniform(0.01, 0.5) # impixelate: amount (0.01, 0.5)
        img_dist = impixelate(img, param)
        label = "pixelate"
        level = param
        condition = "remove the pixelate distortion and make the image clear."
    elif choice == 25:
        param = random.randint(20, 100) # imnoneccentricity: amount (20, 100)
        img_dist = imnoneccentricity(img, param)
        label = "discontinuous"
        level = param
        condition = "make the image continuous."
    elif choice == 26:
        param = random.uniform(0.05, 1) # imjitter: amount (0.05, 1)
        img_dist = imjitter(img, param)
        label = "jitter"
        level = param
        condition = "remove the jitter distortion."
    elif choice == 27:
        img_dist = immosaic(img)
        label = "mosaic"
        level = 1
        condition = "remove the mosaic mask."
    elif choice == 28:
        # impainting_irregular
        max_angle = 4
        max_len = 100
        max_width = 30
        min_times=0
        max_times=20
        img_dist = impainting_irregular(img, max_angle, max_len, max_width, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask."
    elif choice == 29:
        # impainting_rectangle
        margin=10
        bbox_min_size=30
        bbox_max_size=100
        min_times=0
        max_times=10
        img_dist = impainting_rectangle(img, margin, bbox_min_size, bbox_max_size, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask."
    elif choice == 30:
        # imrain
        noise = random.randint(1000, 5000)
        rain_len = random.randint(4, 10)
        rain_angle = random.randint(0, 45)
        rain_thickness = random.randrange(3,8,2)    # odd
        alpha = random.uniform(0.85, 0.95)
        img_dist = imrain(img, noise, rain_len, rain_angle, rain_thickness, alpha)
        label = "rain streak"
        level = noise
        condition = "remove the rain streak."
    elif choice == 31:
        # imsnow
        noise = random.randint(300, 1000)
        snow_len = random.randint(1, 3)
        snow_angle = random.randint(0, 30)
        snow_thickness = random.randrange(1,4,2)    # odd
        alpha = random.uniform(0.9, 1)
        img_dist = imsnow(img, noise, snow_len, snow_angle, snow_thickness, alpha)
        label = "snow streak"
        level = noise
        condition = "remove the snow streak."
    else :
        img_dist = img
        label = "clean high-quality image."
        level = 0
        clean = True
        condition = "keep image unchanged."
        
    if clean:
        noise_label = label
    else:
        noise_label = "low-quality image with " + label + " distortion."

    # img_dist = np.clip(img_dist.astype(np.uint8),0,255) # make sure the image range
    return img_dist, noise_label, level, condition


def add_distortions_test(choice, img):
    clean = False
    if choice == 1:
        # if random.random() > 0.75:
        #     param = random.randint(5, 96)   # imjpeg: quality_factor (8,96)
        # else:
        #     param = random.choice([10,20,30,40])   # imjpeg quality_factor: [10,20,30,40,50,60]
        param = 7  # range: [70, 43, 36, 24, 7]
        img_dist = imcompressjpeg(img, param)
        label = "jpeg compression"
        level = param
        condition = "remove the jpeg compression artifact"
    elif choice == 2:
        # param = random.uniform(0.1, 5)   # imblurgauss: sigma (0.1,5)
        param = 5  # range: [0.1, 0.5, 1, 2, 5]
        img_dist = imblurgauss(img, param)
        label = "gaussian blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 3:
        # param = random.uniform(1, 8)   # imblurlens: radius (1,8)
        param = 8  # range: [1, 2, 4, 6, 8]
        img_dist = imblurlens(img, param)
        label = "lens blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 4:
        # param = random.randint(12, 28)   # imblurmotion: kernel_size (12,28)
        param = 28  # range: [12, 16, 20, 24, 28]
        img_dist = imblurmotion(img, param)
        label = "motion blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 5:
        # param = random.uniform(1, 12)   # imcolordiffuse: amount (1,12)
        param = 12  # range: [1, 3, 6, 8, 12]
        img_dist = imcolordiffuse(img, param)
        label = "color diffuse"
        level = param
        condition = "make the image color normal"
    elif choice == 6:
        # param = random.uniform(1, 12)   # imcolorshift: amount (1,12)
        param = 12  # range: [1, 3, 6, 8, 12]
        img_dist = imcolorshift(img, param)
        label = "color shift"
        level = param
        condition = "make the image color normal"
    elif choice == 7:
        # param = random.uniform(0.4, -0.4)   # imcolorsaturate: amount (0.4,-0.4)
        param = -0.4  # range: [0.4, 0.2, 0.1, 0, -0.4]
        img_dist = imcolorsaturate(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal"
    elif choice == 8:
        # param = random.uniform(1, 9)   # imcolorsaturate2: amount (1,9)
        param = 9 # range: [1, 2, 3, 6, 9]
        img_dist = imcolorsaturate2(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal"
    elif choice == 9:
        # param = random.uniform(0, 50)   # imnoisegauss: amount (0, 50)
        param = 25 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
        img_dist = imnoisegauss(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 10:
        # param = random.uniform(0.0001, 0.003)   # imnoisecolormap: amount (0.0001, 0.003)
        param = 0.003    # range: [0.0001, 0.0005, 0.001, 0.002, 0.003]
        param = (param**0.5)*255.
        img_dist = imnoisecolormap(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 11:
        # param = random.uniform(0.001, 0.03) # imnoiseimpulse: amount (0.001, 0.03)
        param = 0.03 # range: [0.001, 0.005, 0.01, 0.02, 0.03]
        img_dist = imnoiseimpulse(img, param)
        label = "impulse noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 12:
        # param = random.uniform(0.001, 0.05)   # imnoisemultiplicative: amount (0.001, 0.05)
        param = 0.05    # range: [0.001, 0.005, 0.01, 0.02, 0.05]
        param = (param**0.5)*255.
        img_dist = imnoisemultiplicative(img, param)
        label = "multiplicative noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 13:
        # param = random.uniform(0, 50) # imdenoise: amount (0, 50)
        param = 25 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
        img_dist = imdenoise(img, param)
        label = "denoise"
        level = param
        condition = "make the image clear"
    elif choice == 14:
        # param = random.uniform(0.1, 1.1) # imcolorbrighten: amount (0.001, 0.03)
        param = 1.1 # range: [0.1, 0.2, 0.4, 0.7, 1.1]
        img_dist = imcolorbrighten(img, param)
        label = "over bright"
        level = param
        condition = "reduce the light in this image and make the image light normal"
    elif choice == 15:
        # param = random.uniform(0.05, 0.8) # imdenoise: amount (0.05, 0.8)
        param = 0.8 # range: [0.05, 0.1, 0.2, 0.4, 0.8]
        img_dist = imcolordarken(img, param)
        label = "over dark"
        level = param
        condition = "enhance the light in this image and make the image light normal"
    elif choice == 16:
        # param = random.uniform(0.15, -0.15) # imdenoise: amount (0.05, 0.8)
        param = -0.15 # range: [0.15, 0.08, 0, -0.08, -0.15]
        img_dist = immeanshift(img, param)
        label = "mean shift"
        level = param
        condition = "make the image color and light normal"
    elif choice == 17:
        # param = random.randint(2, 16)   # imresizedist2: kernel_size (2, 16)
        param = 16 # range: [2,3,4,8,16]
        img_dist = imresizedist2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 18:
        # param = random.randint(2, 16)   # imresizedist_bilinear2: kernel_size (2, 16)
        param = 16 # range: [2,3,4,8,16]
        img_dist = imresizedist_bilinear2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 19:
        # param = random.randint(2, 16)   # imresizedist_nearest2: kernel_size (2, 16)
        param = 6 # range: [2,3,4,5,6]
        img_dist = imresizedist_nearest2(img, param)
        label = "resize"
        level = param
        condition = "remove the pixelate distortion and make the image clear"
    elif choice == 20:
        # param = random.randint(2, 16)   # imresizedist_lanczos2: kernel_size (2, 16)
        param = 16 # range: [2,3,4,8,16]
        img_dist = imresizedist_lanczos2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 21:
        # param = random.randint(1, 12) # imsharpenHi: amount (1, 12)
        param = 12 # range: [1, 2, 3, 6, 12]
        img_dist = imsharpenHi(img, param)
        label = "over sharpening"
        level = param
        condition = "smooth image"
    elif choice == 22:
        # param = random.uniform(0.3, -0.6) # imcontrastc: amount (0.05, 0.8)
        param = -0.6 # range: [0.3, 0.15, 0, -0.4, -0.6]
        # param = 16
        img_dist = imcontrastc(img, param)
        label = "contrast imbalance"
        level = param
        condition = "balance contrast"
    elif choice == 23:
        # param = random.randint(2, 10) # imcolorblock: amount (2, 10)
        param = 10 # range: [2, 4, 6, 8, 10]
        img_dist = imcolorblock(img, param)
        label = "color block"
        level = param
        condition = "remove color blocks"
    elif choice == 24:
        # param = random.uniform(0.01, 0.5) # impixelate: amount (0.01, 0.5)
        param = 0.5 # range: [0.01, 0.05, 0.1, 0.2, 0.5]
        img_dist = impixelate(img, param)
        label = "pixelate"
        level = param
        condition = "remove the pixelate distortion and make the image clear"
    elif choice == 25:
        # param = random.randint(20, 100) # imnoneccentricity: amount (20, 100)
        param = 100 # range: [20, 40, 60, 80, 100]
        img_dist = imnoneccentricity(img, param)
        label = "discontinuous"
        level = param
        condition = "make the image continuous"
    elif choice == 26:
        # param = random.uniform(0.05, 1) # imjitter: amount (0.05, 1)
        param = 1 # range: [0.05, 0.1, 0.2, 0.5, 1]
        img_dist = imjitter(img, param)
        label = "jitter"
        level = param
        condition = "remove the jitter distortion"
    elif choice == 27:
        img_dist = immosaic(img)
        label = "mosaic"
        level = 1
        condition = "remove the mosaic mask"
    elif choice == 28:
        # impainting_irregular
        max_angle = 4
        max_len = 100
        max_width = 30
        min_times=0
        max_times=20
        img_dist = impainting_irregular(img, max_angle, max_len, max_width, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask"
    elif choice == 29:
        # impainting_rectangle
        margin=10
        bbox_min_size=30
        bbox_max_size=100
        min_times=0
        max_times=10
        img_dist = impainting_rectangle(img, margin, bbox_min_size, bbox_max_size, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask"
    elif choice == 30:
        # imrain
        noise = 5000
        rain_len = 10
        rain_angle = 30
        rain_thickness = 7
        alpha = 0.9
        img_dist = imrain(img, noise, rain_len, rain_angle, rain_thickness, alpha)
        label = "rain streak"
        level = noise
        condition = "remove the rain streak"
    elif choice == 31:
        # imsnow
        noise = 500
        snow_len = 3
        snow_angle = 10
        snow_thickness = 1
        alpha = 0.9
        img_dist = imsnow(img, noise, snow_len, snow_angle, snow_thickness, alpha)
        label = "snow streak"
        level = noise
        condition = "remove the snow streak"
    else :
        img_dist = img
        label = "high-quality clean image."
        level = 0
        clean = True
        condition = "keep image unchanged."
        
    if clean:
        noise_label = label
    else:
        noise_label = "low-quality image with " + label + " distortion."

    # img_dist = np.clip(img_dist.astype(np.uint8),0,255) # make sure the image range
    return img_dist, noise_label, level, condition


def add_distortions_test_heavy(choice, img):
    clean = False
    if choice == 1:
        # if random.random() > 0.75:
        #     param = random.randint(5, 96)   # imjpeg: quality_factor (8,96)
        # else:
        #     param = random.choice([10,20,30,40])   # imjpeg quality_factor: [10,20,30,40,50,60]
        param = 7  # range: [70, 43, 36, 24, 7]
        img_dist = imcompressjpeg(img, param)
        label = "jpeg compression"
        level = param
        condition = "remove the jpeg compression artifact"
    elif choice == 2:
        # param = random.uniform(0.1, 5)   # imblurgauss: sigma (0.1,5)
        param = 5  # range: [0.1, 0.5, 1, 2, 5]
        img_dist = imblurgauss(img, param)
        label = "gaussian blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 3:
        # param = random.uniform(1, 8)   # imblurlens: radius (1,8)
        param = 8  # range: [1, 2, 4, 6, 8]
        img_dist = imblurlens(img, param)
        label = "lens blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 4:
        # param = random.randint(12, 28)   # imblurmotion: kernel_size (12,28)
        param = 28  # range: [12, 16, 20, 24, 28]
        img_dist = imblurmotion(img, param)
        label = "motion blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 5:
        # param = random.uniform(1, 12)   # imcolordiffuse: amount (1,12)
        param = 12  # range: [1, 3, 6, 8, 12]
        img_dist = imcolordiffuse(img, param)
        label = "color diffuse"
        level = param
        condition = "make the image color normal"
    elif choice == 6:
        # param = random.uniform(1, 12)   # imcolorshift: amount (1,12)
        param = 12  # range: [1, 3, 6, 8, 12]
        img_dist = imcolorshift(img, param)
        label = "color shift"
        level = param
        condition = "make the image color normal"
    elif choice == 7:
        # param = random.uniform(0.4, -0.4)   # imcolorsaturate: amount (0.4,-0.4)
        param = -0.4  # range: [0.4, 0.2, 0.1, 0, -0.4]
        img_dist = imcolorsaturate(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal"
    elif choice == 8:
        # param = random.uniform(1, 9)   # imcolorsaturate2: amount (1,9)
        param = 9 # range: [1, 2, 3, 6, 9]
        img_dist = imcolorsaturate2(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal"
    elif choice == 9:
        # param = random.uniform(0, 50)   # imnoisegauss: amount (0, 50)
        param = 50 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
        img_dist = imnoisegauss(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 10:
        # param = random.uniform(0.0001, 0.003)   # imnoisecolormap: amount (0.0001, 0.003)
        param = 0.003    # range: [0.0001, 0.0005, 0.001, 0.002, 0.003]
        param = (param**0.5)*255.
        img_dist = imnoisecolormap(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 11:
        # param = random.uniform(0.001, 0.03) # imnoiseimpulse: amount (0.001, 0.03)
        param = 0.03 # range: [0.001, 0.005, 0.01, 0.02, 0.03]
        img_dist = imnoiseimpulse(img, param)
        label = "impulse noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 12:
        # param = random.uniform(0.001, 0.05)   # imnoisemultiplicative: amount (0.001, 0.05)
        param = 0.05    # range: [0.001, 0.005, 0.01, 0.02, 0.05]
        param = (param**0.5)*255.
        img_dist = imnoisemultiplicative(img, param)
        label = "multiplicative noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 13:
        # param = random.uniform(0, 50) # imdenoise: amount (0, 50)
        param = 50 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
        img_dist = imdenoise(img, param)
        label = "denoise"
        level = param
        condition = "make the image clear"
    elif choice == 14:
        # param = random.uniform(0.1, 1.1) # imcolorbrighten: amount (0.001, 0.03)
        param = 1.1 # range: [0.1, 0.2, 0.4, 0.7, 1.1]
        img_dist = imcolorbrighten(img, param)
        label = "over bright"
        level = param
        condition = "reduce the light in this image and make the image light normal"
    elif choice == 15:
        # param = random.uniform(0.05, 0.8) # imdenoise: amount (0.05, 0.8)
        param = 0.8 # range: [0.05, 0.1, 0.2, 0.4, 0.8]
        img_dist = imcolordarken(img, param)
        label = "over dark"
        level = param
        condition = "enhance the light in this image and make the image light normal"
    elif choice == 16:
        # param = random.uniform(0.15, -0.15) # imdenoise: amount (0.05, 0.8)
        param = -0.15 # range: [0.15, 0.08, 0, -0.08, -0.15]
        img_dist = immeanshift(img, param)
        label = "mean shift"
        level = param
        condition = "make the image color and light normal"
    elif choice == 17:
        # param = random.randint(2, 16)   # imresizedist2: kernel_size (2, 16)
        param = 16 # range: [2,3,4,8,16]
        img_dist = imresizedist2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 18:
        # param = random.randint(2, 16)   # imresizedist_bilinear2: kernel_size (2, 16)
        param = 16 # range: [2,3,4,8,16]
        img_dist = imresizedist_bilinear2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 19:
        # param = random.randint(2, 16)   # imresizedist_nearest2: kernel_size (2, 16)
        param = 6 # range: [2,3,4,5,6]
        img_dist = imresizedist_nearest2(img, param)
        label = "resize"
        level = param
        condition = "remove the pixelate distortion and make the image clear"
    elif choice == 20:
        # param = random.randint(2, 16)   # imresizedist_lanczos2: kernel_size (2, 16)
        param = 16 # range: [2,3,4,8,16]
        img_dist = imresizedist_lanczos2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 21:
        # param = random.randint(1, 12) # imsharpenHi: amount (1, 12)
        param = 12 # range: [1, 2, 3, 6, 12]
        img_dist = imsharpenHi(img, param)
        label = "over sharpening"
        level = param
        condition = "smooth image"
    elif choice == 22:
        # param = random.uniform(0.3, -0.6) # imcontrastc: amount (0.05, 0.8)
        param = -0.6 # range: [0.3, 0.15, 0, -0.4, -0.6]
        # param = 16
        img_dist = imcontrastc(img, param)
        label = "contrast imbalance"
        level = param
        condition = "balance contrast"
    elif choice == 23:
        # param = random.randint(2, 10) # imcolorblock: amount (2, 10)
        param = 10 # range: [2, 4, 6, 8, 10]
        img_dist = imcolorblock(img, param)
        label = "color block"
        level = param
        condition = "remove color blocks"
    elif choice == 24:
        # param = random.uniform(0.01, 0.5) # impixelate: amount (0.01, 0.5)
        param = 0.5 # range: [0.01, 0.05, 0.1, 0.2, 0.5]
        img_dist = impixelate(img, param)
        label = "pixelate"
        level = param
        condition = "remove the pixelate distortion and make the image clear"
    elif choice == 25:
        # param = random.randint(20, 100) # imnoneccentricity: amount (20, 100)
        param = 100 # range: [20, 40, 60, 80, 100]
        img_dist = imnoneccentricity(img, param)
        label = "discontinuous"
        level = param
        condition = "make the image continuous"
    elif choice == 26:
        # param = random.uniform(0.05, 1) # imjitter: amount (0.05, 1)
        param = 1 # range: [0.05, 0.1, 0.2, 0.5, 1]
        img_dist = imjitter(img, param)
        label = "jitter"
        level = param
        condition = "remove the jitter distortion"
    elif choice == 27:
        img_dist = immosaic(img)
        label = "mosaic"
        level = 1
        condition = "remove the mosaic mask"
    elif choice == 28:
        # impainting_irregular
        max_angle = 4
        max_len = 100
        max_width = 30
        min_times=0
        max_times=20
        img_dist = impainting_irregular(img, max_angle, max_len, max_width, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask"
    elif choice == 29:
        # impainting_rectangle
        margin=10
        bbox_min_size=30
        bbox_max_size=100
        min_times=0
        max_times=10
        img_dist = impainting_rectangle(img, margin, bbox_min_size, bbox_max_size, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask"
    elif choice == 30:
        # imrain
        noise = 5000
        rain_len = 10
        rain_angle = 30
        rain_thickness = 7
        alpha = 0.9
        img_dist = imrain(img, noise, rain_len, rain_angle, rain_thickness, alpha)
        label = "rain streak"
        level = noise
        condition = "remove the rain streak"
    elif choice == 31:
        # imsnow
        noise = 1000
        snow_len = 4
        snow_angle = 10
        snow_thickness = 1
        alpha = 1
        img_dist = imsnow(img, noise, snow_len, snow_angle, snow_thickness, alpha)
        label = "snow streak"
        level = noise
        condition = "remove the snow streak"
    else :
        img_dist = img
        label = "high-quality clean image."
        level = 0
        clean = True
        condition = "keep image unchanged."
        
    if clean:
        noise_label = label
    else:
        noise_label = "low-quality image with " + label + " distortion."

    # img_dist = np.clip(img_dist.astype(np.uint8),0,255) # make sure the image range
    return img_dist, noise_label, level, condition


def add_distortions_test_middle(choice, img):
    clean = False
    if choice == 1:
        # if random.random() > 0.75:
        #     param = random.randint(5, 96)   # imjpeg: quality_factor (8,96)
        # else:
        #     param = random.choice([10,20,30,40])   # imjpeg quality_factor: [10,20,30,40,50,60]
        param = 20  # range: [70, 43, 36, 24, 7]
        img_dist = imcompressjpeg(img, param)
        label = "jpeg compression"
        level = param
        condition = "remove the jpeg compression artifact"
    elif choice == 2:
        # param = random.uniform(0.1, 5)   # imblurgauss: sigma (0.1,5)
        param = 1  # range: [0.1, 0.5, 1, 2, 5]
        img_dist = imblurgauss(img, param)
        label = "gaussian blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 3:
        # param = random.uniform(1, 8)   # imblurlens: radius (1,8)
        param = 4  # range: [1, 2, 4, 6, 8]
        img_dist = imblurlens(img, param)
        label = "lens blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 4:
        # param = random.randint(12, 28)   # imblurmotion: kernel_size (12,28)
        param = 20  # range: [12, 16, 20, 24, 28]
        img_dist = imblurmotion(img, param)
        label = "motion blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 5:
        # param = random.uniform(1, 12)   # imcolordiffuse: amount (1,12)
        param = 6  # range: [1, 3, 6, 8, 12]
        img_dist = imcolordiffuse(img, param)
        label = "color diffuse"
        level = param
        condition = "make the image color normal"
    elif choice == 6:
        # param = random.uniform(1, 12)   # imcolorshift: amount (1,12)
        param = 6  # range: [1, 3, 6, 8, 12]
        img_dist = imcolorshift(img, param)
        label = "color shift"
        level = param
        condition = "make the image color normal"
    elif choice == 7:
        # param = random.uniform(0.4, -0.4)   # imcolorsaturate: amount (0.4,-0.4)
        param = 0.1  # range: [0.4, 0.2, 0.1, 0, -0.4]
        img_dist = imcolorsaturate(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal"
    elif choice == 8:
        # param = random.uniform(1, 9)   # imcolorsaturate2: amount (1,9)
        param = 3 # range: [1, 2, 3, 6, 9]
        img_dist = imcolorsaturate2(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal"
    elif choice == 9:
        # param = random.uniform(0, 50)   # imnoisegauss: amount (0, 50)
        param = 25 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
        img_dist = imnoisegauss(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 10:
        # param = random.uniform(0.0001, 0.003)   # imnoisecolormap: amount (0.0001, 0.003)
        param = 0.001    # range: [0.0001, 0.0005, 0.001, 0.002, 0.003]
        param = (param**0.5)*255.
        img_dist = imnoisecolormap(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 11:
        # param = random.uniform(0.001, 0.03) # imnoiseimpulse: amount (0.001, 0.03)
        param = 0.01 # range: [0.001, 0.005, 0.01, 0.02, 0.03]
        img_dist = imnoiseimpulse(img, param)
        label = "impulse noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 12:
        # param = random.uniform(0.001, 0.05)   # imnoisemultiplicative: amount (0.001, 0.05)
        param = 0.01    # range: [0.001, 0.005, 0.01, 0.02, 0.05]
        param = (param**0.5)*255.
        img_dist = imnoisemultiplicative(img, param)
        label = "multiplicative noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 13:
        # param = random.uniform(0, 50) # imdenoise: amount (0, 50)
        param = 25 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
        img_dist = imdenoise(img, param)
        label = "denoise"
        level = param
        condition = "make the image clear"
    elif choice == 14:
        # param = random.uniform(0.1, 1.1) # imcolorbrighten: amount (0.001, 0.03)
        param = 0.4 # range: [0.1, 0.2, 0.4, 0.7, 1.1]
        img_dist = imcolorbrighten(img, param)
        label = "over bright"
        level = param
        condition = "reduce the light in this image and make the image light normal"
    elif choice == 15:
        # param = random.uniform(0.05, 0.8) # imdenoise: amount (0.05, 0.8)
        param = 0.2 # range: [0.05, 0.1, 0.2, 0.4, 0.8]
        img_dist = imcolordarken(img, param)
        label = "over dark"
        level = param
        condition = "enhance the light in this image and make the image light normal"
    elif choice == 16:
        # param = random.uniform(0.15, -0.15) # imdenoise: amount (0.05, 0.8)
        param = -0.15 # range: [0.15, 0.08, 0, -0.08, -0.15]
        img_dist = immeanshift(img, param)
        label = "mean shift"
        level = param
        condition = "make the image color and light normal"
    elif choice == 17:
        # param = random.randint(2, 16)   # imresizedist2: kernel_size (2, 16)
        param = 4 # range: [2,3,4,8,16]
        img_dist = imresizedist2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 18:
        # param = random.randint(2, 16)   # imresizedist_bilinear2: kernel_size (2, 16)
        param = 4 # range: [2,3,4,8,16]
        img_dist = imresizedist_bilinear2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 19:
        # param = random.randint(2, 16)   # imresizedist_nearest2: kernel_size (2, 16)
        param = 4 # range: [2,3,4,5,6]
        img_dist = imresizedist_nearest2(img, param)
        label = "resize"
        level = param
        condition = "remove the pixelate distortion and make the image clear"
    elif choice == 20:
        # param = random.randint(2, 16)   # imresizedist_lanczos2: kernel_size (2, 16)
        param = 4 # range: [2,3,4,8,16]
        img_dist = imresizedist_lanczos2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 21:
        # param = random.randint(1, 12) # imsharpenHi: amount (1, 12)
        param = 3 # range: [1, 2, 3, 6, 12]
        img_dist = imsharpenHi(img, param)
        label = "over sharpening"
        level = param
        condition = "smooth image"
    elif choice == 22:
        # param = random.uniform(0.3, -0.6) # imcontrastc: amount (0.05, 0.8)
        param = 0 # range: [0.3, 0.15, 0, -0.4, -0.6]
        # param = 16
        img_dist = imcontrastc(img, param)
        label = "contrast imbalance"
        level = param
        condition = "balance contrast"
    elif choice == 23:
        # param = random.randint(2, 10) # imcolorblock: amount (2, 10)
        param = 6 # range: [2, 4, 6, 8, 10]
        img_dist = imcolorblock(img, param)
        label = "color block"
        level = param
        condition = "remove color blocks"
    elif choice == 24:
        # param = random.uniform(0.01, 0.5) # impixelate: amount (0.01, 0.5)
        param = 0.1 # range: [0.01, 0.05, 0.1, 0.2, 0.5]
        img_dist = impixelate(img, param)
        label = "pixelate"
        level = param
        condition = "remove the pixelate distortion and make the image clear"
    elif choice == 25:
        # param = random.randint(20, 100) # imnoneccentricity: amount (20, 100)
        param = 60 # range: [20, 40, 60, 80, 100]
        img_dist = imnoneccentricity(img, param)
        label = "discontinuous"
        level = param
        condition = "make the image continuous"
    elif choice == 26:
        # param = random.uniform(0.05, 1) # imjitter: amount (0.05, 1)
        param = 0.2 # range: [0.05, 0.1, 0.2, 0.5, 1]
        img_dist = imjitter(img, param)
        label = "jitter"
        level = param
        condition = "remove the jitter distortion"
    elif choice == 27:
        img_dist = immosaic(img)
        label = "mosaic"
        level = 1
        condition = "remove the mosaic mask"
    elif choice == 28:
        # impainting_irregular
        max_angle = 2
        max_len = 50
        max_width = 15
        min_times=0
        max_times=10
        img_dist = impainting_irregular(img, max_angle, max_len, max_width, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask"
    elif choice == 29:
        # impainting_rectangle
        margin=10
        bbox_min_size=15
        bbox_max_size=50
        min_times=0
        max_times=5
        img_dist = impainting_rectangle(img, margin, bbox_min_size, bbox_max_size, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask"
    elif choice == 30:
        # imrain
        noise = 2500
        rain_len = 7
        rain_angle = 30
        rain_thickness = 3
        alpha = 0.9
        img_dist = imrain(img, noise, rain_len, rain_angle, rain_thickness, alpha)
        label = "rain streak"
        level = noise
        condition = "remove the rain streak"
    elif choice == 31:
        # imsnow
        noise = 500
        snow_len = 3
        snow_angle = 10
        snow_thickness = 1
        alpha = 0.9
        img_dist = imsnow(img, noise, snow_len, snow_angle, snow_thickness, alpha)
        label = "snow streak"
        level = noise
        condition = "remove the snow streak"
    else :
        img_dist = img
        label = "high-quality clean image."
        level = 0
        clean = True
        condition = "keep image unchanged."
        
    if clean:
        noise_label = label
    else:
        noise_label = "low-quality image with " + label + " distortion."

    # img_dist = np.clip(img_dist.astype(np.uint8),0,255) # make sure the image range
    return img_dist, noise_label, level, condition


def add_distortions_test_slight(choice, img):
    clean = False
    if choice == 1:
        # if random.random() > 0.75:
        #     param = random.randint(5, 96)   # imjpeg: quality_factor (8,96)
        # else:
        #     param = random.choice([10,20,30,40])   # imjpeg quality_factor: [10,20,30,40,50,60]
        param = 40  # range: [70, 43, 36, 24, 7]
        img_dist = imcompressjpeg(img, param)
        label = "jpeg compression"
        level = param
        condition = "remove the jpeg compression artifact"
    elif choice == 2:
        # param = random.uniform(0.1, 5)   # imblurgauss: sigma (0.1,5)
        param = 0.1  # range: [0.1, 0.5, 1, 2, 5]
        img_dist = imblurgauss(img, param)
        label = "gaussian blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 3:
        # param = random.uniform(1, 8)   # imblurlens: radius (1,8)
        param = 1  # range: [1, 2, 4, 6, 8]
        img_dist = imblurlens(img, param)
        label = "lens blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 4:
        # param = random.randint(12, 28)   # imblurmotion: kernel_size (12,28)
        param = 12  # range: [12, 16, 20, 24, 28]
        img_dist = imblurmotion(img, param)
        label = "motion blur"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 5:
        # param = random.uniform(1, 12)   # imcolordiffuse: amount (1,12)
        param = 1  # range: [1, 3, 6, 8, 12]
        img_dist = imcolordiffuse(img, param)
        label = "color diffuse"
        level = param
        condition = "make the image color normal"
    elif choice == 6:
        # param = random.uniform(1, 12)   # imcolorshift: amount (1,12)
        param = 1  # range: [1, 3, 6, 8, 12]
        img_dist = imcolorshift(img, param)
        label = "color shift"
        level = param
        condition = "make the image color normal"
    elif choice == 7:
        # param = random.uniform(0.4, -0.4)   # imcolorsaturate: amount (0.4,-0.4)
        param = 0.4  # range: [0.4, 0.2, 0.1, 0, -0.4]
        img_dist = imcolorsaturate(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal"
    elif choice == 8:
        # param = random.uniform(1, 9)   # imcolorsaturate2: amount (1,9)
        param = 1 # range: [1, 2, 3, 6, 9]
        img_dist = imcolorsaturate2(img, param)
        label = "color saturate"
        level = param
        condition = "make the image color normal"
    elif choice == 9:
        # param = random.uniform(0, 50)   # imnoisegauss: amount (0, 50)
        param = 15 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
        img_dist = imnoisegauss(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 10:
        # param = random.uniform(0.0001, 0.003)   # imnoisecolormap: amount (0.0001, 0.003)
        param = 0.0001    # range: [0.0001, 0.0005, 0.001, 0.002, 0.003]
        param = (param**0.5)*255.
        img_dist = imnoisecolormap(img, param)
        label = "gaussian noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 11:
        # param = random.uniform(0.001, 0.03) # imnoiseimpulse: amount (0.001, 0.03)
        param = 0.001 # range: [0.001, 0.005, 0.01, 0.02, 0.03]
        img_dist = imnoiseimpulse(img, param)
        label = "impulse noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 12:
        # param = random.uniform(0.001, 0.05)   # imnoisemultiplicative: amount (0.001, 0.05)
        param = 0.001    # range: [0.001, 0.005, 0.01, 0.02, 0.05]
        param = (param**0.5)*255.
        img_dist = imnoisemultiplicative(img, param)
        label = "multiplicative noise"
        level = param
        condition = "remove the noise distortion and make the image clear"
    elif choice == 13:
        # param = random.uniform(0, 50) # imdenoise: amount (0, 50)
        param = 15 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
        img_dist = imdenoise(img, param)
        label = "denoise"
        level = param
        condition = "make the image clear"
    elif choice == 14:
        # param = random.uniform(0.1, 1.1) # imcolorbrighten: amount (0.001, 0.03)
        param = 0.1 # range: [0.1, 0.2, 0.4, 0.7, 1.1]
        img_dist = imcolorbrighten(img, param)
        label = "over bright"
        level = param
        condition = "reduce the light in this image and make the image light normal"
    elif choice == 15:
        # param = random.uniform(0.05, 0.8) # imdenoise: amount (0.05, 0.8)
        param = 0.05 # range: [0.05, 0.1, 0.2, 0.4, 0.8]
        img_dist = imcolordarken(img, param)
        label = "over dark"
        level = param
        condition = "enhance the light in this image and make the image light normal"
    elif choice == 16:
        # param = random.uniform(0.15, -0.15) # imdenoise: amount (0.05, 0.8)
        param = 0 # range: [0.15, 0.08, 0, -0.08, -0.15]
        img_dist = immeanshift(img, param)
        label = "mean shift"
        level = param
        condition = "make the image color and light normal"
    elif choice == 17:
        # param = random.randint(2, 16)   # imresizedist2: kernel_size (2, 16)
        param = 2 # range: [2,3,4,8,16]
        img_dist = imresizedist2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 18:
        # param = random.randint(2, 16)   # imresizedist_bilinear2: kernel_size (2, 16)
        param = 2 # range: [2,3,4,8,16]
        img_dist = imresizedist_bilinear2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 19:
        # param = random.randint(2, 16)   # imresizedist_nearest2: kernel_size (2, 16)
        param = 2 # range: [2,3,4,5,6]
        img_dist = imresizedist_nearest2(img, param)
        label = "resize"
        level = param
        condition = "remove the pixelate distortion and make the image clear"
    elif choice == 20:
        # param = random.randint(2, 16)   # imresizedist_lanczos2: kernel_size (2, 16)
        param = 2 # range: [2,3,4,8,16]
        img_dist = imresizedist_lanczos2(img, param)
        label = "resize"
        level = param
        condition = "remove the blur distortion and make the image clear"
    elif choice == 21:
        # param = random.randint(1, 12) # imsharpenHi: amount (1, 12)
        param = 1 # range: [1, 2, 3, 6, 12]
        img_dist = imsharpenHi(img, param)
        label = "over sharpening"
        level = param
        condition = "smooth image"
    elif choice == 22:
        # param = random.uniform(0.3, -0.6) # imcontrastc: amount (0.05, 0.8)
        param = 0 # range: [0.3, 0.15, 0, -0.4, -0.6]
        # param = 16
        img_dist = imcontrastc(img, param)
        label = "contrast imbalance"
        level = param
        condition = "balance contrast"
    elif choice == 23:
        # param = random.randint(2, 10) # imcolorblock: amount (2, 10)
        param = 2 # range: [2, 4, 6, 8, 10]
        img_dist = imcolorblock(img, param)
        label = "color block"
        level = param
        condition = "remove color blocks"
    elif choice == 24:
        # param = random.uniform(0.01, 0.5) # impixelate: amount (0.01, 0.5)
        param = 0.01 # range: [0.01, 0.05, 0.1, 0.2, 0.5]
        img_dist = impixelate(img, param)
        label = "pixelate"
        level = param
        condition = "remove the pixelate distortion and make the image clear"
    elif choice == 25:
        # param = random.randint(20, 100) # imnoneccentricity: amount (20, 100)
        param = 20 # range: [20, 40, 60, 80, 100]
        img_dist = imnoneccentricity(img, param)
        label = "discontinuous"
        level = param
        condition = "make the image continuous"
    elif choice == 26:
        # param = random.uniform(0.05, 1) # imjitter: amount (0.05, 1)
        param = 0.05 # range: [0.05, 0.1, 0.2, 0.5, 1]
        img_dist = imjitter(img, param)
        label = "jitter"
        level = param
        condition = "remove the jitter distortion"
    elif choice == 27:
        img_dist = immosaic(img)
        label = "mosaic"
        level = 1
        condition = "remove the mosaic mask"
    elif choice == 28:
        # impainting_irregular
        max_angle = 2
        max_len = 25
        max_width = 10
        min_times=0
        max_times=10
        img_dist = impainting_irregular(img, max_angle, max_len, max_width, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask"
    elif choice == 29:
        # impainting_rectangle
        margin=10
        bbox_min_size=10
        bbox_max_size=25
        min_times=0
        max_times=5
        img_dist = impainting_rectangle(img, margin, bbox_min_size, bbox_max_size, min_times, max_times)
        label = "black mask"
        level = max_times
        condition = "remove the black mask"
    elif choice == 30:
        # imrain
        noise = 700
        rain_len = 7
        rain_angle = 30
        rain_thickness = 1
        alpha = 0.9
        img_dist = imrain(img, noise, rain_len, rain_angle, rain_thickness, alpha)
        label = "rain streak"
        level = noise
        condition = "remove the rain streak"
    elif choice == 31:
        # imsnow
        noise = 250
        snow_len = 3
        snow_angle = 10
        snow_thickness = 1
        alpha = 0.9
        img_dist = imsnow(img, noise, snow_len, snow_angle, snow_thickness, alpha)
        label = "snow streak"
        level = noise
        condition = "remove the snow streak"
    else :
        img_dist = img
        label = "high-quality clean image."
        level = 0
        clean = True
        condition = "keep image unchanged."
        
    if clean:
        noise_label = label
    else:
        noise_label = "low-quality image with " + label + " distortion."

    # img_dist = np.clip(img_dist.astype(np.uint8),0,255) # make sure the image range
    return img_dist, noise_label, level, condition







import time
if __name__ == '__main__':
    im = Image.open("test.png") # w,h; rgb
    print(im.size)
    img = cv2.imread("test.png")    # h,w,c; bgr
    print(img.shape)
    level = 4
    out_dir = 'test_add_distortion/specific'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 1. jpeg
    param = 7  # range: [70, 43, 36, 24, 7]
    img_dist = imcompressjpeg(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imcompressjpeg"+".png"), img_dist)
    # 2. blurgauss
    param = 5  # range: [0.1, 0.5, 1, 2, 5]
    img_dist = imblurgauss(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imblurgauss"+".png"), img_dist)
    # 3. imblurlens
    param = 8  # range: [1, 2, 4, 6, 8]
    img_dist = imblurlens(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imblurlens"+".png"), img_dist)
    # 4. imblurmotion
    param = 28  # range: [12, 16, 20, 24, 28]
    img_dist = imblurmotion(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imblurmotion"+".png"), img_dist)
    # 5. imblurmotion2
    param = 8  # range: [1, 2, 4, 6, 8]
    img_dist = imblurmotion2(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imblurmotion2"+".png"), img_dist)
    # 6. imcolordiffuse
    param = 12  # range: [1, 3, 6, 8, 12]
    img_dist = imcolordiffuse(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imcolordiffuse"+".png"), img_dist)
    # 7. imcolorshift
    param = 12  # range: [1, 3, 6, 8, 12]
    img_dist = imcolorshift(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imcolorshift"+".png"), img_dist)
    # 8. imcolorsaturate
    param = -0.4  # range: [0.4, 0.2, 0.1, 0, -0.4]
    img_dist = imcolorsaturate(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imcolorsaturate"+".png"), img_dist)
    # 9. imcolorsaturate2
    param = 9 # range: [1, 2, 3, 6, 9]
    img_dist = imcolorsaturate2(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imcolorsaturate2"+".png"), img_dist)
    # 10. imnoisegauss
    param = 25.5 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
    img_dist = imnoisegauss(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imnoisegauss"+".png"), img_dist)
    # 11. imnoisecolormap
    param = (0.003**0.5)*255. # range: [0.0001, 0.0005, 0.001, 0.002, 0.003]
    img_dist = imnoisecolormap(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imnoisecolormap"+".png"), img_dist)
    # 12. imnoiseimpulse
    param = 0.03 # range: [0.001, 0.005, 0.01, 0.02, 0.03]
    img_dist = imnoiseimpulse(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imnoiseimpulse"+".png"), img_dist)
    # 13. imnoisemultiplicative
    param = (0.05**0.5)*255. # range: [0.001, 0.005, 0.01, 0.02, 0.05]
    img_dist = imnoisemultiplicative(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imnoisemultiplicative"+".png"), img_dist)
    # 14. imdenoise
    param = 25.5 # range: [0.001, 0.002, 0.003, 0.005, 0.01]
    img_dist = imdenoise(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imdenoise"+".png"), img_dist)
    # 15. imcolorbrighten
    param = 1.1 # range: [0.1, 0.2, 0.4, 0.7, 1.1]
    img_dist = imcolorbrighten(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imcolorbrighten"+".png"), img_dist)
    # 16. imcolordarken
    param = 0.8 # range: [0.05, 0.1, 0.2, 0.4, 0.8]
    img_dist = imcolordarken(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imcolordarken"+".png"), img_dist)
    # 17. immeanshift
    param = -0.15 # range: [0.15, 0.08, 0, -0.08, -0.15]
    img_dist = immeanshift(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"immeanshift"+".png"), img_dist)
    # 18. imresizedist
    param = 16 # range: [2,3,4,8,16]
    # img_dist = imresizedist(img, param)
    img_dist = imresizedist2(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imresizedist"+".png"), img_dist)
    # 19. imresizedist_bilinear
    param = 16 # range: [2,3,4,8,16]
    # img_dist = imresizedist_bilinear(img, param)
    img_dist = imresizedist_bilinear2(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imresizedist_bilinear"+".png"), img_dist)
    # 20. imresizedist_nearest
    param = 6 # range: [2,3,4,5,6]
    # img_dist = imresizedist_nearest(img, param)
    img_dist = imresizedist_nearest2(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imresizedist_nearest"+".png"), img_dist)
    # 21. imresizedist_lanczos
    param = 16 # range: [2,3,4,8,16]
    # img_dist = imresizedist_lanczos(img, param)
    img_dist = imresizedist_lanczos2(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imresizedist_lanczos"+".png"), img_dist)
    # 22. imsharpenHi
    param = 12 # range: [1, 2, 3, 6, 12]
    img_dist = imsharpenHi(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imsharpenHi"+".png"), img_dist)
    # 23. imcontrastc
    param = -0.6 # range: [0.3, 0.15, 0, -0.4, -0.6]
    img_dist = imcontrastc(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imcontrastc"+".png"), img_dist)
    # 24. imcolorblock
    param = 10 # range: [2, 4, 6, 8, 10]
    img_dist = imcolorblock(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imcolorblock"+".png"), img_dist)
    # 25. impixelate
    param = 0.5 # range: [0.01, 0.05, 0.1, 0.2, 0.5]
    img_dist = impixelate(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"impixelate"+".png"), img_dist)
    # 26. imnoneccentricity
    param = 100 # range: [20, 40, 60, 80, 100]
    img_dist = imnoneccentricity(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imnoneccentricity"+".png"), img_dist)
    # 27. imjitter
    param = 1 # range: [0.05, 0.1, 0.2, 0.5, 1]
    img_dist = imjitter(img, param)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imjitter"+".png"), img_dist)
    # 28. immosaic
    img_dist = immosaic(img)
    cv2.imwrite(os.path.join(out_dir, "test_"+"immosaic"+".png"), img_dist)
    # 29. impainting_irregular
    max_angle=4
    max_len=60
    max_width=20
    min_times=0
    max_times=10
    img_dist = impainting_irregular(img, max_angle, max_len, max_width, min_times, max_times)
    cv2.imwrite(os.path.join(out_dir, "test_"+"impainting_irregular"+".png"), img_dist)
    # 30. impainting_rectangle
    margin=10
    bbox_min_size=30
    bbox_max_size=100
    min_times=0
    max_times=3
    img_dist = impainting_rectangle(img, margin, bbox_min_size, bbox_max_size, min_times, max_times)
    cv2.imwrite(os.path.join(out_dir, "test_"+"impainting_rectangle"+".png"), img_dist)
    # 31. imrain
    noise = 5000
    rain_len = 10
    rain_angle = 30
    rain_thickness = 7
    alpha = 0.9
    img_dist = imrain(img, noise, rain_len, rain_angle, rain_thickness, alpha)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imrain"+".png"), img_dist)
    # 32. imsnow
    noise = 500
    snow_len = 3
    snow_angle = 10
    snow_thickness = 1
    alpha = 0.9
    img_dist = imsnow(img, noise, snow_len, snow_angle, snow_thickness, alpha)
    cv2.imwrite(os.path.join(out_dir, "test_"+"imsnow"+".png"), img_dist)
    # 


    out_dir2 = 'test_add_distortion/random'
    if not os.path.exists(out_dir2):
        os.makedirs(out_dir2)
    # while 1:
    for i in range(32):
        time_start = time.time()
        img_dist,_,_,_ = add_distortions_random(i+1, img)
        # print(i+1)
        # print(img_dist.max())
        cv2.imwrite(os.path.join(out_dir2, "test_"+str(i+1)+".png"), img_dist)
        time_end = time.time()
        print('time cost: ', time_end-time_start, ' s')



