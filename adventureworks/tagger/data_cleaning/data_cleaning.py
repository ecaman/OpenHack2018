from PIL import Image
import cv2
import numpy as np

def load_resize_image(path_to_img):
    '''
    Load an image and resize it to 128x128x3 format
    input:
        - path_to_img (string): local path to Image (.png, .jpg, .jpeg)
    output:
        - img (PIL.Image): Image in corrected size
    '''
    img = Image.open(path_to_img)
    
    if img.size[0] > img.size[1]:
        white = np.zeros((img.size[0] - img.size[1], img.size[0],  3)) + 255
        img = np.vstack((np.array(img)[:,:,:3], white))
    elif img.size[0] < img.size[1]:
        white = np.zeros((img.size[1], img.size[1] - img.size[0], 3)) + 255
        img = np.hstack((np.array(img)[:,:,:3], white))
    else:
        img = np.array(img)[:,:,:3]
    basewidth = 128
    img = Image.fromarray(np.uint8(img))
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img

def exagerate_contrast(img):
    '''
    Exagerate contrast of resized image
    input:
        - img (PIL.Image): an image that will be casted into numpy array
    output:
        - img (PIL.Image): an image in which contrast are exagerated.
    '''
    hist,bins = np.histogram(np.array(img).flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]

    img = Image.fromarray(img2)
    return img

def exagerate_contrast_2(img):
    '''
    Exagerate contrast of resized image
    input:
       - img (PIL.Image): an image that will be casted into numpy array
    output:
       - new_img (PIL.Image): an image in which contrast are exagerated.
    '''
    arr = np.array(img)
    for i in range(3):
        minval = np.percentile(arr[...,i], 5)
        maxval = np.percentile(arr[...,i], 95)
        #print(minval, maxval)
        if minval != maxval:
            arr[...,i] = np.subtract(arr[...,i], minval, out=arr[...,i], casting="unsafe")
            arr[...,i] = np.multiply(arr[...,i], 255.0/(maxval-minval), out=arr[...,i], casting="unsafe")
    new_img = Image.fromarray(arr.astype('uint8'),'RGB')
    return new_img
    