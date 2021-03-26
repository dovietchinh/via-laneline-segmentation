""" author: 
    name :              Do Viet Chinh
    personal email:     dovietchinh1998@mgail.com
    personal facebook:  https://www.facebook.com/profile.php?id=100005935236259
    VNOpenAI team:      vnopenai@gmail.com
    via team :          

date:
    26.3.2021
"""


import numpy as np
import cv2
import os

import math


from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout,
    ShiftScaleRotate
)
crop_size = (256-32, 256-32)
size = (256, 256)
x_min = 10
y_min = 10
x_max = -x_min + size[0]
y_max = -y_min + size[1]
ops = { 
        'CenterCrop' : CenterCrop(p=1, height=crop_size[0], width=crop_size[1]),
        'Crop' : Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
       # RandomRotate90(p=1),
        #Transpose(p=1),
        'ElasticTransform': ElasticTransform(
                p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        'GridDistortion':GridDistortion(p=1),
        #OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
        #'VerticalFlip': VerticalFlip(p=1),
        'HorizontalFlip': HorizontalFlip(p=1),
        'RandomBrightnessContrast': RandomBrightnessContrast(p=1),
        'RandomGamma' : RandomGamma(p=1),
        'HueSaturationValue': HueSaturationValue(p=1),
        'RGBShift': RGBShift(p=1),
        'RandomBrightness': RandomBrightness(p=1),
        'RandomContrast': RandomContrast(p=1),
        'MotionBlur': MotionBlur(p=1, blur_limit=7),
        'MedianBlur': MedianBlur(p=1, blur_limit=9),
        'GaussianBlur':GaussianBlur(p=1, blur_limit=9),
        'GaussNoise': GaussNoise(p=1),
        'ChannelShuffle':ChannelShuffle(p=1),
        'CoarseDropout': CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32),     
        'ShiftScaleRotate': ShiftScaleRotate(p =1,shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REPLICATE)
}

class RandAugment():
    def __init__(self,N=3,M=10):
        self.N = N
        self.M = M
    
    def get_random_ops(self,):
        n = np.random.randint(1,self.N+1)
        ops_random = np.random.choice( list(ops.keys()), n)
        return ops_random
    
    def __call__(self,img,mask):
        img_aug = img.copy()
        mask_aug = mask.copy()
        ops_random= self.get_random_ops()
        for name in ops_random:
            aug = ops[name]
            augmented = aug(image=img_aug, mask=mask_aug)
            img_aug = augmented['image']
            mask_aug = augmented['mask']
            if img_aug.shape[0] !=256:
                img_aug = cv2.resize(img_aug,(256,256))
                mask_aug = cv2.resize(mask_aug,(256,256))
        return img_aug,mask_aug        

if __name__ =='__main__':
    x = cv2.imread('data/train/new_images/train_00000.jpg', cv2.IMREAD_COLOR)
    y = cv2.imread('data/train/new_masks/train_00000.png', cv2.IMREAD_GRAYSCALE)  
    x_aug = x.copy()
    y_aug = y.copy()

    #ops_random = np.random.choice( ops, 4)
    #for aug in ops_random:
    for name in ops:
        aug = ops[name]
        print(aug)
        
        augmented = aug(image=x_aug, mask=y_aug)
        x_aug = augmented['image']
        y_aug = augmented['mask']
        if (x_aug.shape[0]!=256):
            x_aug = cv2.resize(x_aug,(256,256))
            y_aug = cv2.resize(y_aug,(256,256))
        cv2.imshow('a',x_aug)
        cv2.imshow('b',y_aug)
        k = cv2.waitKey(0)
        if k==ord('q'):
            break
        print(x_aug.shape,y_aug.shape)
    cv2.destroyAllWindows()

