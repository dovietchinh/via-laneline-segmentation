""" author: 
    name :              Do Viet Chinh
    personal email:     dovietchinh1998@mgail.com
    personal facebook:  https://www.facebook.com/profile.php?id=100005935236259
    VNOpenAI team:      vnopenai@gmail.com
    via team :          

date:
    26.3.2021
"""
import tensorflow as tf
import cv2 
import pandas as pd 
import os
import numpy as np
import random
from augment import RandAugment
import sys 
path_ = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_+'/../')
from configs import config
class DataSequenceTensorFlow(tf.keras.utils.Sequence):
    """DataSequence : 

    Args:
        tf ([type]): [description]
    """


    def __init__(self, data_folder, mask_folder, batch_size=32, phase='val'):
        try:
            assert phase in ['train','val','test'], "Invalid keywork, phase must be in 'train','val' or 'test'"
        except AssertionError as msg: 
            print(msg)

        self.batch_size = batch_size
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.phase = phase
        if phase == 'train':
            self.augmenter = RandAugment(config.N,config.M)
        self.data_names = os.listdir(data_folder)
        self.mask_names = list(map(lambda x : x.replace('.jpg','.png'),self.data_names))
        self.all_data = list(zip(self.data_names,self.mask_names))
    def __len__(self,):
        return len(self.all_data)//self.batch_size

    def __getitem__(self,index):
        current_batch = self.all_data[index*self.batch_size : (index+1)*self.batch_size]      
        data = []
        labels = []
        for img_path,mask_path in current_batch:
            img = cv2.imread(self.data_folder + img_path, )
            mask = cv2.imread(self.mask_folder + mask_path)
            if (img is None) or (mask is None):
                print('Unable to read this sample: ',img_path.replace('.jpg',''))
                continue
            if self.phase =='train':
                img,mask = self.augmenter(img,mask)
            data.append(img)
            labels.append(mask)
            
        data = tf.convert_to_tensor(data, tf.float32)
        labels = tf.convert_to_tensor(labels, tf.float32)
        data = data/255.
        labels = labels/255.
        return data,labels
    
    def on_epoch_end(self,):
        if config.SHUFFLE:
            random.shuffle(self.all_data)
        pass


import torch

class DataSeuqenceTorch(torch.utils.data.Dataset):
    def __init__(self,data_folder, mask_folder, phase = 'train'):
        try:
            assert phase in ['train', 'val', 'test'], "Invalid keyworks, phase must be in ['train','val','test']"
        except Exception as msg:
            print(msg)
        self.data_folder = data_folder
        self.mask_folder = mask_folder
        self.data_names = os.listdir(data_folder)
        self.mask_names = list(map(lambda x : x.replace('.jpg','.png'),self.data_names))
        self.phase = phase 
        if phase == 'train':
            self.augmenter = RandAugment(config.N,config.M)
        self.all_data = list(zip(self.data_names,self.mask_names))
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self,index):
        data = []
        labels = []
        img_path,mask_path = self.all_data[index]
        img = cv2.imread(os.path.join(self.data_folder,img_path), cv2.IMREAD_COLOR )
        mask = cv2.imread(os.path.join(self.mask_folder,mask_path), cv2.IMREAD_COLOR )
        if (img is None) or (mask is None):
            print('Unable to read this sample: ',img_path.replace('.jpg',''))
            img = np.zeros(3,112,112)
            mask = np.zeros(1,112,112)
        else:
            img = np.transpose(img,axes=[2,0,1])
            mask = np.transpose(mask,axes=[2,1,0])
            mask = mask[:1,:,:]
        if self.phase =='train':
            img,mask = self.augmenter(img,mask)
        img = img.astype('float32')/255.
        mask = mask.astype('float32')/255.       
        return img,mask
