import tensorflow as tf
import cv2 
import pandas as pd 
import os
import numpy as np
import random
from augment import RandAugment

class DataSequence(tf.keras.utils.Sequence):
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
            self.augmenter = RandAugment()
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
        random.shuffle(self.all_data)
