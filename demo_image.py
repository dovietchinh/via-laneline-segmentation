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
import numpy as np 
import cv2 
from configs import config


def main_tensorflow():
    model = tf.keras.models.load_model(config.PRE_TRAIN_MODEL_PATH, compile=False)
    img = cv2.imread(config.IMG_DEMO_PATH, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(256,256))
    img_show = img.copy()
    img = np.expand_dims(img,axis=0)
    img_pred = img.astype('float32')/255.
    output = model(img).numpy()
    output = (output + 0.5).astype('uint8')*255
    output = np.concatenate([output[0]]*3,axis=-1)
    cv2.imshow('predict result',output)
    cv2.imshow('input image',img_show)
    cv2.waitKey(0)
def main_pytorch():
    print('Have no pre-train pytorch model ')
    pass

if __name__ =='__main__':
    if config.FRAME_WORK == 'TENSORFLOW':
        main_tensorflow()
    if config.FRAME_WORK == 'PYTORCH':
        main_pytorch()