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
import argparse
parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--model', default='./models/pre_train_unet.h5',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--img_path', default='./images/demo_image.jpg', type=str, help='Whether use origin image size to evaluate')
args = parser.parse_args()

def main():
    model = tf.keras.models.load_model(args.model, compile=False)
    img = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(256,256))
    img_show = img.copy()
    img = np.expand_dims(img,axis=0)
    img_pred = img.astype('float32')/255.
    output = model(img).numpy()
    output = (output + 0.5).astype('uint8')*255
    output = np.concatenate([output[0]]*3,axis=-1)
    cv2.imshow('a',output)
    cv2.imshow('b',img_show)
    cv2.waitKey(0)

if __name__ =='__main__':
    main()