""" 
**Author: **
    name :              Do Viet Chinh
    personal email:     dovietchinh1998@mgail.com
    personal facebook:  https://www.facebook.com/profile.php?id=100005935236259
    VNOpenAI team:      vnopenai@gmail.com
    via team :          

date:
    26.3.2021
"""

import sys
import os 
path_import = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_import+'/../')
from configs import config

if config.FRAME_WORK == 'TENSORFLOW':
    import tensorflow as tf
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import *
#-------------------TENSORFLOW CODE---------------------------------------



def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = Multiply()([init, se])
    return x
def conv_block(inputs, filters):
    x = inputs
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    return x
def encoder1(inputs):
    skip_connections = []
    model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
    for name in names:
        skip_connections.append(model.get_layer(name).output)
    output = model.get_layer("block5_conv4").output
    return output, skip_connections
 
def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]
    skip_connections.reverse()
    x = inputs
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        x = conv_block(x, f)
    return x
# def encoder2(inputs):
#     skip_connections = []
#     output = DenseNet121(include_top=False, weights='imagenet')(inputs)
#     model = tf.keras.models.Model(inputs, output)
#
#     names = ["input_2", "conv1/relu", "pool2_conv", "pool3_conv"]
#     for name in names:
#         skip_connections.append(model.get_layer(name).output)
#     output = model.get_layer("pool4_conv").output
#
#     return output, skip_connections
def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs
    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)
    return x, skip_connections
def decoder2(inputs, skip_1, skip_2):
    num_filters = [256, 128, 64, 32]
    skip_2.reverse()
    x = inputs
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_1[i], skip_2[i]])
        x = conv_block(x, f)
    return x
def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x
def Upsample(tensor, size):
    """Bilinear upsampling"""
    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)
    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)
def ASPP(x, filter):
    shape = x.shape
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)
    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x) # ASPP
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)
    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)
    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)
    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)
    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    return y
def create_double_u_net(input_shape):
    inputs = Input(input_shape)
    x, skip_1 = encoder1(inputs)
    x = ASPP(x, 64)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)
    x = inputs * outputs1
    x, skip_2 = encoder2(x)
    x = ASPP(x, 64)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = output_block(x)
    outputs = Concatenate()([outputs1, outputs2])
    model = Model(inputs, outputs)
    return model


def create_unet(input_shape):
    inputs = tf.keras.Input(input_shape)    
    conv1 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv1)

    pool2 = MaxPool2D(pool_size=(2,2), padding='same')(conv2)
    conv3 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(pool2)
    conv4 = Conv2D(128, kernel_size=3, activation='relu', padding='same')(conv3)

    pool4 = MaxPool2D(pool_size=(2,2), padding='same')(conv4)
    conv5 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(pool4)
    conv6 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(conv5)

    pool6 = MaxPool2D(pool_size=(2,2))(conv6)
    conv7 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(pool6)
    conv8 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(conv7)

    pool8 = MaxPool2D(pool_size=(2,2))(conv8)
    conv9 = Conv2D(1024, kernel_size=(3,3), activation='relu', padding='same')(pool8)
    conv10 = Conv2D(1024, kernel_size=(3,3), activation='relu', padding='same')(conv9)

    up11 = UpSampling2D(size=(2, 2))(conv10)
    up11 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(up11)
    up11 = Concatenate(axis=-1)([conv8,up11])
    conv12 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(up11)
    conv13 = Conv2D(512, kernel_size=(3,3), activation='relu', padding='same')(conv12)

    up14 = UpSampling2D((2,2))(conv13)
    up14 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(up14)
    up14 = Concatenate(axis=-1)([conv6,up14])
    conv15 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(up14)
    conv16 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(conv15)

    up17 = UpSampling2D((2,2))(conv16)
    up17 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(up17)
    up17 = Concatenate(axis=-1)([conv4,up17])
    conv18 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(up17)
    conv19 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(conv18)

    up20 = UpSampling2D((2,2))(conv19)
    up20 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(up20)
    up20 = Concatenate(axis=-1)([conv2,up20])
    conv21 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(up20)
    conv22 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv21)
    conv23 = Conv2D(1, kernel_size=(1,1), activation='sigmoid', padding = 'same')(conv22)

    output = conv23
    model = tf.keras.Model(inputs,output)

    return model

#-------------------PYTORCH CODE---------------------------------------

if config.FRAME_WORK == 'PYTORCH':
    import torch
    import torch.nn.functional as F 


class UNet(torch.nn.Module):
    def __init__(self,):
        super(UNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=(1,1))
        self.conv1_relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1,1))
        self.conv2_relu = torch.nn.ReLU(inplace=False)

        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(1,1))
        self.conv3_relu = torch.nn.ReLU(inplace=False)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(1,1))
        self.conv4_relu = torch.nn.ReLU(inplace=False)

        self.pool4 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=(1,1))
        self.conv5_relu = torch.nn.ReLU(inplace=False)
        self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1,1))
        self.conv6_relu = torch.nn.ReLU(inplace=False)

        self.pool6 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=(1,1))
        self.conv7_relu = torch.nn.ReLU(inplace=False)
        self.conv8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=(1,1))
        self.conv8_relu = torch.nn.ReLU(inplace=False)

        self.pool8 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv9 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=(1,1))
        self.conv9_relu = torch.nn.ReLU(inplace=False)
        self.conv10 = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=(1,1))
        self.conv10_relu = torch.nn.ReLU(inplace=False)
        
        self.up11 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.up11_2 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=(1,1))
        self.up11_2_relu = torch.nn.ReLU(inplace=False)
        self.up11_3 = 'concat'
        self.conv12 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=(1,1))
        self.conv12_relu = torch.nn.ReLU(inplace=False)
        self.conv13 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=(1,1))
        self.conv13_relu = torch.nn.ReLU(inplace=False)

        self.up14 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.up14_2 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=(1,1))
        self.up14_2_relu = torch.nn.ReLU(inplace=False)
        self.up14_3 = 'concat'
        self.conv15 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=(1,1))
        self.conv15_relu = torch.nn.ReLU(inplace=False)
        self.conv16 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1,1))
        self.conv16_relu = torch.nn.ReLU(inplace=False)

        self.up17 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.up17_2 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=(1,1))
        self.up17_2_relu = torch.nn.ReLU(inplace=False)
        self.up17_3 = 'concat'
        self.conv18 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=(1,1))
        self.conv18_relu = torch.nn.ReLU(inplace=False)
        self.conv19 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(1,1))
        self.conv19_relu = torch.nn.ReLU(inplace=False)

        self.up20 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.up20_2 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(1,1))
        self.up20_2_relu = torch.nn.ReLU(inplace=False)
        self.up20_3 = 'concat'
        self.conv21 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(1,1))
        self.conv21_relu = torch.nn.ReLU(inplace=False)
        self.conv22 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1,1))
        self.conv22_relu = torch.nn.ReLU(inplace=False)
        self.conv23 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=(1,1))
        self.conv23_sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        conv1 = self.conv1(x)
        conv1 = self.conv1_relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.conv2_relu(conv2)

        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        conv3 = self.conv3_relu(conv3)
        conv4 = self.conv4(conv3)
        conv4 = self.conv4_relu(conv4)

        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        conv5 = self.conv5_relu(conv5)
        conv6 = self.conv6(conv5)
        conv6 = self.conv6_relu(conv6)

        pool6 = self.pool6(conv6)
        conv7 = self.conv7(pool6)
        conv7 = self.conv7_relu(conv7)
        conv8 = self.conv8(conv7)
        conv8 = self.conv8_relu(conv8)

        pool8 = self.pool8(conv8)
        conv9 = self.conv9(pool8)
        conv9 = self.conv9_relu(conv9)
        conv10 = self.conv10(conv9)
        conv10 = self.conv10_relu(conv10)

        up11 = self.up11(conv10)
        up11 = self.up11_2(up11)
        up11 = self.up11_2_relu(up11)
        up11 = torch.cat([conv8,up11], axis=1)
        conv12 = self.conv12(up11)
        conv12 = self.conv12_relu(conv12)
        conv13 = self.conv13(conv12)
        conv13 = self.conv13_relu(conv13)

        up14 = self.up14(conv13)
        up14 = self.up14_2(up14)
        up14 = self.up14_2_relu(up14)
        up14 = torch.cat([conv6,up14], axis=1)
        conv15 = self.conv15(up14)
        conv15 = self.conv15_relu(conv15)
        conv16 = self.conv16(conv15)
        conv16 = self.conv16_relu(conv16)

        up17 = self.up17(conv16)
        up17 = self.up17_2(up17)
        up17 = self.up17_2_relu(up17)
        up17 = torch.cat([conv4,up17], axis=1)
        conv18 = self.conv18(up17)
        conv18 = self.conv18_relu(conv18)
        conv19 = self.conv19(conv18)
        conv19 = self.conv19_relu(conv19)

        up20 = self.up20(conv19)
        up20 = self.up20_2(up20)
        up20 = self.up20_2_relu(up20)
        up20 = torch.cat([conv2,up20], axis=1)
        conv21 = self.conv21(up20)
        conv21 = self.conv21_relu(conv21)
        conv22 = self.conv22(conv21)
        conv22 = self.conv22_relu(conv22)
        conv23 = self.conv23(conv22)
        conv23 = self.conv23_sigmoid(conv23)

        return conv23


def build_model_tensorflow(inputs_shape, name_model = 'unet'):
    try:
        assert name_model in ['unet', 'double-unet'], "Invalid Keyword, name_model must be in ['unet','double-unet']"
    except Exception as msg:
        print(msg)
        
    if name_model =='unet':
        return create_unet(inputs_shape)
    if name_model =='double_unet':
        return create_double_u_net(inputs_shape)

def build_model_pytorch( name_model = 'unet'):
    try:
        assert name_model in ['unet', 'double-unet'], "Invalid Keyword, name_model must be in ['unet','double-unet']"
    except Exception as msg:
        print(msg)
        
    if name_model =='unet':
        return UNet()
    if name_model =='double_unet':
        print(" Double have'nt implement on PyTorch option yet, we will update soon")
        return None