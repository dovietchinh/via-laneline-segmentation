""" author: 
    name :              Do Viet Chinh
    personal email:     dovietchinh1998@mgail.com
    personal facebook:  https://www.facebook.com/profile.php?id=100005935236259
    VNOpenAI team:      vnopenai@gmail.com
    via team :          

date:
    26.3.2021
"""

# Selecting frame-work
FRAME_WORK = 'PYTORCH'#'TENSORFLOW'                 # must be in ['TENSORFLOW','PYTORCH','MXNET'],   
                                           # MXNET is not available on this repo, this feature will be update soon
USE_GPU = False


# Model Architecture

NAME_MODEL = 'unet'                                 # must be in ['unet','double-unet']
INPUT_SHAPE = 256                         # (256,256,3)

# Training 

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
INITIAL_EPOCH = 0
USE_MULTIPROCESSING = True
OPTIMIZER = 'Adam'
LOSS_NAME = 'DICE_LOSS'
METRIC_NAME = 'iou'
SHUFFLE = True
# Augmentaion 

N = 3
M = 10

# Demo params

PRE_TRAIN_MODEL_PATH = './models/pre_train_unet_tensorflow.h5'
IMG_DEMO_PATH = './images/demo_image.jpg'
