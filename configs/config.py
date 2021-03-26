""" author: 
    name :              Do Viet Chinh
    personal email:     dovietchinh1998@mgail.com
    personal facebook:  https://www.facebook.com/profile.php?id=100005935236259
    VNOpenAI team:      vnopenai@gmail.com
    via team :          

date:
    26.3.2021
"""

# Model Architecture

NAME_MODEL = 'unet'                                 # must be in ['unet','double-unet']
INPUT_SHAPE = 256                         # (256,256,3)

# Training 

EPOCHS = 100
INITIAL_EPOCH = 0
USE_MULTIPROCESSING = True
OPTIMIZER = 'Adam'
LOSS_NAME = 'DICE_LOSS'
METRIC_NAME = 'iou'

# Augmentaion 

N = 3
M = 10
