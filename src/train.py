""" author: 
    name :              Do Viet Chinh
    personal email:     dovietchinh1998@mgail.com
    personal facebook:  https://www.facebook.com/profile.php?id=100005935236259
    VNOpenAI team:      vnopenai@gmail.com
    via team :          

date:
    26.3.2021
"""
from posix import NGROUPS_MAX
import tensorflow as tf
import os 
from datasequence import DataSequence
from model import build_model
from utils import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from metrics import iou,DiceLoss
import pathlib
import warnings
from metrics import iou,DiceLoss
warnings.filterwarnings('ignore')
import sys 
import_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(import_path+'/../')
from configs import config

EPOCHS = config.EPOCHS
INITIAL_EPOCH = config.INITIAL_EPOCH
INPUT_SHAPE = config.INPUT_SHAPE
NAME_MODEL = config.NAME_MODEL
USE_MULTIPROCESSING = config.USE_MULTIPROCESSING
OPTIMIZER = config.OPTIMIZER
LOSS_NAME = config.LOSS_NAME
METRIC_NAME = config.METRIC_NAME

data_folder_train = './data/train/new_images/'
mask_folder_train = './data/train/new_masks/'
data_folder_val = './data/val/new_images/'
mask_folder_val = './data/val/new_masks/'
data_sequence_train = DataSequence(data_folder_train, mask_folder_train, batch_size=32, phase='train')
data_sequence_val = DataSequence(data_folder_val, mask_folder_val, batch_size=32, phase='val')


checkpoint_path = './checkpoint/'
log_path = './log/'

pathlib.Path(checkpoint_path).mkdir(exist_ok=True, parents=True)

mc = ModelCheckpoint(filepath=os.path.join(
    checkpoint_path, "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=1)
tb = TensorBoard(log_dir=log_path, write_graph=True)
lr_scheduler = LearningRateScheduler(lr_function)


model = build_model((INPUT_SHAPE,INPUT_SHAPE,3), name_model = NAME_MODEL)

loss_function = get_loss(LOSS_NAME)
metric = get_metrics(METRIC_NAME)

model.compile(OPTIMIZER, loss=loss_function, metrics=[iou])

model.fit(data_sequence_train,
                        epochs=EPOCHS,
                        initial_epoch=INITIAL_EPOCH,
                        validation_data=data_sequence_val,
                        use_multiprocessing=USE_MULTIPROCESSING,
                        callbacks=[mc,tb,lr_scheduler,],
                         verbose=1 )