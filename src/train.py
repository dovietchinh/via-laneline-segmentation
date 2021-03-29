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
import os 
from datasequence import DataSeuqenceTorch
from utils import *
import pathlib
import warnings
warnings.filterwarnings('ignore')
import sys 
import_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(import_path+'/../')
from configs import config
import numpy as np
import random



#---------------------------TENSORFLOW CODE-------------------------------------
def main_tensorflow():
    data_sequence_train = DataSequenceTensorFlow(data_folder_train, mask_folder_train, batch_size=32, phase='train')
    data_sequence_val = DataSequenceTensorFlow(data_folder_val, mask_folder_val, batch_size=32, phase='val')


    checkpoint_path = './checkpoint/'
    log_path = './log/'

    pathlib.Path(checkpoint_path).mkdir(exist_ok=True, parents=True)

    mc = ModelCheckpoint(filepath=os.path.join(
        checkpoint_path, "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=1)
    tb = TensorBoard(log_dir=log_path, write_graph=True)
    lr_scheduler = LearningRateScheduler(lr_function)


    model = build_model_tensorflow((INPUT_SHAPE,INPUT_SHAPE,3), name_model = NAME_MODEL)

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

#---------------------------PYTORCH CODE-------------------------------------
def main_pytorch():

    data_sequence_train = DataSeuqenceTorch(data_folder_train, mask_folder_train, phase ='train')
    data_sequence_val = DataSeuqenceTorch(data_folder_val, mask_folder_val, phase='val')

    train_data_loader = torch.utils.data.DataLoader(data_sequence_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    val_data_loader = torch.utils.data.DataLoader(data_sequence_val, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    dataloader_dic = {"train":train_data_loader,"val": val_data_loader}
    
    model = build_model_pytorch(name_model=NAME_MODEL)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)#, momentum=0.9)
    criterior = BinaryDiceLoss()

    for epoch in range(INITIAL_EPOCH,EPOCHS):
        print("Epoch {}/{}".format(epoch,EPOCHS))
        for phase in ["train","val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            epoch_loss = 0.0
            epoch_corrects = 0
            if(epoch == 0) and (phase == "train"):
                continue
            for inputs, labels in tqdm(dataloader_dic[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(inputs)
                    loss = criterior(outputs,labels)
                    _,preds = torch.max(outputs,1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item()*inputs.size(0)
                    epoch_corrects+=torch.sum(preds==labels.data)
            epoch_loss = epoch_loss/len(dataloader_dic[phase].dataset)
            epoch_acc = epoch_corrects.double()/len(dataloader_dic[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase,epoch_loss,epoch_acc))
 
if __name__ =='__main__':

    FRAME_WORK = config.FRAME_WORK  
    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    INITIAL_EPOCH = config.INITIAL_EPOCH
    INPUT_SHAPE = config.INPUT_SHAPE
    NAME_MODEL = config.NAME_MODEL
    USE_MULTIPROCESSING = config.USE_MULTIPROCESSING
    OPTIMIZER = config.OPTIMIZER
    LOSS_NAME = config.LOSS_NAME
    METRIC_NAME = config.METRIC_NAME
    SHUFFLE = config.SHUFFLE
    USE_GPU = config.USE_GPU 
    data_folder_train = './data/train/new_images/'
    mask_folder_train = './data/train/new_masks/'
    data_folder_val = './data/val/new_images/'
    mask_folder_val = './data/val/new_masks/'

    try:
        assert FRAME_WORK in ['TENSORFLOW', 'PYTORCH'],"Invalid keyword FRAME_WORK"
    except Exception as msg:
        print(msg)

    if FRAME_WORK == 'TENSORFLOW':

        import tensorflow as tf
        from datasequence import DataSequenceTensorFlow
        from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
        from metrics import iou,DiceLoss
        from model import build_model_tensorflow
        import random
        import numpy as np
        tf.random.set_seed(1)
        np.random.seed(1)
        random.seed(1)
        if USE_GPU == True:
            with tf.device('gpu'):
                main_tensorflow()
        else:
            with tf.device('cpu'):
                main_tensorflow()

    if FRAME_WORK == 'PYTORCH':
        import torch.optim
        from model import build_model_pytorch
        from datasequence import DataSeuqenceTorch
        import torch
        from metrics import DiceLossTorch,BinaryDiceLoss
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        from tqdm import tqdm
        if USE_GPU == True:
            device = 'cuda'
        else:
            device = 'cpu'
        main_pytorch()