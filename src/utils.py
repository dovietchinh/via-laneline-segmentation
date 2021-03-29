""" author: 
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
import_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(import_path+'/../')
from configs import config
if config.FRAME_WORK == 'TENSORFLOW':
        
    import tensorflow as tf
    from metrics import DiceLoss,iou

    def lr_function(epoch,current_lr):
        if epoch <=3:
            return 5e-4
        if 3<= epoch <10:
            return 1e-3
        if epoch >=10:
            return current_lr *tf.math.exp(-0.1)

    def get_loss(loss_name):
        try:
            assert loss_name in ['DICE_LOSS'], "Invalid Keyword, loss_name must be in ['DICE_LOSS']"
        except Exception as msg:
            print(msg)
        
        if loss_name =='DICE_LOSS':
            return DiceLoss()

    def get_metrics(metrics_name):
        try:
            assert metrics_name in ['iou'], "Invalid Keyword, loss_name must be in ['iou']"
        except Exception as msg:
            print(msg) 
        
        if metrics_name =='iou':
            return iou