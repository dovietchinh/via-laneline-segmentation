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
    import numpy as np
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.backend import epsilon
    class DiceLoss(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            y_true = Flatten()(y_true)
            y_pred = Flatten()(y_pred)
            intersection = tf.reduce_sum(y_true*y_pred)
            return 1 - ((2*intersection + epsilon()) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon()))

    """def focal_loss(y_true,y_pred):
        y_true = Flatten()(y_true)
        y_pred = Flatten()(y_pred)
        loss = alpha * y_true * """


    def iou(y_true, y_pred):
        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + epsilon()) / (union + epsilon())
            x = x.astype(np.float32)
            return x
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)




if config.FRAME_WORK == 'PYTORCH':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    EPS = 1e-6
    def iou_pytorch(y_true, y_pred):
        intersection = (y_true * y_pred).sum(axis=[1,2,3])
        union = y_true.sum(axis=[1,2,3]) + y_pred.sum(axis=[1,2,3]) - intersection
        x = (intersection + EPS )/ (union + EPS)
        
        return x
    def make_one_hot(input, num_classes):
        """Convert class index tensor to one hot encoding tensor.
        Args:
            input: A tensor of shape [N, 1, *]
            num_classes: An int of number of class
        Returns:
            A tensor of shape [N, num_classes, *]
        """
        shape = np.array(input.shape)
        shape[1] = num_classes
        shape = tuple(shape)
        result = torch.zeros(shape)
        result = result.scatter_(1, input.cpu(), 1)

        return result


    class BinaryDiceLoss(nn.Module):
        """Dice loss of binary class
        Args:
            smooth: A float number to smooth loss, and avoid NaN error, default: 1
            p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
            predict: A tensor of shape [N, *]
            target: A tensor of shape same with predict
            reduction: Reduction method to apply, return mean over batch if 'mean',
                return sum if 'sum', return a tensor of shape [N,] if 'none'
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        def __init__(self, smooth=1, p=2, reduction='mean'):
            super(BinaryDiceLoss, self).__init__()
            self.smooth = smooth
            self.p = p
            self.reduction = reduction

        def forward(self, predict, target):
            assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
            intersection = torch.sum(target*predict,axis=[1,2,3])
            result =  1 - ((2*intersection + EPS) / (torch.sum(target,axis=[1,2,3]) + torch.sum(target,axis=[1,2,3]) + EPS))
            return result.sum()

        #    if self.reduction == 'mean':
        #        return loss.mean()
        #    elif self.reduction == 'sum':
        #        return loss.sum()
        #    elif self.reduction == 'none':
        #        return loss
        #    else:
        #        raise Exception('Unexpected reduction {}'.format(self.reduction))


    class DiceLossTorch(nn.Module):
        """Dice loss, need one hot encode input
        Args:
            weight: An array of shape [num_classes,]
            ignore_index: class index to ignore
            predict: A tensor of shape [N, C, *]
            target: A tensor of same shape with predict
            other args pass to BinaryDiceLoss
        Return:
            same as BinaryDiceLoss
        """
        def __init__(self, weight=None, ignore_index=None, **kwargs):
            super(DiceLossTorch, self).__init__()
            self.kwargs = kwargs
            self.weight = weight
            self.ignore_index = ignore_index

        def forward(self, predict, target):
            assert predict.shape == target.shape, 'predict & target shape do not match'
            dice = BinaryDiceLoss(**self.kwargs)
            total_loss = 0
            predict = F.softmax(predict, dim=1)

            for i in range(target.shape[1]):
                if i != self.ignore_index:
                    dice_loss = dice(predict[:, i], target[:, i])
                    if self.weight is not None:
                        assert self.weight.shape[0] == target.shape[1], \
                            'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                        dice_loss *= self.weights[i]
                    total_loss += dice_loss

            return total_loss/target.shape[1]

    class DiceLoss_2(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(DiceLoss_2, self).__init__()

        def forward(self, inputs, targets, smooth=100):
            
            #comment out if your model contains a sigmoid or equivalent activation layer
            #inputs = F.sigmoid(inputs)       
            
            #flatten label and prediction tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            intersection = (inputs * targets).sum()                            
            dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
            return 1 - dice
            #return 1 - dice