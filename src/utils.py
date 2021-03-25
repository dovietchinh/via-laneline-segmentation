import tensorflow as tf





def lr_function(epoch,current_lr):
    if epoch <=3:
        return 5e-4
    if 3<= epoch <10:
        return 1e-3
    if epoch >=10:
        return current_lr *tf.math.exp(-0.1)