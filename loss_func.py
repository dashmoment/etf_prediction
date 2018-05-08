import tensorflow as tf

def l2loss(x,y):   
    loss = tf.reduce_mean(tf.nn.l2_loss(x - y))
    return loss 

def l1loss(x,y):   
    loss = tf.reduce_mean(tf.losses.absolute_difference(y,x))
    return loss 

def cross_entropy_loss(x,y):   
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x))
    return loss 