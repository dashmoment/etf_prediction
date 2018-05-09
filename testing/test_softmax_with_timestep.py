import numpy as np
import tensorflow as tf



x = tf.placeholder(tf.float32, [None, 4, 3])
y = tf.placeholder(tf.float32, [None, 4, 3])

label = np.array([[[1,0,0],[0,1,0], [1,0,0], [0,0,1]],[[1,0,0],[0,1,0], [1,0,0], [0,0,1]]],dtype=np.float32)
test = np.array([[[1,1,1],[1,2,1], [1,3,2], [1,1,1]],[[1,1,1],[1,2,1], [1,3,2], [1,1,1]]],dtype=np.float32)
#x = tf.placeholder(tf.float32, [None, 3])
#test = np.array([[1.2,1.3,1.4]], dtype=np.float32)
softmax_out = tf.nn.softmax(x,dim=-1)
argmax_out = tf.argmax(softmax_out, axis=-1)

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x)
mean_loss = tf.reduce_mean(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    soft, arg = sess.run([softmax_out, argmax_out], feed_dict={x:test})
    ml, l = sess.run([mean_loss, loss], feed_dict={x:test,y:label})