from codebase.models.extra_layers import leaky_relu, wndense, noise, wnconv2d, wnconv2d_transpose, scale_gradient
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorbayes.layers import dense, conv2d, conv2d_transpose, upsample, avg_pool, max_pool, batch_norm, instance_norm
from codebase.args import args

dropout = tf.layers.dropout

def discriminator(x, phase, reuse=None):
    with tf.variable_scope('disc/gan', reuse=reuse):
        with arg_scope([wnconv2d, wndense], activation=leaky_relu), \
             arg_scope([noise], phase=phase):

            x = dropout(x, rate=0.2, training=phase)
            x = wnconv2d(x, 64, 3, 2)

            x = dropout(x, training=phase)
            x = wnconv2d(x, 128, 3, 2)

            x = dropout(x, training=phase)
            x = wnconv2d(x, 256, 3, 2)

            x = dropout(x, training=phase)
            x = wndense(x, 1024)

            q = dense(x, args.Y, activation=None, bn=False)
            d = dense(x, 1, activation=None, bn=False)

    return d, q

def generator(x, y, phase, reuse=None):
    with tf.variable_scope('gen', reuse=reuse):
        with arg_scope([dense], bn=True, phase=phase, activation=tf.nn.relu), \
             arg_scope([conv2d_transpose], bn=True, phase=phase, activation=tf.nn.relu):

            if y is not None:
                x = tf.concat([x, y], 1)

            x = dense(x, 4 * 4 * 512)
            x = tf.reshape(x, [-1, 4, 4, 512])
            x = conv2d_transpose(x, 256, 5, 2)
            x = conv2d_transpose(x, 128, 5, 2)
            x = wnconv2d_transpose(x, 1, 5, 2, bn=False, activation=tf.nn.tanh, scale=True)

    return x
