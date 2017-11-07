import tensorflow as tf
import tensorbayes as tb
from extra_layers import basic_accuracy, scale_gradient, normalize_vector
from tensorbayes.layers import placeholder, constant, gaussian_sample
from tensorbayes.distributions import log_normal
from codebase.args import args
from pprint import pprint
exec "from designs import {:s} as des".format(args.design)
sigmoid_xent = tf.nn.sigmoid_cross_entropy_with_logits
softmax_xent = tf.nn.softmax_cross_entropy_with_logits
softmax_xent_two = tb.tfutils.softmax_cross_entropy_with_two_logits
import numpy as np

def generate_img():
    if args.Y == 2:
        nrow, ncol = 5, 20
    elif args.Y == 10:
        nrow, ncol = 1, 20

    z = np.random.randn(args.Y * nrow * ncol, 100)
    # z = np.tile(np.random.randn(nrow * ncol, 100), (args.Y, 1))
    y = np.tile(np.eye(args.Y), (nrow * ncol, 1))
    y = y.T.reshape(nrow * ncol * args.Y, -1)

    z, y = constant(z), constant(y)
    img = des.generator(z, y, phase=True, reuse=True)
    img = tf.reshape(img, [args.Y * nrow, ncol, 32, 32, 1])
    img = tf.reshape(tf.transpose(img, [0, 2, 1, 3, 4]), [1, 32 * args.Y * nrow, 32 * ncol, 1])
    img = (img + 1) / 2
    img = tf.clip_by_value(img, 0, 1)
    return img

def acgan():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    T = tb.utils.TensorDict(dict(
        sess = tf.Session(config=config),
        trg_x = placeholder((None, 32, 32, 1)),
        trg_y = placeholder((None, args.Y)),
        fake_z = placeholder((None, 100)),
        fake_y = placeholder((None, args.Y)),
        phase = placeholder((), tf.bool)
    ))

    fake_x = des.generator(T.fake_z, T.fake_y, T.phase)
    fake_logit, fake_y = des.discriminator(fake_x, T.phase)
    real_logit, real_y = des.discriminator(T.trg_x, T.phase, reuse=True)

    loss_gen_disc = 0.5 * tf.reduce_mean(
        sigmoid_xent(labels=tf.ones_like(real_logit), logits=real_logit) +
        sigmoid_xent(labels=tf.zeros_like(fake_logit), logits=fake_logit))
    loss_gen = tf.reduce_mean(sigmoid_xent(labels=tf.ones_like(fake_logit), logits=fake_logit))

    loss_class = tf.reduce_mean(softmax_xent(labels=T.trg_y, logits=real_y))
    loss_info = tf.reduce_mean(softmax_xent(labels=T.fake_y, logits=fake_y))

    # Optimizer
    loss_main = loss_gen + args.info * loss_info
    var_main = tf.get_collection('trainable_variables', 'gen')
    train_main = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main, var_list=var_main)

    loss_disc = loss_gen_disc + loss_info + args.cw * loss_class
    var_disc = tf.get_collection('trainable_variables', 'disc')
    train_disc = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_disc, var_list=var_disc)

    # Summarizations
    summary_disc = [tf.summary.scalar('gen/loss_gen_disc', loss_gen_disc),
                    tf.summary.scalar('class/loss_info', loss_info),
                    tf.summary.scalar('class/loss_class', loss_class)]
    summary_disc = tf.summary.merge(summary_disc)
    summary_main = tf.summary.scalar('gen/loss_gen', loss_gen)
    summary_image = tf.summary.image('image/gen', generate_img())

    # Saved ops
    c = tf.constant
    T.ops_print = [c('gen_disc'), loss_gen_disc,
                   c('gen'), loss_gen,
                   c('class'), loss_class,
                   c('info'), loss_info,]
    T.ops_disc = [summary_disc, train_disc]
    T.ops_main = [summary_main, train_main]
    T.ops_image = summary_image
    T.fake_x = fake_x

    return T
