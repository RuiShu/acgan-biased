import tensorflow as tf
import tensorbayes as tb
from codebase.args import args
from utils import delete_existing
import os
import sys
import numpy as np

def update_dict(M, feed_dict, trg=None, add_z=False, bs=100, y_prior=None):
    if trg:
        trg_x, trg_y = trg.train.next_batch(bs)
        feed_dict.update({M.trg_x: trg_x, M.trg_y: trg_y})

    if add_z:
        y = np.random.multinomial(1, y_prior, size=bs)
        z = np.random.randn(bs, 100)
        feed_dict.update({M.fake_y: y, M.fake_z: z})

def train(M, trg=None, has_disc=True, add_z=False,
          saver=None, model_name=None, y_prior=None):
    log_dir = os.path.join(args.logdir, model_name)
    delete_existing(log_dir)
    train_writer = tf.summary.FileWriter(log_dir)

    if saver:
        model_dir = os.path.join('checkpoints', model_name)
        delete_existing(model_dir)
        os.makedirs(model_dir)

    bs = 64
    iterep = 500
    n_epoch = 1000
    epoch = 0
    feed_dict = {M.phase: 1}

    if trg: print "Trg size:", trg.train.images.shape
    print "Batch size:", bs
    print "Iterep:", iterep
    print "Total iterations:", n_epoch * iterep
    print "Log directory:", log_dir

    for i in xrange(n_epoch * iterep):
        # Discriminator
        if has_disc:
            update_dict(M, feed_dict, trg, add_z, bs, y_prior)
            summary, _ = M.sess.run(M.ops_disc, feed_dict)
            train_writer.add_summary(summary, i + 1)

        # Main
        update_dict(M, feed_dict, trg, add_z, bs, y_prior)

        summary, _ = M.sess.run(M.ops_main, feed_dict)
        train_writer.add_summary(summary, i + 1)
        train_writer.flush()

        end_epoch, epoch = tb.utils.progbar(i, iterep,
                                            message='{}/{}'.format(epoch, i),
                                            display=args.run >= 999)

        if end_epoch:
            print_list = M.sess.run(M.ops_print, feed_dict)
            print_list += ['epoch', epoch]
            print print_list
            sys.stdout.flush()

        if end_epoch and epoch % 10 == 0:
            if hasattr(M, 'ops_image'):
                summary = M.sess.run(M.ops_image, feed_dict)
                train_writer.add_summary(summary, i + 1)

        if saver and (i + 1) % 20000 == 0:
            path = saver.save(M.sess,
                              os.path.join(model_dir, 'model'),
                              global_step=i + 1)
            print "Saving model to {:s}".format(path)
            sys.stdout.flush()
