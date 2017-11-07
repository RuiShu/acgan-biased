import os
import sys
import argparse
from codebase import args as codebase_args
from pprint import pprint
import tensorflow as tf

# Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--trg',    type=str,   default='mnistbias',    help="Trg data")
parser.add_argument('--design', type=str,   default='v1',     help="design")
parser.add_argument('--info',   type=float, default=1,      help="Info weight")
parser.add_argument('--cw',     type=float, default=1,      help="Cw weight")
parser.add_argument('--lr',     type=float, default=3e-4,      help="Learning rate")
parser.add_argument('--run',    type=int,   default=999,       help="Run index")
parser.add_argument('--logdir', type=str,   default='log',       help="Run index")
codebase_args.args = args = parser.parse_args()

# Set Y
if args.trg == 'mnist32':
    args.Y = 10
else:
    args.Y = 2

pprint(vars(args))

from codebase.models.acgan import acgan
from codebase.train import train
from codebase.utils import get_data

# Make model name
setup = [
    ('model={:s}',  'acgan'),
    ('trg={:s}',  args.trg),
    ('info={:.1e}',   args.info),
    ('cw={:.1e}',   args.cw),
    ('run={:d}',   args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in setup])
print "Model name:", model_name

M = acgan()
M.sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

trg = get_data(args.trg)
y_prior = [1. / args.Y] * args.Y

train(M, trg,
      saver=saver,
      has_disc=True,
      add_z=True,
      y_prior=y_prior,
      model_name=model_name)
