'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import time
import utils

from vae.vae import reset_graph
from rnn.tcn import HyperParams, MDNTCN

os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

DATA_DIR = os.path.join("series", '20181205-222042')
subdir = utils.gettime()
model_save_path = os.path.join("tf_rnn", subdir)
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)
  
initial_z_save_path = os.path.join("tf_initial_z", subdir)
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

def random_batch():
  indices = np.random.permutation(N_data)[0:batch_size]
  mu = data_mu[indices]
  logvar = data_logvar[indices]
  action = data_action[indices]
  s = logvar.shape
  z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  return z, action

def default_hps():
  return HyperParams(num_steps=8000,
                     max_seq_len=999, # train on sequences of 1000 (so 999 + teacher forcing shift)
                     input_seq_width=35,    # width of our data (32 + 3 actions)
                     output_seq_width=32,    # width of our data is 32
                     nrof_hidden_units=128,    # number of units in the hidden layers
                     batch_size=25,   # minibatch sizes
                     nrof_levels=6,
                     kernel_size=8,
                     dropout=0.2,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     is_training=1)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, is_training=0)

raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# load preprocessed data
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]
max_seq_len = hps_model.max_seq_len

N_data = len(data_mu) # should be 10k
batch_size = hps_model.batch_size

# save 1000 initial mu and logvars:
initial_mu = np.copy(data_mu[:1000, 0, :]*10000).astype(np.int).tolist()
initial_logvar = np.copy(data_logvar[:1000, 0, :]*10000).astype(np.int).tolist()
with open(os.path.join("tf_initial_z", "initial_z.json"), 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

reset_graph()
rnn = MDNTCN(hps_model)

# train loop:
hps = hps_model
t = time.time()
train_cost_list = []
for local_step in range(hps.num_steps):

  step = rnn.sess.run(rnn.global_step)
  curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

  raw_z, raw_a = random_batch()
  inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2)
  outputs = raw_z[:, 1:, :] # teacher forcing (shift by one predictions)

  feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
  train_cost, train_step, _ = rnn.sess.run([rnn.cost, rnn.global_step, rnn.train_op], feed)
  train_cost_list.append(train_cost)
  if (step%20==0 and step > 0):
    print("step=%d, time=%.4f, lr=%.6f, cost=%.4f" % (step, time.time()-t, curr_learning_rate, np.mean(train_cost_list)))
    t = time.time()
    train_cost_list = []

# save the model (don't bother with tf checkpoints json all the way ...)
rnn.save_json(os.path.join(model_save_path, "rnn.json"))
