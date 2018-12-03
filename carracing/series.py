'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import numpy as np
import os
import tensorflow as tf
from tensorflow.python.data import Dataset
from vae.vae import ConvVAE, reset_graph
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATA_DIR = "record/20181130-233040"
SERIES_DIR = "series"
model_path_name = "tf_vae"

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)
    
def gen(filelist, path):
  for fn in filelist:
    o = np.load(os.path.join(path, fn))['obs']
    a = np.load(os.path.join(path, fn))['action']
    for i in range(o.shape[0]):
      yield (o[i,:,:,:], a[i,:])

def create_dataset(filelist, path, nrof_epochs, shuffle_buffer_size, batch_size):
  ds1 = Dataset.from_generator(lambda: gen(filelist, path), (tf.float32, tf.float32), (tf.TensorShape([64, 64, 3]), tf.TensorShape([3,])))
  ds2 = Dataset.range(10000000)
  ds = tf.data.Dataset.zip((ds1, ds2))
  ds = ds.repeat(nrof_epochs)
  ds = ds.prefetch(shuffle_buffer_size)
  ds = ds.batch(batch_size)
  return ds

# Hyperparameters for ConvVAE
z_size=32
batch_size=1000 # treat every episode as a batch of 1000!
learning_rate=0.0001
kl_tolerance=0.5

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]

reset_graph()

shuffle_buffer_size = 10000

reset_graph()

dataset = create_dataset(filelist, DATA_DIR, 1, shuffle_buffer_size, batch_size)

# Create an iterator over the dataset
iterator = dataset.make_one_shot_iterator()

(obs, action), index = iterator.get_next()

vae = ConvVAE(obs, z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=False,
              reuse=False,
              gpu_mode=True) # use GPU on batchsize of 1000 -> much faster

vae.load_json(os.path.join(model_path_name, 'vae.json'))

mu_dataset = []
logvar_dataset = []
action_dataset = []
try:
    print('Started')
    t = time.time()
    # Keep running next_batch till the Dataset is exhausted
    i = 0
    t = time.time()
    while True:
        mu, logvar, action_, index_ = vae.sess.run([vae.mu, vae.logvar, action, index])
        if not ((index_[1:] - index_[:-1])==1).all():
          raise ValueError('Examples are out-of-order')
        i += 1
        mu_dataset.append(mu.astype(np.float16))
        logvar_dataset.append(logvar.astype(np.float16))
        action_dataset.append(action_)
        print("step=%-8d time=%-8.4f batch_size=%-8d" % (i, (time.time()-t), mu.shape[0]))
        t = time.time()
        
except tf.errors.OutOfRangeError:
    print('Done!')


action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)
