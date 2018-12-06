'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
import tensorflow as tf
import numpy as np
import time
import utils
from tensorflow.python.data import Dataset 

from vae.vae import ConvVAE, reset_graph

def count_length_of_filelist(filelist, datadir):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join(datadir, filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

def create_dataset(filelist, path, nrof_epochs, shuffle_buffer_size, batch_size):
  def gen(filelist, path):
    for fn in filelist:
      data = np.float32(np.load(os.path.join(path, fn))['obs']) / 255.0
      for i in range(data.shape[0]):
        yield data[i,:,:,:]
        
  ds = Dataset.from_generator(lambda: gen(filelist, path), tf.float32, tf.TensorShape([64, 64, 3]))
  ds = ds.repeat(nrof_epochs)
  ds = ds.prefetch(shuffle_buffer_size)
  ds = ds.shuffle(shuffle_buffer_size)
  ds = ds.batch(batch_size)
  return ds

# Hyperparameters for ConvVAE
z_size=32
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = os.path.join('record', '20181203-220648')

model_save_path = os.path.join("tf_vae", utils.gettime())
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)
  
vae_filename = os.path.join(model_save_path, 'vae.json')

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
#print("total number of images:", count_length_of_filelist(filelist, DATA_DIR))  # 6659360 images

nrof_epochs = NUM_EPOCH
shuffle_buffer_size = 100000

reset_graph()

dataset = create_dataset(filelist, DATA_DIR, nrof_epochs, shuffle_buffer_size, batch_size)

# Create an iterator over the dataset
iterator = dataset.make_one_shot_iterator()

x = iterator.get_next()
sess = tf.Session()

vae = ConvVAE(x, z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)


try:
    print('Started training')
    t = time.time()
    # Keep running next_batch till the Dataset is exhausted
    while True:
      train_loss, r_loss, kl_loss, train_step, _ = vae.sess.run([vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op])
      if (train_step+1) % 500 == 0:
        print("step=%-8d time=%-8.4f loss=%-8.2f rec_loss=%-8.2f kl_loss=%-8.2f" % (train_step+1, (time.time()-t)/500, train_loss, r_loss, kl_loss))
        t = time.time()
      if (train_step+1) % 5000 == 0:
        vae.save_json(vae_filename)
        
except tf.errors.OutOfRangeError:
    print('Done!')


# finished, final model:
vae.save_json(vae_filename)
