import numpy as np
from collections import namedtuple
import json
import tensorflow as tf

# hyperparameters for our model. I was using an older tf version, when HParams was not available ...

# controls whether we concatenate (z, c, h), etc for features used for car.
# MODE_ZCH = 0
# MODE_ZC = 1
# MODE_Z = 2
# MODE_Z_HIDDEN = 3 # extra hidden later
# MODE_ZH = 4

HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'max_seq_len',
                                         'input_seq_width',
                                         'output_seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                        ])

class CausalConv1D(tf.layers.Conv1D):
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )
       
    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)

class TemporalBlock(tf.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2, 
                 trainable=True, name=None, dtype=None, batch_size=32,
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )        
        self.dropout = dropout
        self.batch_size = tf.cast(batch_size, tf.int32)
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv1")
        self.conv2 = CausalConv1D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv2")
        self.down_sample = None

    
    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.layers.Dropout(self.dropout, [self.batch_size, tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(self.dropout, [self.batch_size, tf.constant(1), tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1, 
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = tf.layers.Dense(self.n_outputs, activation=None)
        self.built = True
    
    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = tf.contrib.layers.layer_norm(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)
      
class TemporalConvNet(tf.layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2,
                 trainable=True, name=None, dtype=None, batch_size=32,
                 activity_regularizer=None, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(
                    out_channels, kernel_size, strides=1, 
                    dilation_rate=dilation_size, batch_size=batch_size,
                    dropout=dropout, name="tblock_{}".format(i))
            )
    
    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs
            
# MDN-TCN model
class MDNTCN():
  def __init__(self, hps, gpu_mode=True, reuse=False):
    self.hps = hps
    with tf.variable_scope('mdn_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device("/cpu:0"):
          print("model using cpu")
          self.g = tf.get_default_graph()
          with self.g.as_default():
            self.build_model(hps)
      else:
        print("model using gpu")
        self.g = tf.get_default_graph()
        with self.g.as_default():
          self.build_model(hps)
    self.init_session()
    
  def build_model(self, hps):
    
    def tf_lognormal(y, mean, logstd):
      logSqrtTwoPi = np.log(np.sqrt(2.0 * np.pi))
      return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPi

    def mdn_loss(logmix, mean, logstd, y):
      v = logmix + tf_lognormal(y, mean, logstd)
      v = tf.reduce_logsumexp(v, 1, keepdims=True)
      return -tf.reduce_mean(v)

    def get_mdn_coef(output):
      logmix, mean, logstd = tf.split(output, 3, 1)
      logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
      return logmix, mean, logstd

    self.num_mixture = hps.num_mixture
    KMIX = self.num_mixture # 5 mixtures
    INWIDTH = hps.input_seq_width # 35 channels
    OUTWIDTH = hps.output_seq_width # 32 channels
    LENGTH = self.hps.max_seq_len # 1000 timesteps
    
    self.sequence_lengths = LENGTH # assume every sample has same length.
    self.input_x = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, INWIDTH])
    self.output_x = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, OUTWIDTH])
    
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      
    X = self.input_x
    is_training = self.hps.is_training == 1
    dropout = False
    nhid = 128
    levels = 6
    kernel_size = 8
    NOUT = OUTWIDTH * KMIX * 3

    conv_out = TemporalConvNet([nhid] * levels, kernel_size, dropout, batch_size=tf.shape(X)[0])(X, training=is_training)
    dense_out = tf.layers.dense(conv_out, NOUT, activation=None, kernel_initializer=tf.orthogonal_initializer())
    output = tf.reshape(dense_out, [-1, KMIX * 3])
      
    out_logmix, out_mean, out_logstd = get_mdn_coef(output)

    self.out_logmix = out_logmix
    self.out_mean = out_mean
    self.out_logstd = out_logstd

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.output_x,[-1, 1])

    loss = mdn_loss(out_logmix, out_mean, out_logstd, flat_target_data)

    self.cost = tf.reduce_mean(loss)

    if self.hps.is_training == 1:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      #gvs = optimizer.compute_gradients(self.cost)
      #capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var) for grad, var in gvs]
      #self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')
      self.train_op = optimizer.minimize(self.cost, self.global_step, name='train_step')

    # initialize vars
    self.init = tf.global_variables_initializer()
    
    t_vars = tf.trainable_variables()
    self.pl_dict = {}
    for var in t_vars:
        if var.name.startswith('mdn_rnn'):
            pshape = var.get_shape()
            pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
            assign_op = var.assign(pl)
            self.pl_dict[var] = (assign_op, pl)
    
  def init_session(self):
    """Launch TensorFlow session and initialize variables"""
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0 / 16
    self.sess = tf.Session(graph=self.g, config=config)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()
  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      for var in t_vars:
        if var.name.startswith('mdn_rnn'):
          param_name = var.name
          p = self.sess.run(var)
          model_names.append(param_name)
          params = np.round(p*10000).astype(np.int).tolist()
          model_params.append(params)
          model_shapes.append(p.shape)
    return model_params, model_shapes, model_names
  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up
    return rparam
  def set_random_params(self, stdev=0.5):
    rparam = self.get_random_model_params(stdev)
    self.set_model_params(rparam)
  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        if var.name.startswith('mdn_rnn'):
          pshape = tuple(var.get_shape().as_list())
          p = np.array(params[idx])
          assert pshape == p.shape, "inconsistent shape"
          assign_op, pl = self.pl_dict[var]
          self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
          idx += 1
  def load_json(self, jsonfile='rnn.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)
  def save_json(self, jsonfile='rnn.json'):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1

def sample_sequence(sess, s_model, hps, init_z, actions, temperature=1.0, seq_len=1000):
  # generates a random sequence using the trained model
  
  OUTWIDTH = hps.output_seq_width
  INWIDTH = hps.input_seq_width

  prev_x = np.zeros((1, 1, OUTWIDTH))
  prev_x[0][0] = init_z

  prev_state = sess.run(s_model.initial_state)

  '''
  if prev_data is not None:
    # encode the previous data into the hidden state first
    for i in range(prev_data.shape[0]):
      prev_x[0][0] = prev_data[i]
      feed = {s_model.input_x: prev_x, s_model.initial_state:prev_state}
      [next_state] = sess.run([s_model.final_state], feed)
      prev_state = next_state
  '''

  strokes = np.zeros((seq_len, OUTWIDTH), dtype=np.float32)

  for i in range(seq_len):
    input_x = np.concatenate((prev_x, actions[i].reshape((1, 1, 3))), axis=2)
    feed = {s_model.input_x: input_x, s_model.initial_state:prev_state}
    [logmix, mean, logstd, next_state] = sess.run([s_model.out_logmix, s_model.out_mean, s_model.out_logstd, s_model.final_state], feed)

    # adjust temperatures
    logmix2 = np.copy(logmix)/temperature
    logmix2 -= logmix2.max()
    logmix2 = np.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

    mixture_idx = np.zeros(OUTWIDTH)
    chosen_mean = np.zeros(OUTWIDTH)
    chosen_logstd = np.zeros(OUTWIDTH)
    for j in range(OUTWIDTH):
      idx = get_pi_idx(np.random.rand(), logmix2[j])
      mixture_idx[j] = idx
      chosen_mean[j] = mean[j][idx]
      chosen_logstd[j] = logstd[j][idx]

    rand_gaussian = np.random.randn(OUTWIDTH)*np.sqrt(temperature)
    next_x = chosen_mean+np.exp(chosen_logstd)*rand_gaussian

    strokes[i,:] = next_x

    prev_x[0][0] = next_x
    prev_state = next_state

  return strokes

def rnn_init_state(rnn):
  return rnn.sess.run(rnn.initial_state)

def rnn_next_state(rnn, z, a, prev_state):
  input_x = np.concatenate((z.reshape((1, 1, 32)), a.reshape((1, 1, 3))), axis=2)
  feed = {rnn.input_x: input_x, rnn.initial_state:prev_state}
  return rnn.sess.run(rnn.final_state, feed)

