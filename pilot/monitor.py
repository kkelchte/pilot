import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

"""
Monitor the learning process and write values away
"""
class Monitor(object):

  def __init__(self, session, prefix='monitor', device='/gpu:0'):
    self.writer = tf.summary.FileWriter(FLAGS.summary_dir+FLAGS.log_tag, sess.graph)
    # create saver for checkpoints
    self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.build_summaries()

  def save(self, logfolder):
    '''save a checkpoint'''
    self.saver.save(self.sess, logfolder+'/my-model', global_step=tf.train.global_step(self.sess, self.global_step))
    
  def add_summary_var(self, name):
    var_name = tf.Variable(0., name=name)
    self.summary_vars[name]=var_name
    self.summary_ops[name] = tf.summary.scalar(name, var_name)
    
  def build_summaries(self): 
    self.summary_vars = {}
    self.summary_ops = {}
    for t in ['train', 'test', 'val']:
      for l in ['total', 'control', 'depth']:
        name='Loss_{0}_{1}'.format(t,l)
        self.add_summary_var(name)
    for d in ['current','furthest']:
      for t in ['train', 'test']:
        for w in ['','sandbox','forest','canyon','esat_corridor_v1', 'esat_corridor_v2']:
          name = 'Distance_{0}_{1}'.format(d,t)
          if len(w)!=0: name='{0}_{1}'.format(name,w)
          self.add_summary_var(name)
      
    if FLAGS.auxiliary_depth and FLAGS.plot_depth:
      name="depth_predictions"
      dep_images = tf.placeholder(tf.uint8, [1, 400, 400, 3])
      # dep_images = tf.placeholder(tf.float32, [1, 400, 400, 3])
      self.summary_vars[name]=dep_images
      self.summary_ops[name]=tf.summary.image(name, dep_images)
    
  def summarize(self, sumvars):
    '''write summary vars with ops'''
    if self.writer:
      feed_dict={self.summary_vars[key]:sumvars[key] for key in sumvars.keys()}
      sum_op = tf.summary.merge([self.summary_ops[key] for key in sumvars.keys()])
      summary_str = self.sess.run(sum_op, feed_dict=feed_dict)
      self.writer.add_summary(summary_str,  tf.train.global_step(self.sess, self.global_step))
      self.writer.flush()
