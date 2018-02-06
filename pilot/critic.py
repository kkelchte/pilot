import tensorflow as tf
from model import Model

FLAGS = tf.app.flags.FLAGS
# Apply batch normalization on batch state input
tf.app.flags.DEFINE_boolean("bn_critic", True, "Apply batch normalization on batch state input for critic network.")

"""
Build basic critic model inherited from model.
"""
class Critic(Model):
  """Critic Model: inherited model mapping state input to action output"""
  
  def __init__(self,  session, input_size, action_size, output_size, num_actor_vars=0, prefix='', device='/gpu:0', hidden_size=1, lr=0.001, tau=0.001, initializer=tf.random_uniform_initializer(-0.03, 0.03)):
    self.tau = tau
    self.hidden_size = hidden_size
    self.action_size = action_size
    
    Model.__init__(self,  session, input_size, output_size, prefix=prefix, device=device, lr=lr, initializer=initializer)
    
    # define actor network and save placeholders & tensors
    self.inputs, self.action, self.outputs = self.define_network()
    self.network_params = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if prefix in v.name and "target" not in v.name]
    #self.network_params = tf.trainable_variables()[num_actor_vars:]#keep params for updating target network and the critic variables come after the actor variables

    # define target networks
    self.target_inputs, self.target_action, self.target_outputs = self.define_network(pref="target_")
    
    #self.target_network_params = tf.trainable_variables()[num_actor_vars+len(self.network_params):]
    self.target_network_params = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if prefix in v.name and "target" in v.name]
    
    # Op for periodically updating target network with online network weights
    self.update_target_network_params = [ self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau)+                tf.mul(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]
    
    
    self.define_loss()
    self.define_train()
    
    # Op for calculating gradients of output towards actions at the input (and not state)
    self.action_grads = tf.gradients(self.outputs, self.action)
    
    
    
    
  def define_network(self, pref=""):
    '''build the network and set the tensors
    First layer: state input to hidden size output and relu activation
    Second layer: relu activation of( linear combo of hidden activation of first layer + linear combo of action input )
    Third layer: linear combo of hidden activation of second layer
    '''
    with tf.device(self.device):
      inputs = tf.placeholder(tf.float32, shape = [None, self.input_size])
      action = tf.placeholder(tf.float32, shape = [None, self.action_size])
      bias_initializer=tf.random_uniform_initializer(-0.003, 0.003)
      
      if FLAGS.bn_critic:
        inputs_n = tf.contrib.layers.batch_norm(inputs, decay=0.9, scope=self.prefix+"_"+pref+"/bn", center=True, scale=True, updates_collections=None, is_training=self.is_training, reuse=None)
      else:
        inputs_n = inputs
      
      with tf.variable_scope(pref+self.prefix+"_layer_1"):
        weights = tf.get_variable("weights", [self.input_size, self.hidden_size], initializer=self.initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, weights)
        biases = tf.get_variable('biases', [self.hidden_size], initializer=bias_initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, biases)
        hidden_activations = tf.nn.relu(tf.matmul(inputs_n, weights))
      
      #Add action in second layer
      with tf.variable_scope(pref+self.prefix+"_layer_2"):
        weights_s = tf.get_variable("weights_state", [self.hidden_size, self.hidden_size], initializer=self.initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, weights_s)
        biases_s = tf.get_variable('weight_biases', [self.hidden_size], initializer=bias_initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, biases_s)
        state_output = tf.matmul(hidden_activations, weights_s) + biases_s
        weights_a = tf.get_variable("weights", [self.action_size, self.hidden_size], initializer=self.initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, weights_a)
        biases_a = tf.get_variable('biases', [self.hidden_size], initializer=bias_initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, biases_a)
        action_output = tf.matmul(action, weights_a) + biases_a
        hidden_activations = tf.nn.relu(state_output+action_output)
      
      with tf.variable_scope(pref+self.prefix+"_layer_out"):
        weights = tf.get_variable("weights", [self.hidden_size, self.output_size], initializer=self.initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, weights)
        biases = tf.get_variable('biases', [self.output_size], initializer=bias_initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, biases)
        outputs = tf.matmul(hidden_activations, weights) + biases
      
    return inputs, action, outputs
  
  def update_target_network(self):
    self.sess.run(self.update_target_network_params)
    
  def forward_target(self, inputs, action):
    return self.sess.run(self.target_outputs, feed_dict={self.target_inputs: inputs, self.target_action: action, self.is_training: False})
  
  def backward(self, inputs, action, targets):
    '''run backward pass applying gradients
    '''
    return self.sess.run([self.outputs, self.train], feed_dict={self.inputs: inputs, self.action: action, self.targets: targets, self.is_training: True})
  
  def action_gradients(self, inputs, actions): 
    '''get gradients of output towards actions at the input (and not the state input)
    '''
    return self.sess.run(self.action_grads, feed_dict={
        self.inputs: inputs,
        self.action: actions, 
        self.is_training: False
    })
