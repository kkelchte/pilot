@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None):
  with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
    with arg_scope([conv2d, max_pool2d]):
        net = _squeeze(inputs, squeeze_depth)
        net = _expand(net, expand_depth)
      return net


def _squeeze(inputs, num_outputs):
  return conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def _expand(inputs, num_outputs):
  with tf.variable_scope('expand'):
    e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
    e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='3x3')
  return tf.concat([e1x1, e3x3], 1)

class Squeezenet(object):
  """Original squeezenet architecture for 224x224 images."""
  name = 'squeezenet'

  def __init__(self, args):
    self._num_classes = args.num_classes
    self._weight_decay = args.weight_decay
    self._batch_norm_decay = args.batch_norm_decay
    self._is_built = False

  def build(self, x, is_training):
    self._is_built = True
    with tf.variable_scope(self.name, values=[x]):
      with arg_scope(_arg_scope(is_training,
                                self._weight_decay,
                                self._batch_norm_decay)):
        return self._squeezenet(x, self._num_classes)

  @staticmethod
  def _squeezenet(images, num_classes=1000):
    net = conv2d(images, 96, [7, 7], stride=2, scope='conv1')
    net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
    net = fire_module(net, 16, 64, scope='fire2')
    net = fire_module(net, 16, 64, scope='fire3')
    net = fire_module(net, 32, 128, scope='fire4')
    net = max_pool2d(net, [3, 3], stride=2, scope='maxpool4')
    net = fire_module(net, 32, 128, scope='fire5')
    net = fire_module(net, 48, 192, scope='fire6')
    net = fire_module(net, 48, 192, scope='fire7')
    net = fire_module(net, 64, 256, scope='fire8')
    net = max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
    net = fire_module(net, 64, 256, scope='fire9')
    net = conv2d(net, num_classes, [1, 1], stride=1, scope='conv10')
    net = avg_pool2d(net, [13, 13], stride=1, scope='avgpool10')
    logits = tf.squeeze(net, [2], name='logits')
    return logits 