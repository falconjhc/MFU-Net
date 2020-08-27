
import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from tensorflow.python.framework import ops


class OneHotRounding(Layer):
    def __init__(self, channels, **kwargs):
        super(OneHotRounding, self).__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        super(OneHotRounding, self).build(input_shape)

    def call(self, x, **kwargs):
        return onehotroundWithGrad(x,self.channels)

    def compute_output_shape(self, input_shape):
        return input_shape


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, name=None, grad=None):
    rnd_name  = 'PyFuncOneHotGrad' + str(np.random.randint(0, 1E+8))  # generate a unique name to avoid duplicates
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"one_hot": rnd_name, "arg_max": rnd_name, "round": rnd_name, "PyFunc": rnd_name}):
        # inp = tf.cast(tf.argmax(inp, axis=-1, name="arg_max"), tf.float32)
        # res = tf.one_hot(tf.cast(inp, tf.int64), depth=8, name=name)
        res = tf.py_func(func, inp, Tout, stateful=True, name=name)
        res[0].set_shape(inp[0].get_shape())
        return res


def onehotroundWithGrad(x, channels,name=None):
    with ops.name_scope(name, "onehotroundWithGrad", [x]) as name:
        onehot = lambda x: get_one_hot(np.argmax(x, axis=-1), channels).astype('float32')
        res = py_func(onehot, [x], [tf.float32], name=name, grad=_onehotroundWithGrad_grad)  # <-- here's the call to the gradient
        return res[0]

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def _onehotroundWithGrad_grad(op, grad):
    return grad * 1  # do whatever with gradient here (e.g. could return grad * 2 * x  if op was f(x)=x**2)