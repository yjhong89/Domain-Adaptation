import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops

class FlipGradient:
    def __init__(self):
        self.num_calls = 0
    
    # When instance of this class acts as a function
    '''
        Custom gradients in tensorflow : 
        https://uoguelph-mirg.github.io/tensorflow_gradients/
    '''
    def __call__(self, x, mag=1.0):
        custom_grad = 'GRL%d' % self.num_calls
        '''
            When adding new ops, tf.RegisterGradient to register a gradient function which computes 
            gradients with respect to input tensors given 'gradients with respect to output tensors and original operation'
            tf.RegisterGradient(op_name)
        '''
        @ops.RegisterGradient(custom_grad)
        def _flip_gradient(op, grad):
            return [tf.negative(grad) * mag]

        g = tf.get_default_graph()
        '''
            gradient_override_map(A,B) : 
                context manager states that use the gradient of A instead of  gradient of B
        '''
        # We want to flip gradient while back-propagating and do not want to do anything while forward passing
        with g.gradient_override_map({"Identity":custom_grad}):
            output = tf.identity(x)
             
        self.num_calls += 1
        return output


def conv2d(inp, output_dim, filter_len, stride, name=None, activation=None):
    with tf.variable_scope(name or 'conv2d'):
        weight = tf.get_variable('weight', [filter_len, filter_len, inp.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        # padding=SAME outputs same shape of input when stride is 1
        convolution = tf.nn.conv2d(inp, weight, strides=[1,stride, stride, 1], padding='SAME')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0))
        outputs = tf.add(convolution, bias)
        if activation:
            outputs = activation(outputs)
        return outputs


def fc(inp_, output_dim, name=None, activation=None):
    with tf.variable_scope(name or 'fc'):
        weight = tf.get_variable('weight', [inp_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0))
	weighted_sum = tf.matmul(inp_, weight)
        outputs = tf.add(weighted_sum, bias)
        if activation:
            outputs = activation(outputs)
        return outputs

def batch_generator(data, batch_size):
    perm_index = np.random.permutation(data[0].shape[0])
    shuffled_data = [d[perm_index] for d in data]

    batch_count = 0
    while True:
        if (batch_count+1) * batch_size > data[0].shape[0]:
            batch_count = 0
            perm_index = np.random.permutation(data[0].shape[0])
            shuffled_data = [d[perm_index] for d in shuffled_data]

        start_index = batch_count * batch_size
        end_index = (batch_count+1) * batch_size
        batch_count += 1
        # Use Generator and yield grammer in python for large dataset
        # Passing batches when next() is called
        yield [d[start_index:end_index] for d in data]


