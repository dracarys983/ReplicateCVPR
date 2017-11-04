import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.framework as framework
import tensorflow.contrib.rnn as rnn

from vgg import *

def common_arg_scope(weight_decay=0.00004,
                     batch_norm_decay=0.9997,
                     batch_norm_epsilon=0.001):
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope

def get_pretrained_model_feats(inputs, scopename='', is_training=True):
    # VGG 19 for feature extraction
    scope = vgg_arg_scope()
    with slim.arg_scope(scope):
        _, end_points = vgg_19(inputs)
        features = end_points['vgg_19/conv5/conv5_1']           # 14 x 14 x 512
        print features
        restore_vars = framework.get_variables_to_restore(exclude=['global_step', 'fc6', 'fc7', 'fc8'])

    tvars = []

    return features, restore_vars, tvars

def get_temporal_mean_pooled_feats(inputs, is_training=True):
    # Temporal Average pooling
    with tf.variable_scope('temporal_mean_pool'):
        inputs = tf.nn.relu(inputs, name='SquashInput')
        pooled_features = slim.avg_pool2d(inputs, (14, 1), stride=1, padding='VALID', scope='AvgPool_14x1')
        features = slim.flatten(pooled_features)
    tvars = framework.get_variables('temporal_mean_pool')

    return features, tvars

def get_classifier_logits(inputs, num_classes, is_training=True, lscope='', reuse=None):
    # Primary Classifier
    scope = common_arg_scope()
    with slim.arg_scope(scope):
        with tf.variable_scope(lscope, reuse=reuse):
            plogits = slim.fully_connected(inputs, 512, activation_fn=None, scope='preLogits')
            plogits = layers.batch_norm(plogits, is_training=is_training, scope='FCBatchNorm')
            rlogits = tf.nn.relu(plogits)
            dropout = slim.dropout(rlogits, 0.5, is_training=is_training, scope='FCDropout')
            logits = slim.fully_connected(dropout, num_classes, activation_fn=None, scope='finalLogits')

    tvars = framework.get_variables(lscope)
    return logits, tvars
