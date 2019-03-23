#coding=utf-8
import tensorflow as tf

from config import Config
from data_processor import DataSet
from math import sqrt

class CharCNN(object):
    def __init__(self, l0, num_classes, conv_layers, fc_layers, l2_reg_lambda = 0.0):
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.l2_reg_lambda = l2_reg_lambda
        self.num_classes = num_classes
        self.l0 = l0
        self.input_x = tf.placeholder(tf.int32, [None, l0], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name = "input_y")
        self.dropout_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        self.init_embedding()
        self.init_layers()

    def init_embedding(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            train_data = DataSet(Config.train_data_source)
            self.W, _ = train_data.onehot_dic_build()
            self.x_image = tf.nn.embedding_lookup(self.W, self.input_x)
            self.x_flat = tf.expand_dims(self.x_image, -1)# batch_size * sequence_length * embedding_size * -1

    def init_layers(self):
        for i, cl in enumerate(self.conv_layers):
            with tf.name_scope("conv-layer%s" % (i + 1)):
                print ("Processing Convolution layer processing %d" % (i + 1))
                filter_width = self.x_flat.get_shape()[2].value
                filter_shape = [cl[1], filter_width, 1, cl[0]]
                s_dev = sqrt(cl[0] * cl[1])
                w_conv = tf.Variable(tf.random_uniform(filter_shape
                    , minval = -s_dev, maxval = s_dev, dtype = 'float32', name = 'w'))
                b_conv = tf.Variable(tf.random_uniform(shape = [cl[0]], minval = -s_dev, maxval = s_dev, name = 'b'))
                conv = tf.nn.conv2d(self.x_flat, w_conv, strides = [1, 1, 1, 1], padding = 'VALID', name = 'conv')
                h_conv = tf.nn.bias_add(conv, b_conv)
                if not cl[-1] is None:
                    ksize_shape = [1, cl[2], 1, 1]
                    h_pool = tf.nn.max_pool(
                        h_conv,ksize=ksize_shape,strides=ksize_shape,padding='VALID',name='pool')
                else:
                    h_pool = h_conv
                self.x_flat = tf.transpose(h_pool, [0, 1, 3, 2], name = "transpose")

        with tf.name_scope("reshape"):
            fc_dim = self.x_flat.get_shape()[1].value * self.x_flat.get_shape()[2].value
            self.x_flat = tf.reshape(self.x_flat, [-1, fc_dim])

        weights = [fc_dim] + self.fc_layers
        for i, fl in enumerate(self.fc_layers):
            with tf.name_scope("fc_layer-%s" % (i + 1)):
                print "MLP layer %s" % str(i + 1)
                stdv = 1 / sqrt(weights[i])
                w_fc = tf.Variable(tf.random_uniform(shape = [weights[i], fl], minval = -stdv, maxval = stdv), dtype = 'float32', name = 'w')
                b_fc = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype='float32', name='b')
                self.x_flat = tf.nn.relu(tf.matmul(self.x_flat, w_fc) + b_fc)
                with tf.name_scope('drop_out'):
                    self.x_flat = tf.nn.dropout(self.x_flat, self.dropout_prob)
        with tf.name_scope("output_layer"):
            print "Processing output"
            print self.x_flat.get_shape()
            stdv = 1 / sqrt(weights[-1])
            w_out = tf.Variable(tf.random_uniform([self.fc_layers[-1], self.num_classes], minval = -stdv, maxval = stdv), dtype = 'float32', name = 'W')
            b_out = tf.Variable(tf.random_uniform(shape = [self.num_classes], minval = -stdv, maxval = stdv), dtype = 'float32', name = 'b')
            self.y_pred = tf.nn.xw_plus_b(self.x_flat, w_out, b_out, name = 'y_pred')
            self.predictions = tf.argmax(self.y_pred, 1, name = 'predictions')
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")