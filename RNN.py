#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import sys
import argparse
import data_helpers

POS_PATH = "/home/mil/tonkou/11_Dataset/1_NegPos/Reddit_crowdsourcinvg/Result/Raw_txt/Task1_positive_1217.txt"
NEG_PATH = "/home/mil/tonkou/11_Dataset/1_NegPos/Reddit_crowdsourcinvg/Result/Raw_txt/Task1_negative_1217.txt"

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", POS_PATH, "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", NEG_PATH, "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("rnn_hidden", 100, "Hidden Unit Size (default: 100)")
tf.flags.DEFINE_integer("num_epochs",10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 140, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 140, "Save model after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print "\nParameters:"
for attr, value in sorted(FLAGS.__flags.items()):
    print "{}={}".format(attr.upper(), value)
print ""

x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
x_text_ = np.array(x_text)
x_shuffled_text = x_text_[shuffle_indices]
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index],x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index],y_shuffled[dev_sample_index:]
sequence_length = x_train.shape[1]
batches = data_helpers.batch_iter(zip(x_train,y_train), FLAGS.batch_size, FLAGS.num_epochs)
#test = np.array([i for i in batches],dtype=np.int32)

print "Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_))
print "Train/Dev split: {:d}/{:d}".format(len(y_train),len(y_dev))

def embed(input_ph):
    with tf.device('/cpu:0'), tf.name_scope('embedding'):
        W = tf.Variable(
            tf.random_uniform([len(vocab_processor.vocabulary_), FLAGS.embedding_dim], -1.0,-1.0),
            name="W")
        embedded_chars = tf.nn.embedding_lookup(W, input_ph) #embedを行う。[None,sequence_length, embedding_size]のベクトル
        #embedded_chars_expanded = tf.expand_dims(embedded_chars, -1) #[None,sequence_length, embedding_size, -1]にしてconv
        print "self.embedded_chars.shape:{}".format(embedded_chars.get_shape())
        #print "self.embedded_chars_expanded.shape:{}".format(embedded_chars_expanded.get_shape())
    #return embedded_chars_expanded
    return embedded_chars

def RNN(input_op,init_state_ph):
    ans = []
    with tf.name_scope("RNN"):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.rnn_hidden,state_is_tuple=False)
        #lstm.output_size = 100
        state = init_state_ph

        #state = lstm.zero_state(64,tf.float32)

        print "input_op.get_shape() =",input_op.get_shape()
        print "state,get_shape() =",state.get_shape()
        #for i in xrange(input_op.get_shape()[1]):
            #output, state = lstm(input_op[i],state)
            #output,state = tf.nn.rnn(lstm,input_op[0][8],initial_state=state)
        output,state = tf.nn.dynamic_rnn(lstm,input_op,initial_state=state,time_major=False)
        """
        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.
        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.        
        """
        #output,state = tf.nn.rnn(lstm,tf.unpack(input_op[:,i,:],axis=0),initial_state=state)
        #tf.get_variable_scope().reuse_variables()

        ans.append(output)
        print "output.get_shape()",output.get_shape()
    return output,ans
    
with tf.Graph().as_default():

    with tf.Session() as sess:
        input_x_ph = tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
        #input_y_ph = tf.placeholder(tf.float32,[None,2],name="input_y")
        init_state_ph=tf.placeholder(tf.float32,[None,2*FLAGS.rnn_hidden],name="initial_state")

        embed_x_op = embed(input_x_ph)
        feature_op , _ = RNN(embed_x_op,init_state_ph)
        

        sess.run(tf.initialize_all_variables())
    
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            batchsize = len(x_batch)
            feed_dict = {
                input_x_ph:x_batch,
                init_state_ph:np.zeros(200*batchsize).reshape([batchsize,200])
            }
            print sess.run(feature_op,feed_dict)
