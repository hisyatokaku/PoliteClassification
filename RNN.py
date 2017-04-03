#coding:utf-8

print "Experiment name:"
Exname = raw_input()

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import sys
import argparse
import data_helpers

POS_PATH = 
NEG_PATH = 


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", POS_PATH, "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", NEG_PATH, "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("lr", 0.1, "Learning rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("rnn_hidden", 100, "Hidden Unit Size (default: 100)")
tf.flags.DEFINE_integer("num_epochs",100, "Number of training epochs (default: 200)")
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
y_dev_   = [np.argmax(i) for i in y_dev]

sequence_length = x_train.shape[1]
batches = data_helpers.batch_iter(zip(x_train,y_train), FLAGS.batch_size, FLAGS.num_epochs)


print "Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_))
print "Train/Dev split: {:d}/{:d}".format(len(y_train),len(y_dev))

def embed(input_ph):
    with tf.device('/cpu:0'), tf.name_scope('embedding'):
        W = tf.Variable(
            tf.random_uniform([len(vocab_processor.vocabulary_), FLAGS.embedding_dim], -1.,-1.),
            name="W")
        embedded_chars = tf.nn.embedding_lookup(W, input_ph) # -> [None,sequence_length, embedding_size]
        weight_decay = tf.reduce_sum(tf.mul(W,W))
        print "self.embedded_chars.shape:{}".format(embedded_chars.get_shape())

    return embedded_chars,weight_decay

def accuracy(x,y):
    with tf.name_scope("accuracy"):
        #correct_of_pred = tf.equal(tf.argmax(x,1),tf.argmax(y,0))
        correct_of_pred = tf.equal(tf.argmax(x,1),tf.convert_to_tensor(y))

        print "x.get_shape():{},y.get_shape():{}".format(x.get_shape(),y.get_shape())

        acc = tf.reduce_mean(tf.cast(correct_of_pred,tf.float32))
        return acc

def loss(x,y):
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
        loss = tf.reduce_mean(cross_entropy,name="cross_entropy_mean")
        """
        softmax is done inside sparse_softmax_cross... func.
        """
    return loss

def FC_3layers(input_op):
    #batchsize = input_op.get_shape()[0]
    #print "batchsize:",batchsize
    w1_dim = 50
    with tf.name_scope("FC_3layers"):
        W1 = tf.Variable(tf.truncated_normal(shape=[FLAGS.rnn_hidden,w1_dim]),name="W1")
        b1 = tf.Variable(tf.constant(0.1,shape=[w1_dim]),name="b1")
        output1 = tf.nn.sigmoid(tf.matmul(input_op,W1) + b1)

        W2 = tf.Variable(tf.truncated_normal(shape=[w1_dim,2]),name="W2")
        b2 = tf.Variable(tf.constant(0.1,shape=[2]),name="b2")
        output2 = tf.nn.sigmoid(tf.matmul(output1,W2) + b2)
    return output2

def RNN(input_op,init_state_ph):
    with tf.name_scope("RNN"):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.rnn_hidden,state_is_tuple=False)
        state = init_state_ph

        print "input_op.get_shape() =",input_op.get_shape()
        print "state,get_shape() =",state.get_shape()

        #for i in xrange(input_op.get_shape()[1]):
            #output, state = lstm(input_op[i],state)
            #output,state = tf.nn.rnn(lstm,input_op[0][8],initial_state=stat
            #output,state = tf.nn.rnn(lstm,tf.unpack(input_op[:,i,:],axis=0),initial_state=state)
            #tf.get_variable_scope().reuse_variables()

        output,state = tf.nn.dynamic_rnn(lstm,input_op,initial_state=state,time_major=False)

        """
        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.
        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.        

        Output: [batch,time,embed_dim]

        """
        print "output.get_shape()",output.get_shape()
    return output[:,-1,:]

def dev_step(x_batch,y_batch,sess,writer=None):
    feed_dict = {
        input_x_ph:x_batch,
        input_y_ph:y_batch,
        init_state_ph:np.zeros(200*batchsize).reshape([batchsize,200])        
    }
    
    loss,acc,summaries = sess.run([loss_op,acc_op,dev_summary_op],feed_dict)
    print "Loss:{},Acc:{}".format(loss,acc)
    if writer:
        writer.add_summary(summaries,current_step)

def train_step(x_batch,y_batch,sess):
    feed_dict = {
        input_x_ph:x_batch,
        input_y_ph:y_batch,
        init_state_ph:np.zeros(200*batchsize).reshape([batchsize,200])
    }
    loss,acc,_,summaries = sess.run([loss_op,acc_op,train_op,train_summary_op],feed_dict)
    print "step {}, loss {:g}, acc{:g}".format(current_step,loss,acc),"\r"
    train_summary_writer.add_summary(summaries,current_step)
    
with tf.Graph().as_default():
    sess=tf.Session()
    with sess.as_default():
        input_x_ph    = tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
        input_y_ph    = tf.placeholder(tf.int64,[None],name="input_y")
        init_state_ph = tf.placeholder(tf.float32,[None,2*FLAGS.rnn_hidden],name="initial_state")

        embed_x_op , wd_op = embed(input_x_ph)
        feature_op = RNN(embed_x_op,init_state_ph)
        output_op  = FC_3layers(feature_op)
        loss_op    = loss(output_op,input_y_ph)
        acc_op     = accuracy(output_op,input_y_ph)
        global_step = tf.Variable(0, name='global_step', trainable=False) #incremented automatically by opt.minimize
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
        train_op = optimizer.minimize(loss_op, global_step=global_step)
        
    now = datetime.datetime.today()
    time ="{:02d}{:02d}_{:02d}:{:02d}_{}".format(now.month,now.day,now.hour,now.minute,Exname) 
    out_dir = os.path.join(os.path.curdir,"runs", time)

    #train
    loss_summary =  tf.scalar_summary("loss",loss_op)
    acc_summary  =  tf.scalar_summary("acc",acc_op)
    wd_summary   =  tf.scalar_summary("embed_wd",acc_op)

    train_summary_op = tf.merge_summary([loss_summary,acc_summary,wd_summary])
    train_summary_dir= os.path.join(out_dir,"summaries","train")
    train_summary_writer=tf.train.SummaryWriter(train_summary_dir,sess.graph)

    #dev
    dev_summary_op = tf.merge_summary([loss_summary,acc_summary])
    dev_summary_dir=  os.path.join(out_dir,"summaries","dev")
    dev_summary_writer=tf.train.SummaryWriter(dev_summary_dir,sess.graph)

    #checkpoint
    checkpoint_dir = os.path.abspath(os.path.join("./", "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    sess.run(tf.initialize_all_variables())


    for batch in batches:
        x_batch, y_batch = zip(*batch)
        batchsize = len(x_batch)
        y_batch = [np.argmax(i) for i in y_batch]
        #global_step = tf.Variable(0, name='global_step', trainable=False) #incremented automatically by opt.minimize
        current_step = tf.train.global_step(sess, global_step)
        print sess.run(wd_op)


        train_step(x_batch,y_batch,sess)

        #if current_step % (10*len(y_train)/FLAGS.batch_size) == 0:
        if current_step % 100 == 0:
            print "Evaluation \nStep:",current_step
            batchsize = len(y_dev)
            dev_step(x_dev,y_dev_,sess,writer=dev_summary_writer)

