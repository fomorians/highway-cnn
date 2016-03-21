from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

# define flags (note that Fomoro will not pass any flags by default)
flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

# define artifact directories where results from the session can be saved
model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

mnist = input_data.read_data_sets('mnist', one_hot=True)

def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(bias_init, shape=b_shape), name='bias')
    return W, b

def dense_layer(x, W_shape, b_shape, activation):
    W, b = weight_bias(W_shape, b_shape)
    return activation(tf.matmul(x, W) + b)

def conv2d_layer(x, W_shape, b_shape, strides, padding):
    W, b = weight_bias(W_shape, b_shape)
    return tf.nn.relu(tf.nn.conv2d(x, W, strides, padding) + b)

def highway_conv2d_layer(x, W_shape, b_shape, strides, padding, carry_bias=-1.0):
    W, b = weight_bias(W_shape, b_shape, carry_bias)
    W_T, b_T = weight_bias(W_shape, b_shape)
    H = tf.nn.relu(tf.nn.conv2d(x, W, strides, padding) + b, name='activation')
    T = tf.sigmoid(tf.nn.conv2d(x, W_T, strides, padding) + b_T, name='transform_gate')
    C = tf.sub(1.0, T, name="carry_gate")
    return tf.add(tf.mul(H, T), tf.mul(x, C), 'y') # y = (H * T) + (x * C)

with tf.Graph().as_default(), tf.Session() as sess:
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])

    carry_bias_init = -1.0

    x_image = tf.reshape(x, [-1, 28, 28, 1]) # reshape for conv

    keep_prob1 = tf.placeholder("float", name="keep_prob1")
    x_drop = tf.nn.dropout(x_image, keep_prob1)

    prev_y = conv2d_layer(x_drop, [5, 5, 1, 32], [32], [1, 1, 1, 1], 'SAME')
    prev_y = highway_conv2d_layer(prev_y, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=carry_bias_init)
    prev_y = highway_conv2d_layer(prev_y, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=carry_bias_init)

    prev_y = tf.nn.max_pool(prev_y, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    keep_prob2 = tf.placeholder("float", name="keep_prob2")
    prev_y = tf.nn.dropout(prev_y, keep_prob2)

    prev_y = highway_conv2d_layer(prev_y, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=carry_bias_init)
    prev_y = highway_conv2d_layer(prev_y, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=carry_bias_init)
    prev_y = highway_conv2d_layer(prev_y, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=carry_bias_init)

    prev_y = tf.nn.max_pool(prev_y, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    keep_prob3 = tf.placeholder("float", name="keep_prob3")
    prev_y = tf.nn.dropout(prev_y, keep_prob3)

    prev_y = highway_conv2d_layer(prev_y, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=carry_bias_init)
    prev_y = highway_conv2d_layer(prev_y, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=carry_bias_init)
    prev_y = highway_conv2d_layer(prev_y, [3, 3, 32, 32], [32], [1, 1, 1, 1], 'SAME', carry_bias=carry_bias_init)

    prev_y = tf.nn.max_pool(prev_y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    keep_prob4 = tf.placeholder("float", name="keep_prob4")
    prev_y = tf.nn.dropout(prev_y, keep_prob4)

    prev_y = tf.reshape(prev_y, [-1, 4 * 4 * 32])
    y = dense_layer(prev_y, [4 * 4 * 32, 10], [10], tf.nn.softmax)

    # define training and accuracy operations
    with tf.name_scope("loss") as scope:
        loss = -tf.reduce_sum(y_ * tf.log(y))
        tf.scalar_summary("loss", loss)

    with tf.name_scope("train") as scope:
        train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

    with tf.name_scope("test") as scope:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.scalar_summary('accuracy', accuracy)

    merged_summaries = tf.merge_all_summaries()

    # create a saver instance to restore from the checkpoint
    saver = tf.train.Saver(max_to_keep=1)

    # initialize our variables
    sess.run(tf.initialize_all_variables())

    # save the graph definition as a protobuf file
    tf.train.write_graph(sess.graph_def, model_path, 'highway.pb', as_text=False)

    # restore variables
    if FLAGS.restore:
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint_path:
            saver.restore(sess, latest_checkpoint_path)

    if not FLAGS.skip_training:
        summary_writer = tf.train.SummaryWriter(summary_path, sess.graph_def)

        num_steps = 5000
        checkpoint_interval = 100
        batch_size = 50

        step = 0
        for i in range(num_steps):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            if step % checkpoint_interval == 0:
                validation_accuracy, summary = sess.run([accuracy, merged_summaries], feed_dict={
                    x: mnist.validation.images,
                    y_: mnist.validation.labels,
                    keep_prob1: 1.0,
                    keep_prob2: 1.0,
                    keep_prob3: 1.0,
                    keep_prob4: 1.0,
                })
                summary_writer.add_summary(summary, step)
                saver.save(sess, checkpoint_path + 'checkpoint', global_step=step)
                print('step %d, training accuracy %g' % (step, validation_accuracy))

            sess.run(train_step, feed_dict={
                x: batch_xs,
                y_: batch_ys,
                keep_prob1: 0.8,
                keep_prob2: 0.7,
                keep_prob3: 0.6,
                keep_prob4: 0.5,
            })

            step += 1

        summary_writer.close()

    test_accuracy = sess.run(accuracy, feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        keep_prob1: 1.0,
        keep_prob2: 1.0,
        keep_prob3: 1.0,
        keep_prob4: 1.0,
    })
    print('test accuracy %g' % test_accuracy)
