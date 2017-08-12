import tensorflow as tf
import numpy as np
import time

input_dataset = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]
output_dataset = [
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 0, 0]
]

n_input = 3
n_hidden = 6
n_output = 3

# model
with tf.name_scope("Input"):
    input_layer = tf.placeholder(tf.float32, [None, n_input])
with tf.name_scope("Hidden"):
    weights0 = tf.Variable(tf.random_normal([n_input, n_hidden]), name='weights')
    biases0 = tf.Variable(tf.random_normal([n_hidden]), name='biases')
    hidden_layer = tf.nn.sigmoid(tf.matmul(input_layer, weights0) + biases0)
with tf.name_scope("Output"):
    weights1 = tf.Variable(tf.random_normal([n_hidden, n_output]), name='weights')
    biases1 = tf.Variable(tf.random_normal([n_output]), name='biases')
    output_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, weights1) + biases1)

# Optimizer
with tf.name_scope("Optimizer"):
    expected_output = tf.placeholder(tf.float32, [None, n_output])
    loss = tf.reduce_mean(tf.pow(expected_output - output_layer, 2))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# summary
tf.summary.scalar("loss", loss)
tf.summary.histogram("weights0", weights0)
tf.summary.histogram("biases0", biases0)
tf.summary.histogram("weights1", weights1)
tf.summary.histogram("biases1", biases1)
merged = tf.summary.merge_all()

with tf.name_scope("Initialiser"):
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    summary_dir = './summary/%s' % time.strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # train
    feed_dict = {input_layer: input_dataset, expected_output: output_dataset}
    for i in range(5000):
        _, s = sess.run([optimizer, merged], feed_dict=feed_dict)
        summary_writer.add_summary(s, i)
