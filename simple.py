import tensorflow as tf
import numpy as np

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

# weights
weights0 = tf.Variable(tf.random_normal([n_input, n_hidden]))
biases0 = tf.Variable(tf.random_normal([n_hidden]))
weights1 = tf.Variable(tf.random_normal([n_hidden, n_output]))
biases1 = tf.Variable(tf.random_normal([n_output]))

# model
input_layer = tf.placeholder(tf.float32, [None, n_input])
hidden_layer = tf.nn.sigmoid(tf.matmul(input_layer, weights0) + biases0)
output_layer = tf.nn.sigmoid(tf.matmul(hidden_layer, weights1) + biases1)

# optimizer
expected_output = tf.placeholder(tf.float32, [None, n_output])
loss = tf.reduce_mean(tf.pow(expected_output - output_layer, 2))
optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train
    for i in range(5000):
        _, c = sess.run([optimizer, loss], feed_dict={
            input_layer: input_dataset,
            expected_output: output_dataset})
        if i % 100 == 0:
            print("i: %4d  Loss: %.6f" % (i, c))

    # sample run
    test_run = sess.run(output_layer, feed_dict={
        input_layer: [[0, 0, 1], [1, 1, 1], [0, 1, 1]]})

    print(np.round(test_run).astype(int))
