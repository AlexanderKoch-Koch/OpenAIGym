import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import Output

gamma = 0.9
epsilon = 0.5
iterations = 5000


# network architecture
x = tf.placeholder(tf.float32, shape=[1, 4])
y = tf.placeholder(tf.float32, shape=[1, 2])

l1_W = tf.Variable(tf.random_normal([4, 24], seed=1), name="l1_W")
l1_b = tf.Variable(tf.random_normal([24], seed=2), name="l1_b")
l1 = tf.nn.relu(tf.matmul(x, l1_W) + l1_b)

l2_W = tf.Variable(tf.random_normal([24, 24], seed=3), name="l2_W")
l2_b = tf.Variable(tf.random_normal([24], seed=4), name="l2_b")
l2 = tf.nn.relu(tf.matmul(l1, l2_W) + l2_b)

l3_W = tf.Variable(tf.random_normal([24, 2], seed=5), name="l3_W")
l3_b = tf.Variable(tf.random_normal([2], seed=6), name="l3_b")
output_layer = tf.transpose(tf.nn.sigmoid(tf.matmul(l2, l3_W) + l3_b))

loss = tf.reduce_sum(tf.square(output_layer - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

init = tf.global_variables_initializer()


#setup environment
environment = gym.make("CartPole-v0")
newState = np.empty(shape=(1, 4), dtype='float32')
lastState = [[]]
newState[0] = environment.reset()

# variables to track performance
steps_per_episode = 0
steps_per_episode_list = []

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(iterations):
        netOutput = sess.run(output_layer, feed_dict={x: newState})
        if np.random.rand() < epsilon:
            action = environment.action_space.sample()
        else:
            action = np.argmax(netOutput)

        # save original state
        lastState = newState
        newState[0], reward, done, info = environment.step(action)
        environment.render()
        if done:
           reward = -100
        # calculate true future reward
        Q1 = sess.run(output_layer, feed_dict={x: newState})

        maxQ1 = np.max(Q1)
        targetQ = netOutput
        targetQ[action, 0] = reward + gamma * maxQ1
        # Train our network using target and predicted Q values
        loss_value, optimizer_value = sess.run((loss, optimizer), feed_dict={x: lastState, y: targetQ.T})
        #loss_value, optimizer_value = sess.run((loss, optimizer), feed_dict={x: [[0, 0.1, 0.9, 0.8]], y: [[1, 1]]})
        print(loss_value)
        steps_per_episode += 1
        if done:
            epsilon /= 1.0005
            Output.addIterationInfo("steps", steps_per_episode)
            #Output.showIterationInfo()
            sys.stdout.flush()
            steps_per_episode_list.append(steps_per_episode)
            steps_per_episode = 0
            newState[0] = environment.reset()


# plot performance
plt.plot(steps_per_episode_list)
plt.show()
