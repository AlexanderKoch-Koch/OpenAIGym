import gym
import numpy as np
import matplotlib.pyplot as plt
import Output
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves
import random     # For sampling batches from the observations
import tensorflow as tf


gamma = 0.95
epsilon = 0.5
num_episodes = 1000
batch_size = 32

#setup environment
environment = gym.make("CartPole-v0")

#These lines establish the feed-forward part of the network used to choose actions
input = tf.placeholder(shape=[1, 4], dtype=tf.float32)
W = tf.Variable(tf.random_normal([4, 2]))

l1_W = tf.Variable(tf.random_normal([4, 8]), name="l1_W")
l1_b = tf.Variable(tf.random_normal([8]), name="l1_b")
l1 = tf.nn.relu(tf.matmul(input, l1_W) + l1_b)

l2_W = tf.Variable(tf.random_normal([8, 2]), name="l2_W")
l2_b = tf.Variable(tf.random_normal([2]), name="l2_b")
output = tf.matmul(l1, l2_W)
#output = l1
#output = tf.matmul(input, W)
prediction = tf.argmax(output, 1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, 2], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - output))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
updateModel = trainer.minimize(loss)


# variables to track performance
steps_per_episode = 0
steps_per_episode_list = []

experience = deque(maxlen=2000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(num_episodes):
        #Reset environment and get first new observation
        s = environment.reset()
        d = False
        step = 0
        print(episode)

        while step < 200:
            step += 1
            action_array, allQ = sess.run([prediction, output], feed_dict={input: s.reshape(1, 4)})
            action = action_array[0]
            if np.random.rand(1) < epsilon:
                action = environment.action_space.sample()

            # Get new state and reward from environment
            s1, reward, is_done, _ = environment.step(action)
            if episode % 100 == 0:  # render every 100th episode
                environment.render()
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(output, feed_dict={input: s.reshape(1, 4)})
            experience.append((s, action, reward, s1, is_done))
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            #if is_done:
            #    targetQ[0, action[0]] = reward
            #else:
            targetQ[0, action] = reward + gamma * maxQ1
            #Train our network using target and predicted Q values
            #_ = sess.run([updateModel], feed_dict={input: s.reshape(1, 4), nextQ: targetQ})
            s = s1
            if is_done:
                epsilon /= 1.005
                steps_per_episode_list.append(step)
                break

        # replay randomly from experience
        if(len(experience) > batch_size):
            batch = random.sample(experience, batch_size)
            for state, action, reward, next_state, is_done in batch:
                allQ = sess.run(output, feed_dict={input: state.reshape(1, 4)})
                Q1 = sess.run(output, feed_dict={input: next_state.reshape(1, 4)})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                if is_done:
                    targetQ[0, action] = reward
                else:
                    targetQ[0, action] = reward + gamma * maxQ1
                # Train our network using target and predicted Q values
                _ = sess.run([updateModel], feed_dict={input: state.reshape(1, 4), nextQ: targetQ})

# plot performance
plt.plot(steps_per_episode_list)
plt.show()
