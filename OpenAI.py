import gym
import numpy as np
import matplotlib.pyplot as plt
import Output
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves
import random     # For sampling batches from the observations
import tensorflow as tf


gamma = 0.92
epsilon = 0.3
num_episodes = 3000
batch_size = 64

#setup environment
environment = gym.make("MountainCar-v0")
num_observations = environment.observation_space.shape[0]
num_actions = environment.action_space.n

#These lines establish the feed-forward part of the network used to choose actions
input = tf.placeholder(shape=[1, num_observations], dtype=tf.float32)

l1_W = tf.Variable(tf.random_normal([num_observations, 24]), name="l1_W")
l1_b = tf.Variable(tf.random_normal([24]), name="l1_b")
l1 = tf.nn.relu(tf.matmul(input, l1_W) + l1_b)

l2_W = tf.Variable(tf.random_normal([24, num_actions]), name="l2_W")
l2_b = tf.Variable(tf.random_normal([num_actions]), name="l2_b")
output = tf.matmul(l1, l2_W)

prediction = tf.argmax(output, 1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, num_actions], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - output))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
updateModel = trainer.minimize(loss)


# variables to track performance
steps_per_episode = 0
steps_per_episode_list = []
reward_per_episode_list = []

experience = deque(maxlen=2000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(num_episodes):
        #Reset environment and get first new observation
        s = environment.reset()
        d = False
        step = 0
        is_done = False
        reward_per_episode = 0
        exploration_score = epsilon

        while is_done == False:
            step += 1
            action_array, allQ = sess.run([prediction, output], feed_dict={input: s.reshape(1, num_observations)})
            action = action_array[0]
            # add noise to action choice
            #print(range(len(allQ[0])))
            '''for possible_action in range(len(allQ[0])):
                allQ[0, possible_action] *= random.gauss(mu=1, sigma=epsilon)

            action = np.argmax(allQ)'''




            if np.random.rand(1) < exploration_score:
                action = environment.action_space.sample()
                exploration_score *= 1.1    # make further exploration more likely
            else:
                exploration_score *= 0.9

            # Get new state and reward from environment
            s1, reward, is_done, _ = environment.step(action)
            reward_per_episode += reward
            # if episode % 100 == 0:  # render every 100th episode
            #environment.render()
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(output, feed_dict={input: s.reshape(1, num_observations)})
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
                epsilon /= 1.0005
                steps_per_episode_list.append(step)
                reward_per_episode_list.append(reward_per_episode)
                print(str(episode) + ": " + str(reward_per_episode))
                print(epsilon)
                if reward_per_episode != -200:
                    print("reached top")
                break

        # replay randomly from experience
        if len(experience) > batch_size:
            batch = random.sample(experience, batch_size)
            for state, action, reward, next_state, is_done in batch:
                allQ = sess.run(output, feed_dict={input: state.reshape(1, num_observations)})
                Q1 = sess.run(output, feed_dict={input: next_state.reshape(1, num_observations)})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                if is_done:
                    targetQ[0, action] = reward
                else:
                    targetQ[0, action] = reward + gamma * maxQ1
                # Train our network using target and predicted Q values
                _ = sess.run([updateModel], feed_dict={input: state.reshape(1, num_observations), nextQ: targetQ})

# plot performance
#plt.plot(steps_per_episode_list)
#plt.show()

# compute average performance over last 100 episodes
performance = sum(steps_per_episode_list[-100:]) / 100
print("recent average performance: " + str(performance))
