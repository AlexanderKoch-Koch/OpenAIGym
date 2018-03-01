import numpy as np
from collections import deque            # For storing moves
import random     # For sampling batches from the observations
import tensorflow as tf


class Agent:


    def __init__(self, gamma, epsilon, learning_rate, neurons, environment):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.neurons = neurons
        self.environment = environment

    def learn(self, q):
        batch_size = 64
        num_episodes = 800
        num_observations = self.environment.observation_space.shape[0]
        num_actions = self.environment.action_space.n

        #These lines establish the feed-forward part of the network used to choose actions
        input = tf.placeholder(shape=[1, num_observations], dtype=tf.float32)

        l1_W = tf.Variable(tf.random_normal([num_observations, self.neurons]), name="l1_W")
        l1_b = tf.Variable(tf.random_normal([self.neurons]), name="l1_b")
        l1 = tf.nn.relu(tf.matmul(input, l1_W) + l1_b)

        l2_W = tf.Variable(tf.random_normal([self.neurons, num_actions]), name="l2_W")
        output = tf.matmul(l1, l2_W)

        prediction = tf.argmax(output, 1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        nextQ = tf.placeholder(shape=[1, num_actions], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(nextQ - output))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        updateModel = trainer.minimize(loss)


        # variables to track performance
        steps_per_episode_list = []
        reward_per_episode_list = []

        experience = deque(maxlen=4000)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for episode in range(num_episodes):
                #Reset environment and get first new observation
                s = self.environment.reset()
                step = 0
                is_done = False
                reward_per_episode = 0
                exploration_score = self.epsilon

                while is_done == False:
                    step += 1
                    action_array, allQ = sess.run([prediction, output], feed_dict={input: s.reshape(1, num_observations)})

                    action = action_array[0]

                    if np.random.rand(1) < exploration_score:
                        action = self.environment.action_space.sample()

                    # Get new state and reward from environment
                    s1, reward, is_done, _ = self.environment.step(action)
                    reward_per_episode += reward
                    experience.append((s, action, reward, s1, is_done))
                    # Obtain maxQ' and set our target value for chosen action.
                    s = s1
                    if is_done:
                        self.epsilon /= 1.0005
                        steps_per_episode_list.append(step)
                        reward_per_episode_list.append(reward_per_episode)
                        break


                # replay randomly from experience every 10 episodes
                if episode % 3 == 0:
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
                                targetQ[0, action] = reward + self.gamma * maxQ1
                            # Train our network using target and predicted Q values
                            _ = sess.run([updateModel], feed_dict={input: state.reshape(1, num_observations), nextQ: targetQ})

        performance = sum(steps_per_episode_list[-100:]) / 100
        q.put(performance)
