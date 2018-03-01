import os
from Agent import Agent
import gym
import random
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Queue
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pool = ThreadPool(processes=1)

# create anvironment
env = gym.make('CartPole-v0')
print("Created environment")

# evolution parameters
generations = 50
num_agents = 240

# start parameters for first gen agents
gamma = 0.9
epsilon = 0.2
epsilon = 0.2
learning_rate = 0.009
neurons = 5

for i in range(generations):
    # for each generation
    print("generation: " + str(i))
    parameters = []
    scores = []
    queues = []
    processes = []
    agents = np.zeros((num_agents, 5), dtype=np.float32)  # gamma, epsilon, learning_rate, neurons, score
    for a in range(num_agents):
        # for each agent a in generation i
        agents[a][0] = random.gauss(mu=gamma, sigma=0.002)
        agents[a][1] = random.gauss(mu=epsilon, sigma=0.003)
        agents[a][2] = random.gauss(mu=learning_rate, sigma=0.0005)
        agents[a][3] = int(max(1, random.gauss(mu=neurons, sigma=0.5)))
        agent = Agent(agents[a][0], agents[a][1], agents[a][2], int(agents[a][3]), env)
        queues.append(Queue())
        processes.append(Process(target=agent.learn, args=(queues[a],)))
        processes[a].start()

    print("launched all threads")
    for a in range(num_agents):
        agents[a][4] = queues[a].get()

    # sort agents by score
    agents = agents[np.argsort(agents[:, 4])]

    # take mean of the 10 best agents parameters
    gamma = np.mean(agents[-5:, 0])
    epsilon = np.mean(agents[-5:, 1])
    learning_rate = np.mean(agents[-5:, 2])
    neurons = np.mean(agents[-5:, 3])

    # print values of ten best agents
    print(agents[-10:, 4])
    print("gamma: " + str(gamma)
          + "   epsilon: " + str(epsilon)
          + "   learning_rate: " + str(learning_rate)
          + "   neurons: " + str(neurons))
