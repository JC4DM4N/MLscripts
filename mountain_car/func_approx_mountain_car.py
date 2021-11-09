import gym
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()


nweights = 400
weights = np.zeros([env.action_space.n, nweights])

# hyperparameters
alpha = 0.1 # learning rate
epsilon = 0.1 # for epsilon-greedy policy
nepisodes = 1000

# position and velocity
env.observation_space

# get random action
# 0 = force left, 1 = nothing, 2 = force right
env.action_space.sample()

def calc_Q(weights, x):
    value = np.dot(weights, x)
    return value

def policy(state, weights, epsilon):
    """
    Epsilon-greedy
    """
    # for action in [0,1,2]
    # calc_Q for each action and take the action with the max Q
    """
    current_max = -999
    current_action = None
    for action in env.action_space:
        value = calc_Q(weights[action], x)
        if value > current_max:
            current_action = action
            current_max = value
    ALSO NEED TO IMPLEMENT EPSILON PART
    return action
    """

# state is x, y position
state = env.reset()
# sample 1000 random actions
for i in range(1000):
    env.render()
    new_state, reward, done, info = env.step(env.action_space.sample())
env.close()

for N in range(nepisodes):
    # reset environment
    state = env.reset()
    """
    while not terminal:
        
    """
