import gym
import numpy as np

import tiles3 as tiles

class MountainCar():
    def __init__(self, iht, maxsize, num_tilings):
        self.env = gym.make('MountainCar-v0')
        self.iht = iht
        self.maxsize = maxsize
        self.num_tilings = num_tilings
        # reset the car
        self.state = self.env.reset()
        # initialise weights
        self.weights = np.zeros([env.action_space.n, self.maxsize])
        # hyperparameters
        self.alpha = 0.1 # learning rate
        self.epsilon = 0.1 # for epsilon-greedy policy

    def reset(self):
        self.state = self.env.reset()

    def scaleToTiles(self, x, y):
        """
        Scale our observation space to the number of x,y tiles used, then return
            the tile that our state is currently in.
        """
        high = self.env.observation_space.high
        low = self.env.observation_space.low
        scaleFactor = np.sqrt(self.maxsize)/(high-low)
        return tiles.tiles(self.iht, self.num_tilings, [scaleFactor*x, scaleFactor*y])

    def update_weights(self, Qpi, Qw):
        TD_loss = Qpi - Qw
        dw = np.dot(TD_loss,state)
        dw = -alpha*d
        return weights + dw

    def calc_Q(self):
        value = np.dot(self.weights.T)
        return value

    def policy(self, tiles):
        """
        Epsilon-greedy
        """
        # for action in [0,1,2]
        # calc_Q for each action and take the action with the max Q
        max_reward = -999
        next_action = None
        for action in range(self.env.action_space):
            value = calc_Q(weights[tiles], )
            if value > max_reward:
                next_action = action
                max_reward = value
        ALSO NEED TO IMPLEMENT EPSILON PART
        return action

"""
# position and velocity
env.observation_space

# get random action
# 0 = force left, 1 = nothing, 2 = force right
env.action_space.sample()

# state is x, y position
state = env.reset()
# sample 1000 random actions
for i in range(1000):
    env.render()
    new_state, reward, done, info = env.step(env.action_space.sample())
env.close()
"""

"""
Set up tile code
"""
maxsize = 1024 # will give a 32x32 grid
# set up index hash table
iht = tiles.IHT(maxsize)
num_tilings = 8 # number of offset tilings

# initialise the MountainCar class
MC = MountainCar(iht, maxsize, num_tilings)

nepisodes = 1000
for N in range(nepisodes):
    # reset environment
    MC.reset()
    while True:
        action = MC.policy
