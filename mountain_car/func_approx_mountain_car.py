"""
Using linear expectation model where q(s,a) = dot(w(s,a).T, x(s))
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

import tiles3 as tiles

class MountainCar():
    def __init__(self):
        # setup the environment and reset the car's state
        self.env = gym.make('MountainCar-v0')
        self.state = self.env.reset()
        # set up the tilings code
        self.maxsize = 512 # will give a 32x32 grid
        self.num_tilings = 8 # number of offset tilings
        self.iht = tiles.IHT(self.maxsize)
        # initialise weights
        self.weights = np.zeros([self.env.action_space.n, self.maxsize])
        # hyperparameters
        self.gamma = 1.0 # memory
        self.alpha = 0.1 # learning rate
        self.epsilon = 0.1 # for epsilon-greedy policy

    def reset(self):
        """
        Reset the state of the car
        """
        self.state = self.env.reset()

    def get_active_tiles(self, state):
        """
        Scale our observation space to the number of x,y tiles used, then return
            the tile that our state is currently in.
        Passing state as input variable, as this function is used to get active_tiles
            for current state and also future states, hence self.state should not always
            be used.
        """
        high_pos = self.env.observation_space.high[0]
        low_pos = self.env.observation_space.low[0]
        position_scale = np.sqrt(self.maxsize)/(high_pos-low_pos)

        high_vel = self.env.observation_space.high[1]
        low_vel = self.env.observation_space.low[1]
        velocity_scale = np.sqrt(self.maxsize)/(high_vel-low_vel)

        active_tiles = tiles.tiles(self.iht,
                                   self.num_tilings,
                                   [position_scale*state[0], velocity_scale*state[1]])
        return active_tiles

    def policy(self, tiles):
        """
        Epsilon-greedy
        """
        ###TRYING WITH GREEDY ONLY
        if np.random.random() < 0: #self.epsilon:
            # choose a random action
            next_action = np.random.choice(range(self.env.action_space.n))
        else:
            # for action in [0,1,2]
            # calc_Q for each action and take the action with the max Q
            max_reward = -999
            next_action = None
            for action in range(self.env.action_space.n)[::-1]:
                value = self.calc_Q(action, tiles)
                if value > max_reward:
                    next_action = action
                    max_reward = value
        if next_action is None:
            import pdb ; pdb.set_trace()
        return next_action

    def calc_Q(self, action, tiles):
        # sum the weights for the active tiles
        value = np.sum(self.weights[action][tiles])
        return value

    def update_weights(self, Qpi, Qw, action, active_tiles):
        """
        Update weights using TD loss and gradient descent
        """
        TD_loss = Qpi - Qw
        dw = self.alpha*TD_loss
        self.weights[action][active_tiles] += dw

def plot_value_func(MC):
    Prange = np.linspace(MC.env.observation_space.low[0], MC.env.observation_space.high[0], 100)
    Vrange = np.linspace(MC.env.observation_space.low[1], MC.env.observation_space.high[1], 100)
    values = []
    for p in Prange:
        values_for_p = []
        for v in Vrange:
            active = MC.get_active_tiles([p, v])
            value = max(MC.calc_Q(a, active) for a in range(MC.env.action_space.n))
            values_for_p.append(value)
        values.append(values_for_p)

    X, Y = np.meshgrid(Prange, Vrange)
    values = np.asarray(values)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, values)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    plt.show()

def main():
    # initialise the MountainCar class
    MC = MountainCar()
    nepisodes = 500
    for N in range(nepisodes):
        print(f"Starting episode {N}")
        # reset environment
        MC.reset()
        num_steps = 0
        while True:
            MC.env.render()
            # get current active tiles and choose epsilon-greedy action
            active_tiles = MC.get_active_tiles(MC.state)
            action = MC.policy(active_tiles)
            # now step using the action, get the tiles for the next state, and
            # determine what the next action would be.
            next_state, reward, done, info = MC.env.step(action)
            next_active_tiles = MC.get_active_tiles(next_state)
            next_action = MC.policy(next_active_tiles)
            # TD loss and gradient descent
            Qw = MC.calc_Q(action, active_tiles)
            Qpi = reward + MC.gamma*MC.calc_Q(next_action, next_active_tiles)
            MC.update_weights(Qpi, Qw, action, active_tiles)

            num_steps += 1
            # check if episode has completed
            if done: # or num_steps == 200:
                break
        print(f"Episode completed in {num_steps} steps with a final reward {reward}")

    plot_value_func(MC)

if __name__=="__main__":
    main()

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
