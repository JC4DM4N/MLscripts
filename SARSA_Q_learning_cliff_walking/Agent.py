import numpy as np
from Cliff import Cliff

class Agent():
    """
    Class to play the game, choosing which actions to take, and calculating cumulative rewards.
    """

    def __init__(self, epsilon=0.2):
        self.cliff = Cliff()
        self.actions = ["up","down","left","right"]
        self.epsilon = epsilon
        self.pos = (3, 0)

    def reset(self):
        self.cliff = Cliff()
        self.pos = (3, 0)

    def choose_action(self):
        """
        Choose the next action using epsilon-greedy policy.
        epsilon chance of choosing a random action.
        (1 - epsilon) chance of executing the greedy policy, which will choose the next state
            with the highest reward.
        """
        max_reward = -999
        if np.random.random() <= self.epsilon:
            next_action = np.random.choice(self.actions)
        else:
            for action in self.actions:
                next_reward = self.cliff.sim_next_reward(action)
                if next_reward > max_reward:
                    next_action = action
                    max_reward = next_reward
        return next_action
