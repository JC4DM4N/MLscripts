import numpy as np
from Cliff import Cliff

class Agent():
    """
    Class to play the game, choosing which actions to take, and calculating cumulative rewards.
    """

    def __init__(self, epsilon=0.5, alpha=0.1, gamma=1.0):
        self.cliff = Cliff()
        self.actions = ["up","down","left","right"]
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.pos = (3, 0)
        self.states = []
        # need a dictionary to contain estimates for all state-action combinations
        # initial estimates are all zeros
        self.state_actions = {}
        rows = np.shape(self.cliff.board)[0]
        cols = np.shape(self.cliff.board)[1]
        for i in range(rows):
            for j in range(cols):
                self.state_actions[(i,j)] = {}
                for action in self.actions:
                    self.state_actions[(i,j)][action] = 0

    def reset(self):
        self.cliff = Cliff()
        self.pos = (3, 0)
        self.states = []

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
            # greedy over all possible state action combinations
            for action in self.actions:
                next_reward = self.state_actions[self.pos][action]
                if next_reward > max_reward:
                    next_action = action
                    max_reward = next_reward
        return next_action

    def update_estimates_SARSA(self):
        """
        Go back through all state actions the agent has taken, and update the state action
            estimates.
        """
        # get reward from the final state
        end_state = self.pos
        # state action value for end state is just the reward from the end state
        state_action_value = self.cliff.give_reward()
        for action in self.actions:
            self.state_actions[end_state][action] = state_action_value
        # go back through all the states
        for state in reversed(self.states):
            s, a, r = state
            current_estimate = self.state_actions[s][a]
            state_action_value = current_estimate + self.alpha*(r +
                                                                self.gamma*(state_action_value) -
                                                                current_estimate)
            self.state_actions[s][a] = state_action_value

    def update_estimates_Q(self):
        """
        Go back through all state actions the agent has taken, and update the state action
            estimates.
        Difference to the SARSA is that we value our next state-action as the maximum value
            over all possible actions.
        """
        # get reward from the final state
        end_state = self.pos
        # state action value for end state is just the reward from the end state
        state_action_value = self.cliff.give_reward()
        for action in self.actions:
            self.state_actions[end_state][action] = state_action_value
        # go back through all the states
        for state in reversed(self.states):
            s, a, r = state
            current_estimate = self.state_actions[s][a]
            state_action_value = current_estimate + self.alpha*(r +
                                                                self.gamma*(state_action_value) -
                                                                current_estimate)
            self.state_actions[s][a] = state_action_value
            # take the current state_action_value to be the max over all actions in the current state
            state_action_value = max(self.state_actions[s].values())

    def play(self, episodes=5000, strategy="Q"):
        """
        Execute the agent playing the game
        """
        for N in range(episodes):
            while 1:
                state = self.pos
                current_reward = self.cliff.give_reward()
                # choose action, move agent, and calculate the reward
                next_action = self.choose_action()
                self.cliff.move_agent(next_action)
                self.pos = self.cliff.pos
                # save positions, actions and rewards history
                self.states.append([state, next_action, current_reward])
                # check if game has ended
                if self.cliff.end:
                    break

            # game has ended, update estimes of state-action combinations
            if strategy=="Q":
                self.update_estimates_Q()
            elif strategy=="SARSA":
                self.update_estimates_SARSA()
            if N < episodes - 1:
                # reset and go again - if on the final episode then don't reset, for debugging purposes.
                self.reset()
                # also, lets decay epsilon at each timestep
                self.epsilon = self.epsilon/1.01
            else:
                # append the final state to the self.states list, for plotting purposes
                self.states.append([self.pos, None, None])
