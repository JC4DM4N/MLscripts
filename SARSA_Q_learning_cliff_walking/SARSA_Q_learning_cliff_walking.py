"""
Script to execute the cliff walking example from Sutton & Barto 2014
(https://towardsdatascience.com/reinforcement-learning-cliff-walking-implementation-e40ce98418d4)
Use SARSA and Q learning to teach an agent to find the optimal path around a cliff
without falling off the cliff.
Compare the policies which are obtained from SARSA and Q learning, and evaluate why each
algorithm reaches the policy that it does.
"""

import matplotlib.pyplot as plt
import numpy as np

from Agent import Agent
from Cliff import Cliff

def plot_route(states):
    board = Cliff().board
    rows = np.shape(board)[0]
    cols = np.shape(board)[1]
    route = np.zeros([rows,cols])

    positions_passed = [s[0] for s in states]

    for i in range(rows):
        for j in range(cols):
            if (i,j) in positions_passed:
                route[i,j] = 1
            elif board[i,j] == "*":
                route[i,j] = -1
    plt.figure()
    plt.imshow(route)

def plot_state_values(states_action_values,strategy):
    board = Cliff().board
    rows = np.shape(board)[0]
    cols = np.shape(board)[1]
    values = np.zeros([rows,cols])

    for i in range(rows):
        for j in range(cols):
            # take the state value as the mean over all values if considering SARSA, and max if considering Q
            if strategy=="Q":
                values[i,j] = np.max([states_action_values[i,j][a] for a in Agent().actions])
            elif strategy=="SARSA":
                values[i,j] = np.max([states_action_values[i,j][a] for a in Agent().actions])
    plt.figure()
    plt.imshow(values)
    plt.colorbar()

agent = Agent()
agent.play(strategy="Q")
final_route = agent.states
states_action_values = agent.state_actions
plot_route(final_route)
plt.title("Q learning final route")
plot_state_values(states_action_values, strategy="Q")
plt.title("Q learning state values")

agent = Agent()
agent.play(strategy="SARSA")
final_route = agent.states
states_action_values = agent.state_actions
plot_route(final_route)
plt.title("SARSA learning final route")
plot_state_values(states_action_values, strategy="SARSA")
plt.title("SARSA learning state values")

plt.show()
