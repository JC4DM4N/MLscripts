"""
Script to execute the cliff walking example from Sutton & Barto 2014
(https://towardsdatascience.com/reinforcement-learning-cliff-walking-implementation-e40ce98418d4)
Use SARSA and Q learning to teach an agent to find the optimal path around a cliff
without falling off the cliff.
Compare the policies which are obtained from SARSA and Q learning, and evaluate why each
algorithm reaches the policy that it does.
"""

from Agent import Agent

agent = Agent()
agent.play()

import pdb ; pdb.set_trace()

print()
