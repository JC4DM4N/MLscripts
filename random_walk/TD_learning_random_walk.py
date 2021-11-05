"""
A script to run a simple example of Temporal Difference 0 (TD(0)) policy evaluation.
Random walk between states A through G, with a 50% probability of moving left and right at each timestep.
Terminal state A has a value of 0, and terminal state G has a reward of +1
System initially looks like:

A-B-C-D-E-F-G
0-0-0-0-0-0-1

The aim is to determine the values of all the states, using TD learning to update the state values at
each timestep.

TD learning value function update is:
V(S_{t+1}) = V(St) + alpha*(R_{t+1} + gamma*V(S_{t+1}) - V(St))
"""

import numpy as np

def move(position):
    """
    Move left or right with 50% probability.
    """
    direction = np.random.choice(["left", "right"])
    if direction == "left":
        # move left
        position -= 1
    elif direction == "right":
        # move right
        position += 1
    return position

def get_reward(position):
    """
    Only state 6 has a direct reward, which is +1.
    """
    if position == 6:
        return 1
    else:
        return 0

def update_values(Vt, old_position, new_position):
    """
    Update values using the TD learning error.
    Note: Don't double count the reward at terminal states.
    """
    reward = get_reward(new_position)
    if new_position == 0 or new_position == 6:
        Vt[old_position] += alpha*(reward - Vt[old_position])
    else:
        Vt[old_position] += alpha*(reward + gamma*Vt[new_position] - Vt[old_position])
    return Vt

Vt = np.asarray([0.,0.,0.,0.,0.,0.,1.])
alpha = 0.1
gamma = 1.0
# start in the middle
position = 3

# number of episodes
N = 5000
for i in range(N):
    terminate = False
    while not terminate:
        new_position = move(position)
        Vt = update_values(Vt, position, new_position)
        position = new_position
        if position==0 or position==6:
            # terminate and start again
            position = 3
            terminate = True

print("Predicted values of states A to G:")
print(Vt)
