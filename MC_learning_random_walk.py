"""
NOT SURE THIS WORKS PROPERLY YET - WILL REVISIT LATER.

A script to run a simple example of Monte Carlo (MC) RL learning policy evaluation.
Random walk between states A through G, with a 50% probability of moving left and right at each timestep.
Terminal state A has a value of 0, and terminal state G has a reward of +1
System initially looks like:

A-B-C-D-E-F-G
0-0-0-0-0-0-1

The aim is to determine the values of all the states, using TD learning to update the state values at
each timestep.

MC learning value function update is:
V(S_{t+1}) = V(St) + alpha*(G_{t} - V(St))
where:
G_{t} = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + ... gamma^{T-1}*R_{T}
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

def get_rewards(all_positions):
    """
    Only state 6 has a direct reward, which is +1.
    """
    # reward is +1 at state 6, and 0 otherwise
    all_rewards = 1*(all_positions==6)
    return all_rewards

def update_values(Vt, all_positions):
    """
    Update values using the TD learning error.
    Note: Don't double count the reward at terminal states.
    """
    total_steps = len(all_positions)
    Vt_new = Vt.copy()
    for i, position in enumerate(all_positions):
        if position==0 or position==6:
            # don't update values for the terminal states
            break
        # get all the rewards since being at this position
        all_rewards = get_rewards(all_positions[i+1:])
        # calculate the discounted rewards for this state
        Gt = sum([reward*gamma**j for j, reward in enumerate(all_rewards)])
        Vt_new[position] += alpha*(Gt - Vt[position])
    return Vt_new

Vt = np.asarray([0.,0.,0.,0.,0.,0.,1.])
alpha = 0.1
gamma = 1.0
# number of episodes
N = 10000
for i in range(N):
    # start in the middle
    initial_position = np.random.choice([1,2,3,4,5])
    terminate = False
    all_positions = [initial_position]
    while not terminate:
        # move from current position to new position
        all_positions.append(move(all_positions[-1]))
        # once in an end state, terminate, update values and start again
        if all_positions[-1]==0 or all_positions[-1]==6:
            Vt = update_values(Vt, np.asarray(all_positions))
            terminate = True

print("Predicted values of states A to G:")
print(Vt)
