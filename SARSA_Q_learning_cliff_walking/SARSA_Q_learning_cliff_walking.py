"""
Script to execute the cliff walking example from Sutton & Barto 2014
(https://towardsdatascience.com/reinforcement-learning-cliff-walking-implementation-e40ce98418d4)
Use SARSA and Q learning to teach an agent to find the optimal path around a cliff
without falling off the cliff.
Compare the policies which are obtained from SARSA and Q learning, and evaluate why each
algorithm reaches the policy that it does.
"""

class Cliff():
    """
    Class to keep track of the Agent's position, execute actions and give rewards

    Board looks like:
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    S * * * * * * * * * * G
    """
    def reset(self):
        self.pos = [0,-3] # start in bottom left corner
        self.board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      ["S", "*", "*", "*", "*", "*", "*", "*", "*", "*", "*", "G"]]
        self.end = False

    def next_pos(self, action):
        if action=="up":
            new_pos = self.pos + [0, 1]
        elif action=="down":
            new_pos = self.pos + [0, -1]
        elif action=="left":
            new_pos = self.pos + [-1, 0]
        elif action=="right":
            new_pos = self.pos + [1, 0]
        else:
            raise Exception("Unknown action attempted")

        # check validity of move
        if new_pos[0] >= 0 and new_pos[0] <= 11:
            if new_pos[1] >= -3 and new_pos[1] <= 0:
                self.pos = new_pos

        if self.board[self.pos] == "G":
            self.end = True
            if self.verbose:
                print("Agent has reached the goal, game ends!")
        elif self.board[self.pos] == "*":
            self.end = True
            if self.verbose:
                print("Agent has fallen off the cliff, game ends :(")

    def give_reward(self):
        if self.board[self.pos] == 0:
            return -1
        elif self.board[self.pos] == "G":
            return -1
        elif self.board[self.pos] == "*":
            return -100


class Agent():
