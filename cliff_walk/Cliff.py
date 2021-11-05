import numpy as np

class Cliff():
    """
    SARSA vs. Q learning cliff walking example.
    Class to keep track of the Agent's position, execute actions and give rewards.

    Board looks like:
    |-----------------------|
    |0 0 0 0 0 0 0 0 0 0 0 0|
    |0 0 0 0 0 0 0 0 0 0 0 0|
    |0 0 0 0 0 0 0 0 0 0 0 0|
    |S * * * * * * * * * * G|
    |-----------------------|
    """
    def __init__(self):
        self.pos = (3, 0) # start in bottom left corner
        self.board = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 ["S", "*", "*", "*", "*", "*", "*", "*", "*", "*", "*", "G"]])
        self.end = False
        self.verbose = True

    def reset(self):
        self.pos = (3, 0) # start in bottom left corner
        self.end = False

    def check_valid_move(self, new_pos):
        # check validity of move
        if new_pos[0] >= 0 and new_pos[0] <= 3:
            if new_pos[1] >= 0 and new_pos[1] <= 11:
                valid = True
            else:
                # the attempted move is invalid, don't move the agent
                valid = False
        else:
            # the attempted move is invalid, don't move the agent
            valid = False
        return valid

    def move_agent(self, action):
        if action=="up":
            new_pos = (self.pos[0] - 1, self.pos[1])
        elif action=="down":
            new_pos = (self.pos[0] + 1, self.pos[1])
        elif action=="left":
            new_pos = (self.pos[0], self.pos[1] - 1)
        elif action=="right":
            new_pos = (self.pos[0], self.pos[1] + 1)
        else:
            raise Exception("Unknown action attempted")

        # check validity of move
        if self.check_valid_move(new_pos):
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
        if self.board[self.pos] == "0":
            return -1
        elif self.board[self.pos] == "G":
            return -1
        elif self.board[self.pos] == "S":
            return -1
        elif self.board[self.pos] == "*":
            return -100

    def sim_next_reward(self,action):
        """
        Calculate reward in the next state given some action, without actually moving
            the agent.
        """
        if action=="up":
            new_pos = (self.pos[0] - 1, self.pos[1])
        elif action=="down":
            new_pos = (self.pos[0] + 1, self.pos[1])
        elif action=="left":
            new_pos = (self.pos[0], self.pos[1] - 1)
        elif action=="right":
            new_pos = (self.pos[0], self.pos[1] + 1)
        else:
            raise Exception("Unknown action attempted")

        # check validity of move
        if not self.check_valid_move(new_pos):
            # the attempted move is invalid, don't move the agent
            new_pos = self.pos
        if self.board[new_pos] == "0":
            return -1
        elif self.board[new_pos] == "G":
            return -1
        elif self.board[new_pos] == "S":
            return -1
        elif self.board[new_pos] == "*":
            return -100
