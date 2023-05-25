import numpy as np
from configparser import ConfigParser

config = ConfigParser()
config.read("config.cfg")


class Grid(object):
    """This class implements a grid MDP."""

    def __init__(self):
        self.size = int(config['PARAMS']['grid_size'])
        self.start = int(config['PARAMS']['start'])

        # 'D', 'U', 'L', 'R', 'N'
        self.actions = [0, 1, 2, 3, 4]
        self.directions = np.array([[1, -1, 0, 0, 0], [0, 0, -1, 1, 0]])

        self.terminal_state = int(self.size * self.size - 1)
        self.n_actions = len(self.actions)
        self.n_states = self.size * self.size

        self.idx2state = {}
        idx = 0
        for row in range(self.size):
            for column in range(self.size):
                self.idx2state[idx] = (row, column)
                idx += 1

    def reset(self):
        return self.start

    def step(self, action, state):
        d_y, d_x = self.directions[:, action]
        state_y, state_x = self.idx2state[state]
        # state prime
        if state != self.terminal_state:
            next_state = max(0, min(self.size - 1, state_x + d_x)) + max(0, min(
                self.size - 1, state_y + d_y)) * self.size
        else:
            next_state = self.start

        return int(next_state)
