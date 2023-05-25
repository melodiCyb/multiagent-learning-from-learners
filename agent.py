from grid import Grid
from scipy import spatial
import numpy as np


class Agent(Grid):

    def __init__(self, agent_type, reward_function):
        super(Agent, self).__init__()
        self.agent_type = agent_type
        self.reward_function = reward_function
        # initialize q values
        self.q = np.zeros((self.n_states, self.n_states, self.n_actions, self.n_actions), np.float32)
        # initialize expected q values
        self.exp_q = np.zeros((self.n_states, self.n_states, self.n_actions), np.float32)
        # initialize policies
        self.pi = np.ones((self.n_states, self.n_states, self.n_actions), np.float32) / self.n_actions
        # We let the agent start always from the top left corner
        self.state = self.start
        self.estimated_other_pi = np.ones((self.n_states, self.n_states, self.n_actions), np.float32) / self.n_actions
        # initialize estimation of the other agent reward
        self.estimated_other_r = np.zeros((self.n_states, self.n_states, self.n_actions, self.n_actions), np.float32)
        # initialize estimation of the other agent expected Q function
        self.estimated_other_exp_q = np.zeros((self.n_states, self.n_states, self.n_actions), np.float32)
        # initialize prediction of next soft policy improvement for the other agent
        self.prediction_other_improvement = np.ones((self.n_states, self.n_states, self.n_actions),
                                                    np.float32) / self.n_actions
        # matrix to count the state and actions of the other agent observed during an iteration step
        self.counter = np.zeros((self.n_states, self.n_states, self.n_actions), np.float32)

    def return_to_start(self):
        self.state = self.start

    def reset_exp_q_val(self):
        self.exp_q = np.zeros((self.n_states, self.n_states, self.n_actions), np.float32)

    def reward(self, agent_one_state, agent_two_state):
        function = self.reward_function
        agent_one_pos, agent_two_pos = np.array(self.idx2state[agent_one_state]), np.array(
            self.idx2state[agent_two_state])
        if self.agent_type == "agent_one":
            own_pos, other_pos = agent_one_pos, agent_two_pos
        else:
            own_pos, other_pos = agent_two_pos, agent_one_pos
        terminal_pos = np.array(self.idx2state[self.terminal_state])

        # negated manhattan distance between the agent and the terminal position minus the negated manhattan distance
        # between the agent and the other agent (in both cases: the closer, the better)
        if function == "manhattan_joint":
            return -spatial.distance.cityblock(own_pos, terminal_pos) - spatial.distance.cityblock(own_pos, other_pos)
        # manhattan_disjoint makes the agent go to the goal location while trying to avoid each others
        elif function == "manhattan_disjoint":
            return -spatial.distance.cityblock(own_pos, terminal_pos) + spatial.distance.cityblock(own_pos, other_pos)

    def true_reward_function(self):
        true_rewards = np.zeros((self.n_states, self.n_states))
        for agent_one_state in range(self.n_states):
            for agent_two_state in range(self.n_states):
                true_rewards[agent_one_state, agent_two_state] = self.reward(agent_one_state, agent_two_state)
        return true_rewards
