import numpy as np


class SoftPolicy:

    def soft_policy_improvement(self, malfl):
        """
        Perform the soft policy improvement using the expected Q function agent.exp_q

        """

        # agent_one
        # Exponent of expected Q function
        # exp-normalize for numerical stability via max val
        agent_one_max_exp_q = (malfl.agent_one.exp_q / malfl.alpha).max(-1).reshape(malfl.n_states, malfl.n_states, 1)
        new_policy_agent_one = np.exp((malfl.agent_one.exp_q / malfl.alpha) - agent_one_max_exp_q)
        z_agent_one = new_policy_agent_one.sum(2)  # Normalizing constants
        z_agent_one = np.repeat(z_agent_one[:, :, np.newaxis], malfl.n_actions, axis=2)
        new_policy_agent_one = new_policy_agent_one / z_agent_one
        malfl.agent_one.pi = new_policy_agent_one

        # agent_two
        agent_two_max_exp_q = (malfl.agent_two.exp_q / malfl.alpha).max(-1).reshape(malfl.n_states, malfl.n_states, 1)
        new_policy_agent_two = np.exp((malfl.agent_two.exp_q / malfl.alpha) - agent_two_max_exp_q)
        z_agent_two = new_policy_agent_two.sum(2)
        z_agent_two = np.repeat(z_agent_two[:, :, np.newaxis], malfl.n_actions, axis=2)
        new_policy_agent_two = new_policy_agent_two / z_agent_two
        malfl.agent_two.pi = new_policy_agent_two

    def bellman_update(self, malfl, action, state_agent_one_prime, state_agent_two_prime, action_prime, r, agent=None):
        """
        Bellman update for learning the expected Q function
        """
        target = r + malfl.gamma * (
                agent.exp_q[state_agent_one_prime, state_agent_two_prime, action_prime]
                - malfl.alpha * np.log(agent.pi[state_agent_one_prime, state_agent_two_prime, action_prime]))
        agent.exp_q[malfl.agent_one.state, malfl.agent_two.state, action] += malfl.beta * (
                target - agent.exp_q[malfl.agent_one.state, malfl.agent_two.state, action])

