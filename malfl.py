from utils import *
import copy
import numpy as np
import torch
import torch.nn as nn
from grid import Grid
from agent import Agent
from policy_improvement import SoftPolicy
from modeling_other_agents import PolicyEstimation

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = ConfigParser()
config.read("config.cfg")


class MALFL(Grid):

    def __init__(self, agent_one: Agent, agent_two: Agent, learning_method: SoftPolicy,
                 policy_estimation: PolicyEstimation):
        super(MALFL, self).__init__()
        self.agent_one = agent_one
        self.agent_two = agent_two
        self.learning_method = learning_method
        self.policy_estimation = policy_estimation
        self.episode_length = int(config['PARAMS']['episode_length'])
        self.alpha = float(config['PARAMS']['alpha'])
        self.gamma = float(config['PARAMS']['gamma'])
        self.beta = float(config['PARAMS']['beta'])
        self.n_iteration = int(config['PARAMS']['n_iteration'])
        self.num_episode = int(config['PARAMS']['num_episode'])
        self.next_state_tensor = next_state_tensor
        self.setNumpySeed()

        self.reward_adam_lr = float(config['REWARDS']['reward_adam_lr'])
        self.reward_adam_epoch = int(config['REWARDS']['reward_adam_epoch'])

        self.first_iteration = True
        self.last_iteration = False

    def setNumpySeed(self):
        seed = config['PARAMS']['seed']
        try:
            np.random.seed(int(seed))
        except ValueError:
            pass

    def choose_action(self, policy, state_agent_one_prime=None, state_agent_two_prime=None, prime=None):
        if not prime:
            action = np.random.choice(self.actions, p=policy[self.agent_one.state, self.agent_two.state, :].flatten())
        else:
            assert isinstance(state_agent_one_prime, int) and isinstance(state_agent_two_prime, int)
            action = np.random.choice(self.actions, p=policy[state_agent_one_prime, state_agent_two_prime, :].flatten())
        return action

    def reward_expectation(self, improved_pi_other_, current_pi_other_, current_pi_self_, agent_type_self):
        improved_pi_other = improved_pi_other_.detach().clone()
        current_pi_other = current_pi_other_.detach().clone()
        current_pi_self = current_pi_self_.detach().clone()

        part_1_for_other_agent_rew_expectation = self.alpha * torch.log(improved_pi_other)

        kl_part1_for_other_agent_rew_expectation = (
                current_pi_other * (torch.log(current_pi_other) - torch.log(improved_pi_other))).sum(-1)
        kl_part1_for_other_agent_rew_expectation = torch.tensordot(
            self.next_state_tensor.float(), kl_part1_for_other_agent_rew_expectation.float(), dims=[[4, 5], [0, 1]])

        if agent_type_self == "agent_one":
            kl_for_other_agent_rew_expectation = (
                    kl_part1_for_other_agent_rew_expectation * current_pi_self.unsqueeze(3)).sum(2)
        else:
            kl_for_other_agent_rew_expectation = (
                    kl_part1_for_other_agent_rew_expectation * current_pi_self.unsqueeze(2)).sum(3)
        part_2_for_other_agent_rew_expectation = self.alpha * self.gamma * kl_for_other_agent_rew_expectation

        other_agent_reward_expectation = part_1_for_other_agent_rew_expectation + part_2_for_other_agent_rew_expectation

        return other_agent_reward_expectation

    def run_episode(self):
        num_step = 0

        while num_step < self.episode_length:
            action_agent_one = self.choose_action(self.agent_one.pi)
            action_agent_two = self.choose_action(self.agent_two.pi)

            r_agent_one = self.agent_one.reward(self.agent_one.state, self.agent_two.state)
            r_agent_two = self.agent_two.reward(self.agent_one.state, self.agent_two.state)
            state_agent_one_prime = self.agent_one.step(action_agent_one, self.agent_one.state)
            state_agent_two_prime = self.agent_two.step(action_agent_two, self.agent_two.state)

            # Next action
            action_agent_one_prime = self.choose_action(self.agent_one.pi, state_agent_one_prime, state_agent_two_prime,
                                                        prime=True)
            action_agent_two_prime = self.choose_action(self.agent_two.pi, state_agent_one_prime, state_agent_two_prime,
                                                        prime=True)
            # Bellman updates for learning the expectation of the Q function directly
            self.learning_method.bellman_update(self, action_agent_one, state_agent_one_prime, state_agent_two_prime,
                                                action_agent_one_prime, r_agent_one, self.agent_one)

            self.learning_method.bellman_update(self, action_agent_two, state_agent_one_prime, state_agent_two_prime,
                                                action_agent_two_prime, r_agent_two, self.agent_two)

            # Keep track of states and action performed by the other agent
            self.agent_one.counter[self.agent_one.state, self.agent_two.state, action_agent_two] += 1
            self.agent_two.counter[self.agent_one.state, self.agent_two.state, action_agent_one] += 1

            self.agent_one.state = state_agent_one_prime
            self.agent_two.state = state_agent_two_prime

            num_step += 1

    def policy_iteration(self):
        # lists of reward expectations
        agent_one_reward_expectation = []
        agent_two_reward_expectation = []
        # list of actual policies
        agent_one_policies = []
        agent_two_policies = []
        # list of estimated policies 
        estimated_agent_one_policies = []
        estimated_agent_two_policies = []

        print("Starting Policy Iteration:", end="\n")
        for i in range(self.n_iteration):
            # Reset the counts
            self.agent_one.counter = np.zeros([self.size * self.size, self.size * self.size, self.n_actions])
            self.agent_two.counter = np.zeros([self.size * self.size, self.size * self.size, self.n_actions])

            self.agent_one.reset_exp_q_val()
            self.agent_two.reset_exp_q_val()
            if i > 0:
                self.first_iteration = False
            if i == self.n_iteration - 1:
                self.last_iteration = True
            # Agents evaluate their own policies and produce trajectories
            for _ in range(self.num_episode):
                self.agent_one.return_to_start()
                self.agent_two.return_to_start()
                self.run_episode()

            self.policy_estimation.estimate_policies_from_trajectories(self)
            estimated_agent_one_pi = copy.deepcopy(self.agent_two.estimated_other_pi)
            estimated_agent_two_pi = copy.deepcopy(self.agent_one.estimated_other_pi)
            # append the estimated policies 
            estimated_agent_one_policies += [torch.tensor(estimated_agent_one_pi, device=device)]
            estimated_agent_two_policies += [torch.tensor(estimated_agent_two_pi, device=device)]

            # Compute expectation of rewards
            current_pi_agent_two, current_pi_agent_one = (
                copy.deepcopy(self.agent_two.pi), copy.deepcopy(self.agent_one.pi))
            agent_one_policies += [torch.tensor(current_pi_agent_one, device=device)]
            agent_two_policies += [torch.tensor(current_pi_agent_two, device=device)]

            # We can compute the reward expectation only starting from the second iteration step
            if i >= 1:
                for agent in [self.agent_one, self.agent_two]:
                    old_pi_other, old_pi_self = (estimated_agent_two_policies[-2], agent_one_policies[-2]) \
                        if agent.agent_type == "agent_one" else (
                        estimated_agent_one_policies[-2], agent_two_policies[-2])

                    improved_pi_other = estimated_agent_two_policies[-1] if agent.agent_type == "agent_one" else \
                        estimated_agent_one_policies[-1]
                    other_rew_exp = self.reward_expectation(improved_pi_other, old_pi_other, old_pi_self,
                                                            agent_type_self=agent.agent_type)
                    if agent.agent_type == 'agent_one':
                        agent_two_reward_expectation += [other_rew_exp]
                    else:
                        agent_one_reward_expectation += [other_rew_exp]

            # Policy improvement
            self.learning_method.soft_policy_improvement(self)

        return agent_two_reward_expectation, agent_one_reward_expectation, agent_one_policies, agent_two_policies

    def loss_fn(self, agent, r_sh, target, p_list, k):
        r_sh_rep = r_sh[k + 1].repeat(1, self.n_actions)
        r_sh_t = torch.reshape(r_sh_rep, (self.size * self.size, self.size * self.size, self.n_actions))
        p_list_repeated_k = torch.reshape(p_list[k].repeat(1, 1, self.n_actions), (
            self.size * self.size, self.size * self.size, self.n_actions, self.n_actions))
        r_sh0 = torch.sum((p_list_repeated_k * r_sh[0]), 2) if agent.agent_type == "agent_one" \
            else torch.sum((p_list_repeated_k * r_sh[0]), 3)

        part_1 = r_sh0 + r_sh_t

        part_2 = r_sh_t.unsqueeze(3).repeat(1, 1, 1, self.n_actions)
        part_2 = part_2.unsqueeze(4).repeat(1, 1, 1, 1, self.n_states)
        part_2 = part_2.unsqueeze(5).repeat(1, 1, 1, 1, 1, self.n_states)

        next_state_t = self.next_state_tensor
        part_2 = part_2 * next_state_t
        part_2 = part_2.sum(-1).sum(-1)

        part_2 = torch.sum((p_list_repeated_k * part_2), 2) if agent.agent_type == "agent_one" \
            else torch.sum((p_list_repeated_k * part_2), 3)
        t_detach = target.detach()
        curr_loss = ((part_1 - self.gamma * part_2 - t_detach) ** 2)
        curr_loss = curr_loss.sum()
        return curr_loss

    def param_regression(self, r_list, p_list, agent):
        # recover state-action reward and shaping
        r_ = nn.Parameter(
            torch.zeros(size=[self.n_states, self.n_states, self.n_actions, self.n_actions], device=device),
            requires_grad=True)
        r_sh = (r_,) + tuple(
            nn.Parameter(torch.zeros(size=[self.n_states, self.n_states], device=device), requires_grad=True) for _ in
            range(self.n_iteration))

        optimizer = torch.optim.Adam(r_sh, lr=self.reward_adam_lr)
        for epoch in range(self.reward_adam_epoch):
            loss = 0
            for k, target in enumerate(r_list):
                loss += self.loss_fn(agent, r_sh, target, p_list, k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        r_ = r_.cpu().detach().numpy()
        return r_

    def reward_recovering(self):
        agent_two_reward_expectation_list, agent_one_reward_expectation_list, agent_one_p_list, agent_two_p_list = \
            self.policy_iteration()
        rew_agent_two = None
        rew_agent_one = None
        for agent in [self.agent_one, self.agent_two]:
            if agent.agent_type == "agent_one":
                rew_agent_two = self.param_regression(agent_two_reward_expectation_list, agent_one_p_list, agent)
            else:
                rew_agent_one = self.param_regression(agent_one_reward_expectation_list, agent_two_p_list, agent)

        self.agent_one.estimated_other_r = rew_agent_two
        self.agent_two.estimated_other_r = rew_agent_one
