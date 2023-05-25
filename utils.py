from configparser import ConfigParser
import numpy as np
import os
import shutil
import torch
from grid import Grid
from itertools import product, starmap

config = ConfigParser()
config.read("config.cfg")

g = Grid()

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

next_state_tensor = torch.zeros(size=[g.n_states, g.n_states, g.n_actions, g.n_actions, g.n_states, g.n_states],
                                device=device)


def getFolderName():
    outer_folder = "simulation_results"
    if not os.path.isdir(outer_folder):
        os.makedirs(outer_folder)
    dir_list = os.listdir(outer_folder)
    index = 0
    if len(dir_list):
        l = len("results")
        index = max([int(file_name[l:]) for file_name in dir_list])
    return outer_folder + "/results" + str(index + 1)


def saveCSV(ma_lfl):
    folder = getFolderName()
    os.makedirs(folder)

    estimated_policies = [ma_lfl.agent_two.estimated_other_pi, ma_lfl.agent_one.estimated_other_pi]
    estimated_rewards = [ma_lfl.agent_two.estimated_other_r, ma_lfl.agent_one.estimated_other_r]
    true_policies = [ma_lfl.agent_one.pi, ma_lfl.agent_two.pi]
    true_rewards = [ma_lfl.agent_one.true_reward_function(), ma_lfl.agent_two.true_reward_function()]

    estimated_policies = [policy.reshape(-1, g.n_actions) for policy in estimated_policies]
    estimated_rewards = [rewards.reshape(-1, g.n_actions ** 2) for rewards in estimated_rewards]
    true_policies = [policy.reshape(-1, g.n_actions) for policy in true_policies]
    np.savetxt(folder + "/agent_one_estimated_policy.csv", estimated_policies[0], delimiter=",")
    np.savetxt(folder + "/agent_one_true_policy.csv", true_policies[0], delimiter=",")
    np.savetxt(folder + "/agent_one_estimated_rewards.csv", estimated_rewards[0], delimiter=",")
    np.savetxt(folder + "/agent_one_true_rewards.csv", true_rewards[0], delimiter=",")
    np.savetxt(folder + "/agent_two_estimated_policy.csv", estimated_policies[1], delimiter=",")
    np.savetxt(folder + "/agent_two_true_policy.csv", true_policies[1], delimiter=",")
    np.savetxt(folder + "/agent_two_estimated_rewards.csv", estimated_rewards[1], delimiter=",")
    np.savetxt(folder + "/agent_two_true_rewards.csv", true_rewards[1], delimiter=",")

    shutil.copy("config.cfg", folder + "/config.cfg")


def assign_val(state_1, state_2, action_1, action_2, state_prime_1, state_prime_2):
    if state_prime_1 == g.step(action_1, state_1) and state_prime_2 == g.step(action_2, state_2):
        next_state_tensor[state_1, state_2, action_1, action_2, state_prime_1, state_prime_2] = 1
    else:
        next_state_tensor[state_1, state_2, action_1, action_2, state_prime_1, state_prime_2] = 0


list(starmap(assign_val,
             product(range(g.n_states), range(g.n_states), range(g.n_actions), range(g.n_actions), range(g.n_states),
                     range(g.n_states))))
