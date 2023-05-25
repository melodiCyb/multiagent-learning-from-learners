from scipy import stats
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser("correlation analysis")
parser.add_argument("--outer_folder", type=str, default="simulation_results", help="parent directory of the results")
parser.add_argument("--basename", type=str, default="results", help="base folder name for the results")
arglist = parser.parse_args()

outer_folder = arglist.outer_folder
l = len(arglist.basename)
dir_list = sorted(os.listdir(arglist.outer_folder), key=lambda d: int(d[l:]))
m = []
for directory in dir_list:
    N = int(directory[l:])
    folder = outer_folder + "/" + directory + "/"
    agent_one_estimated_rewards = pd.read_csv(folder + "agent_one_estimated_rewards.csv", header=None)
    states_n = agent_one_estimated_rewards.shape[0]
    positions_n = int(states_n ** 0.5)
    agent_one_estimated_rewards_aggregated = np.array(agent_one_estimated_rewards.mean(axis=1)).reshape(positions_n,
                                                                                                        positions_n)
    agent_one_estimated_rewards_flattened = agent_one_estimated_rewards_aggregated.flatten()
    agent_one_true_rewards = pd.read_csv(folder + "agent_one_true_rewards.csv", header=None)
    agent_one_true_rewards_flattened = np.array(agent_one_true_rewards).flatten()
    try:
        agent_one_pearson = stats.pearsonr(agent_one_true_rewards_flattened, agent_one_estimated_rewards_flattened)[0]
        agent_one_spearman = stats.spearmanr(agent_one_true_rewards_flattened, agent_one_estimated_rewards_flattened)[0]
    except:
        print("An error occurred with results " + str(N) + "! Cannot compute correlation for agent_one!")
        agent_one_pearson = float("NaN")
        agent_one_spearman = float("NaN")

    agent_two_estimated_rewards = pd.read_csv(folder + "agent_two_estimated_rewards.csv", header=None)
    agent_two_estimated_rewards_aggregated = np.array(agent_two_estimated_rewards.mean(axis=1)).reshape(positions_n,
                                                                                                        positions_n)
    agent_two_estimated_rewards_flattened = agent_two_estimated_rewards_aggregated.flatten()
    agent_two_true_rewards = pd.read_csv(folder + "agent_two_true_rewards.csv", header=None)
    agent_two_true_rewards_flattened = np.array(agent_two_true_rewards).flatten()
    try:
        agent_two_pearson = stats.pearsonr(agent_two_true_rewards_flattened, agent_two_estimated_rewards_flattened)[0]
        agent_two_spearman = stats.spearmanr(agent_two_true_rewards_flattened, agent_two_estimated_rewards_flattened)[0]
    except:
        print("An error occurred with results " + str(N) + "! Cannot compute correlation for agent_two!")
        agent_two_pearson = float("NaN")
        agent_two_spearman = float("NaN")
    m.append([N, agent_one_pearson, agent_one_spearman, agent_two_pearson, agent_two_spearman])

df = pd.DataFrame(m, columns=["Number", "agent_one_Pearson", "agent_one_Spearman", "agent_two_Pearson",
                              "agent_two_Spearman"])
df.to_csv(os.path.join("correlation.csv"), index=False)
