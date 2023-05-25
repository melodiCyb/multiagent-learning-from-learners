from configparser import ConfigParser
from malfl import MALFL, SoftPolicy, Agent, PolicyEstimation, Grid
from utils import saveCSV


config = ConfigParser()
config.read('config.cfg')


def main():
    reward_function_agent_one = config['REWARDS']['agent_one_function'].lower()
    reward_function_agent_two = config['REWARDS']['agent_two_function'].lower()
    agent_one = Agent("agent_one", reward_function_agent_one)
    agent_two = Agent("agent_two", reward_function_agent_two)
    learning_method = SoftPolicy()
    env = Grid()
    policy_estimation = PolicyEstimation(env)
    ma_lfl = MALFL(agent_one, agent_two, learning_method, policy_estimation)
    ma_lfl.reward_recovering()

    if config.getboolean('REWARDS', 'analysis'):
        saveCSV(ma_lfl)


if __name__ == '__main__':
    main()
