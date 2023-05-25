from torch.distributions.categorical import Categorical
from configparser import ConfigParser
import torch
import torch.nn as nn
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = ConfigParser()
config.read("config.cfg")


class PolicyEstimation:
    def __init__(self, env):
        self.policy_estimation_model = config['POLICY_ESTIMATION']['model']
        self.entr_coeff = float(config['POLICY_ESTIMATION']['entr_coeff'])  # entropy coefficient for policy est
        self.adam_lr = float(config['POLICY_ESTIMATION']['adam_lr'])
        self.adam_epoch = int(config['POLICY_ESTIMATION']['adam_epoch'])
        self.env = env
        self.setTorchSeed()

    def setTorchSeed(self):
        seed = config['PARAMS']['seed']
        try:
            torch.manual_seed(int(seed))
        except ValueError:
            pass

    def maximum_likelihood(self, malfl):
        """
         Entropy regularized maximum likelihood estimation with Adam optimizer
        """
        for agent in [malfl.agent_one, malfl.agent_two]:
            parameters = [
                nn.Parameter(torch.rand(size=[self.env.n_states, self.env.n_states, self.env.n_actions], device=device),
                             requires_grad=True)]
            optimizer = torch.optim.Adam(parameters, lr=self.adam_lr)
            C = torch.tensor(agent.counter, device=device)
            for epoch in range(self.adam_epoch):
                dist = Categorical(torch.exp(parameters[0]))
                log_probs = torch.log(dist.probs)
                entropy_matrix = dist.entropy()
                loss = - ((log_probs * C).sum() + self.entr_coeff * (entropy_matrix * C.sum(-1)).sum())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            agent.estimated_other_pi = Categorical(torch.exp(parameters[0]).cpu()).probs.detach().numpy()

    def estimate_policies_from_trajectories(self, malfl=None):
        if self.policy_estimation_model == "MLE":
            self.maximum_likelihood(malfl)
        elif self.policy_estimation_model == 'TEST':
            # use the actual policies
            malfl.agent_two.estimated_other_pi = copy.deepcopy(malfl.agent_one.pi)
            malfl.agent_one.estimated_other_pi = copy.deepcopy(malfl.agent_two.pi)
        else:
            print("select one from ['TEST', 'MLE'] for policy estimation")
