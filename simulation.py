from configparser import ConfigParser
import itertools
import os
import argparse


parser = argparse.ArgumentParser("simulation")
parser.add_argument("--config_file", type=str, default="experiments.cfg", help="config file")
arglist = parser.parse_args()

config_file = arglist.config_file
experiments = ConfigParser()
experiments.read(config_file)

l = []
for section in experiments.sections():
    for key in experiments[section]:
        l.append(experiments[section][key].split(","))

p = list(itertools.product(*l))
for combination in p:
    config_file = experiments
    counter = 0
    for section in experiments.sections():
        for key in experiments[section]:
            config_file[section][key] = combination[counter]
            counter += 1
    with open("config.cfg", 'w') as f:
        config_file.write(f)
    os.system("python main.py")
    os.remove("config.cfg")
