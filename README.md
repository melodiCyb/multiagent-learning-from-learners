# MA-LfL

Source code for "Multi-Agent Learning from Learners" by Mine Melodi Caliskan, Francesco Chini and Setareh Maghsudi.

## How to run

````
conda env create -f environment.yml
conda activate multi-lfl38
python simulation.py --config_file experiments.cfg
````

In order to calculate correlations run the following 
````
python correlation.py
````

## References
Reward recovering is adapted from https://github.com/alexis-jacq/Learning_from_a_Learner

