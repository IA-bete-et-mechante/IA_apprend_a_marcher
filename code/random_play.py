import gym
import os
import yaml


#### Load the config in a dict called "config" 
CONFIG_PATH = "./config/"

def load_yaml(config_name):
    """Function to load yaml configuration file"""
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_yaml("my_config.yaml")


#launch the environement
env = gym.make(config["env"])
env.reset()

#play 10 episodes of 200 steps each
episodes = 10
for ep in range(episodes):
	env.reset()
	for step in range(200):
		env.render()
		# take random action
		env.step(env.action_space.sample())




env.close()
