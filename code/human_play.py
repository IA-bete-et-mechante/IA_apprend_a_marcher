import gym
import pygame
from gym.utils.play import play
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


# define the keys to play
if "Pacman" in config["env"]:
    mapping = {(pygame.K_UP,): 1, (pygame.K_LEFT,): 3, (pygame.K_DOWN,): 4, (pygame.K_RIGHT,): 2}
elif "Lunar" in config["env"]:
    mapping = {(pygame.K_UP,): 0, (pygame.K_LEFT,): 1, (pygame.K_DOWN,): 2, (pygame.K_RIGHT,): 3}


#launch the play
play(gym.make(config["env"]), keys_to_action=mapping)
