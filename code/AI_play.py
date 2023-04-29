import gym
from stable_baselines3 import PPO, A2C
from sb3_contrib import TQC
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

models_dir = "models/"

env = gym.make(config["env"])
env.reset()

# model_path = f"{models_dir}/3000000.zip"
model_path = config["model_path"]
model_type = config["model_type"]

if model_type == "PPO":
    model = PPO.load(model_path, env=env)
elif model_type == "A2C":
    model = A2C.load(model_path, env=env)
elif model_type == "TQC":
    model = TQC.load(model_path, env=env)
else: 
    print(f"Model type unknown. You need to modify the code in AI_play.py to add this model type: {model_type}")

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        # action, _states = model.predict(obs, deterministic = True)
        action, _states = model.predict(obs, deterministic = True)
        obs, rewards, done, info = env.step(action)

        print(rewards)
