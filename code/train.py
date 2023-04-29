import gym
from stable_baselines3 import PPO, A2C
from sb3_contrib import TQC
import os
import os
import yaml
from datetime import datetime



#### Load the config in a dict called "config" 
CONFIG_PATH = "./config/"

def load_yaml(config_name):
    """Function to load yaml configuration file"""
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_yaml("my_config.yaml")


# datetime object containing current date and time
now = datetime.now().strftime("%Y_%m_%d_at_%Hh-%Mm-%Ss")

models_dir = "models/" + config['model'] + "/" + now
log_dir = "logs/" + config['model'] + "/" + now

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# env = gym.make(config["env"], frame_skip = 0)
env = gym.make(config["env"])
env.reset()

# model = TQC(
#     policy = config["policy"],
#     env = config["env"],
#     learning_rate = config["learning_rate"],
#     gamma = config["gamma"],
#     gae_lambda= config["gae_lambda"],
#     ent_coef= config["ent_coef"],
#     vf_coef = config["vf_coef"],
#     max_grad_norm = config["max_grad_norm"],
#     use_rms_prop = config["use_rms_prop"],
#     use_sde = config["use_sde"],
#     normalize_advantage = config["normalize_advantage"],
#     tensorboard_log = log_dir,
#     verbose = 1
#     )

policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300])
model = TQC("MlpPolicy", env, gamma = 0.98, learning_rate = 0.00073, tau = 0.02, use_sde = True, policy_kwargs=policy_kwargs)



total_timesteps = config['total_timesteps']
checkpoint = config['checkpoint_freq']
nb_iter = total_timesteps // checkpoint
# iters = 0
for i in range(nb_iter):
    print(i)
    model.learn(total_timesteps=checkpoint, reset_num_timesteps=False, tb_log_name= config["log_name_tensorboard"])
    model.save(f"{models_dir}/{checkpoint*(i+1)}")
