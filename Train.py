from math import sqrt
import parl
from gym import Env
import numpy as np
from parl.utils import ReplayMemory, logger
from parl.env.continuous_wrappers import ActionMappingWrapper
from yaml import load, Loader
import os
from time import time

from Model import Model
import numpy as np
# from Algorithm import DDPG
from Agent import Agent
# np.random.seed(1234)
param_path = os.path.join(os.path.dirname(__file__), "training_parameters.yaml")
param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
print(param_dict)

def run_episode(env : Env, agent : parl.Agent, rpm : ReplayMemory, return_time : bool = False):
    actor_loss, critic_loss, Q_value = 0., 0., 0.
    total_reward, steps = 0., 0
    obs = env.reset()

    while True:
        steps += 1

        if np.random.random() < param_dict["EPSILON"]:
            action = np.random.uniform(-1., 1., size=(2,))
        else:

            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype("float32"))
            action = np.squeeze(action)
            # add guassion noise, clip, map to corresponding interval
            action = np.clip(action + np.random.normal(action, 1.), -1., 1.)
            


        next_obs, reward, done, info = env.step(action)

        next_obs[-2] = next_obs[-2] / (20 * sqrt(2))
        
        rpm.append(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
        # do warm up until rpm size reach MEMORY_WARMUP_SIZE
        if rpm.size() > param_dict["MEMORY_WARMUP_SIZE"]:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                batch_terminal = rpm.sample_batch(param_dict["BATCH_SIZE"])

            actor_loss, critic_loss, Q_value = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)
            
            
        
        if done:
            break
    
    

    
    
    return total_reward, steps, actor_loss, critic_loss, Q_value

def evaluate(env : Env, agent : parl.Agent):       
    obs = env.reset()
    total_reward, steps = 0., 0
   
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype("float32"))
        action = np.squeeze(action)
        action = np.clip(action, -1., 1.)
       
        next_obs, reward, done, info = env.step(action)
        next_obs[-2] = next_obs[-2] / (20 * sqrt(2))

        obs = next_obs
        total_reward += reward
        if done:
            break

    return total_reward, info               
