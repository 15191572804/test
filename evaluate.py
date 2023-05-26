# -*- encoding: utf-8 -*-
# author : Zhelong Huang
import sys
import os

path = os.path.dirname(__file__)
path = "/".join(path.split("\\")[:-1])
sys.path.append(path)
import time as tm
from Train import *
from env.maplessNaviEnv import *
from parl.env import ActionMappingWrapper

env = MaplessNaviEnv(scene_name='plane_static_obstacle-A', render=True, evaluate=True, seedNum=247787180)
env = ActionMappingWrapper(env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# create 3-level model
model = Model(obs_dim, act_dim)
algorithm = parl.algorithms.DDPG(
    model=model,
    gamma=param_dict["GAMMA"],
    tau=param_dict["TAU"],
    actor_lr=param_dict["ACTOR_LR"],
    critic_lr=param_dict["CRITIC_LR"]
)

# GAMMA = 0.99
# TAU = 0.005
# ACTOR_LR = 3e-4
# CRITIC_LR = 3e-4
# POLICY_FREQ = 2
# algorithm = parl.algorithms.TD3(
#     model,
#     gamma=GAMMA,
#     tau=TAU,
#     actor_lr=ACTOR_LR,
#     critic_lr=CRITIC_LR,
#     policy_freq=POLICY_FREQ)
agent = Agent(
    algorithm=algorithm,
    obs_dim=obs_dim,
    act_dim=act_dim
)

# ./model/2021.2.25/s_72001_r_963976
# ./model/2021.2.26/s_592286_r_884427
# ./model/2021.2.26/s_615055_r_977890
# ./model/2021.2.26/s_759057_r_700960
# ./model/2021.2.26/s_778768_r_881551s

agent.restore("./machineLearning/ReinforcementLearning/RobotObstacleAvoidance/MaplessNavigation-main/MaplessNavigation-main/model/DDPG/s_409048_r_1001550")
# tm.sleep(10)
reward, info = evaluate(env, agent)
print(reward)
print(info)
# p.setRealTimeSimulation(1)
# obs = env.reset()
# while True:
#     action = agent.predict(obs)[0]
#     obs, reward, done, info = env.step(action=action)
#     if done:
#         break
