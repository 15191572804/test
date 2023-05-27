from Train import *
from env.maplessNaviEnv import *
import datetime
import warnings
import paddle
import numpy as np
import parl
from parl.env import ActionMappingWrapper
from parl.utils import summary
from parl.algorithms.paddle import ddpg
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # 默认值，输出所有信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 屏蔽通知信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 屏蔽通知信息和警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 屏蔽通知信息、警告信息和报错信息

scene_name = "plane_static_obstacle-A"

# make sure to save all the model files to a dir named today's date
today = datetime.date.today()
today_str = f"{today.year}.{today.month}.{today.day}({scene_name})"
if not os.path.exists(f"./machineLearning/ReinforcementLearning/RobotObstacleAvoidance/MaplessNavigation-main/MaplessNavigation-main/model/{today_str}"):
    os.makedirs(f"./machineLearning/ReinforcementLearning/RobotObstacleAvoidance/MaplessNavigation-main/MaplessNavigation-main/model/{today_str}")       

env = MaplessNaviEnv(scene_name=scene_name, render=True, seedNum=247787180)
env = ActionMappingWrapper(env)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]


# create 3-level model
model = Model(obs_dim, act_dim)
# DDPG
algorithm = ddpg.DDPG(
    model=model,
    gamma=param_dict["GAMMA"],
    tau=param_dict["TAU"],
    actor_lr=param_dict["ACTOR_LR"],
    critic_lr=param_dict["CRITIC_LR"]
)

# TD3
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

# SAC
# GAMMA = 0.99
# TAU = 0.005
# ACTOR_LR = 3e-4
# CRITIC_LR = 3e-4
# ALPHA = 0.2
# algorithm = parl.algorithms.SAC(
#         model,
#         gamma=GAMMA,
#         tau=TAU,
#         alpha=ALPHA,
#         actor_lr=ACTOR_LR,
#         critic_lr=CRITIC_LR)

agent = Agent(
    algorithm=algorithm,
    obs_dim=obs_dim,
    act_dim=act_dim
)

# replay memory vessel
rpm = ReplayMemory(int(param_dict["MEMORY_SIZE"]), obs_dim, act_dim)

# persistence training
save_path = ""
if os.path.exists(save_path):
    print(f"\033[33msuccessfully load existing model\033[0m: {save_path}")
    agent.restore(save_path=save_path)

test_flag, total_steps = 0, 0
while total_steps < param_dict["TRAIN_TOTAL_STEPS"]:
    train_reward, steps, actor_loss, critic_loss, Q_value = run_episode(env, agent, rpm, return_time=True)
    total_steps += steps
    # print(total_steps)
    # print("train_reward:", train_reward)
    # because total_steps are not increased step by step, we need to set a threshold instead of use mod
    if total_steps >= param_dict["MEMORY_WARMUP_SIZE"]:
        summary.add_scalar('actor_loss', actor_loss.item(), total_steps)
        summary.add_scalar('critic_loss', critic_loss.item(), total_steps)
        summary.add_scalar('Q_value', max(Q_value).item(), total_steps)
    summary.add_scalar('train_reward', round(train_reward, 2), total_steps)
    if total_steps // param_dict["TEST_EVERY_STEPS"] >= test_flag:
        test_flag += total_steps // param_dict["TEST_EVERY_STEPS"] - test_flag + 1
        evaluate_reward, info = evaluate(env, agent)
        
        log_str = "Steps: {} | Test reward: {} | distance: {} | collision : {}".format(
            total_steps, round(evaluate_reward, 2), round(info["distance"], 2), info["collision_num"]
        )
        summary.add_scalar('test_reward', round(evaluate_reward, 2), total_steps)
        logger.info(log_str)
        # print(f"\033[34m{time_info}\033[0m")
        with open(f"./machineLearning/ReinforcementLearning/RobotObstacleAvoidance/MaplessNavigation-main/MaplessNavigation-main/model/{today_str}/train.log", "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

        # save model
        # if save_path == "":
            # save_path = f"./machineLearning/ReinforcementLearning/RobotObstacleAvoidance/MaplessNavigation-main/MaplessNavigation-main/model/{today_str}/s_{total_steps}_r_{round(evaluate_reward, 0)}"
        save_path = "./machineLearning/ReinforcementLearning/RobotObstacleAvoidance/MaplessNavigation-main/MaplessNavigation-main/model/{}/s_{}_r_{}.ckpt".format(today_str, total_steps, int(evaluate_reward))
        agent.save(save_path=save_path)

f.close()
# import os
# os.system("shutdown /s /t 1")