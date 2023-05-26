# -*- encoding: utf-8 -*-
# author : Zhelong Huang
import sys
import os
path = os.path.dirname(__file__)
path = "/".join(path.split("\\")[:-1])
sys.path.append(path)
import numpy as np
import gym
from gym import spaces
from robot.utils import *
from robot.miniBox import *
from robot.scene import *

class MaplessNaviEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, scene_name : str = "plane_static_obstacle-A", render : bool = False, evaluate : bool = False, seedNum : int = None):
        """
            :param scene_name: 场景名称(场景是否存在的判断逻辑在_register_scene的construct方法中)
            :param render:     是否需要渲染，训练情况下为了更快的训练速度，一般设为False
            :param evaluate:   是否为评估模式，评估模式下会绘制终点标记
        """
        self.all_scene = ["plane_static_obstacle-A", "plane_static_obstacle-B", "plane_static_obstacle-C"]
        self.scene_name = scene_name
        self._render = render
        self._evaluate = evaluate
        # 读入各项参数
        for file in os.listdir("./machineLearning/ReinforcementLearning/RobotObstacleAvoidance/MaplessNavigation-main/MaplessNavigation-main/robot/config"):
            param_path = os.path.join("./machineLearning/ReinforcementLearning/RobotObstacleAvoidance/MaplessNavigation-main/MaplessNavigation-main/robot/config/", file)
            param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
            for key, value in param_dict.items():
                setattr(self, key, value)

        # 动作空间: 左轮速度， 右轮速度
        self.action_space = spaces.Box(
            low=np.array([-self.TARGET_VELOCITY, -self.TARGET_VELOCITY]),
            high=np.array([self.TARGET_VELOCITY, self.TARGET_VELOCITY]),
            dtype=np.float32
        )
        # 状态空间: laser1, ..., 5,   distance, alpha
        
        # self.observation_space = spaces.Box(
        #     low=np.array([0.] * self.LASER_NUM  + [0., 0.]),
        #     high=np.array([self.LASER_LENGTH + 1] * self.LASER_NUM + [self.MAX_DISTANCE, np.pi])
        # )


        self.observation_space = spaces.Box(
            low=np.array([0.] * self.LASER_NUM + [-self.TARGET_VELOCITY, -self.TARGET_VELOCITY] + [0., 0.]),
            high=np.array([self.LASER_LENGTH + 1] * self.LASER_NUM + [self.TARGET_VELOCITY, self.TARGET_VELOCITY] + [self.MAX_DISTANCE, np.pi])
        )
        
        # self.observation_space = spaces.Box(
        #     low=np.array([0.] * self.LASER_NUM + [-self.TARGET_VELOCITY, -self.TARGET_VELOCITY] +  [0., 0.] + [0., 0.]),
        #     high=np.array([self.LASER_LENGTH + 1] * self.LASER_NUM + [self.TARGET_VELOCITY, self.TARGET_VELOCITY] + [2*np.pi, 2*np.pi] +  [self.MAX_DISTANCE, np.pi])
        # )
        # 根据参数选择引擎的连接方式
        self._physics_client_id = p.connect(p.GUI if render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # 获取注册环境并reset
        self._register_scene = RegisterScenes()
        self.seed(seedNum)
        self.reset()
    
    def __reward_func(self, state : list):
        if checkCollision(self.robot.robot, debug=False):
            self.collision_num += 1
            Rc = self.COLLISION_REWARD
        else:
            Rc = 0
        
        cur_dis = self.__distance(self.robot.curPos(), self.TARGET_POS)
        Rp = self.DISTANCE_CHANGE_REWARD_COE * (self.pre_dis - cur_dis)
        self.pre_dis = cur_dis

        if state[-2] < self.TARGET_RADIUS:
            Rr = self.REACH_TARGET_REWARD
        else:
            Rr = 0.
        
        target_angle = state[-1]
        Ra = target_angle * self.ANGLE_REWARD
        
        return Rc + Rp + Rr + Ra

    # def __reward_func(self, state: list):
    #     Rc_min, Rc_max = self.COLLISION_REWARD, 0
    #     if checkCollision(self.robot.robot, debug=False):
    #         self.collision_num += 1
    #         Rc = self.COLLISION_REWARD
    #     else:
    #         Rc = 0

    #     # Normalize collision reward
    #     Rc_norm = -1 if Rc == -100 else 1

    #     cur_dis = self.__distance(self.robot.curPos(), self.TARGET_POS)
    #     Rp = self.DISTANCE_CHANGE_REWARD_COE * (self.pre_dis - cur_dis)
    #     self.pre_dis = cur_dis

    #     Rp_min, Rp_max = -5000, 5000
    #     # Normalize distance progress reward
    #     Rp_norm = (Rp - Rp_min) / (Rp_max - Rp_min)

    #     if state[-2] < self.TARGET_RADIUS:
    #         Rr = self.REACH_TARGET_REWARD
    #     else:
    #         Rr = 0.
    #     Rr_min, Rr_max = 0, 120
    #     # Normalize reaching target reward
    #     Rr_norm = (Rr - Rr_min) / (Rr_max - Rr_min)

    #     target_angle = state[-1]
    #     Ra = target_angle * self.ANGLE_REWARD
    #     Ra_min, Ra_max = -0.1, 0
    #     # Normalize angle reward
    #     Ra_norm = 2 * (Ra - Ra_min) / (Ra_max - Ra_min) - 1

    #     return Rc_norm + Rp_norm + Rr_norm + Ra_norm


        # return Rc + Rp + Rr + Rt
       
        # laser_distance = np.sum(state[:15])
        # target_distance = state[-2]
        # target_angle = state[-1]
        # collision = 0
        # success = 0
        # if target_distance <= 0.5:
        #     success = 1
        # if checkCollision(self.robot.robot, debug=False):
        #     self.collision_num += 1
        #     collision=1
        # reward = self.laser_distance * laser_distance \
        #      + self.target_distance * target_distance \
        #      + self.target_angle * target_angle \
        #      + self.collision * collision \
        #      + self.success * success
        # return reward

        


    def __distance(self, v1, v2):
        return sqrt(sum([(x - y) * (x - y) for x, y in zip(v1, v2)]))
    
    def sample(self):
        return self.np_random.uniform(low=-TARGET_VELOCITY, high=TARGET_VELOCITY, size=(2,))
    
    def step(self, action):
        """
            first set, second step
            then calculate the reward
            return state, reward, done, info
        """
        # print(int(action[0]),int(action[1]))
        self.robot.apply_action(action=action)
        p.stepSimulation(physicsClientId=self._physics_client_id)    
        self.step_num += 1
        state = self.robot.get_observation(self.TARGET_POS)
        reward = self.__reward_func(state)
        self.robot.get_last_speed(action)
        if state[-2] < self.TARGET_RADIUS:
            done = True
        elif self.step_num > self.DONE_STEP_NUM:
            done = True
        elif self.collision_num >= self.DONE_COLLISION:
            done = True
        else:
            done = False
        
        info = {"distance" : state[-2], "collision_num" : self.collision_num}

        # under evaluate mode, extra debug items need to be rendered
        if self._evaluate:
            froms, tos, results = rayTest(self.robot.robot, ray_length=self.LASER_LENGTH, ray_num=self.LASER_NUM)
            for index, result in enumerate(results):
                self.rayDebugLineIds[index] = p.addUserDebugLine(
                    lineFromXYZ=froms[index], 
                    lineToXYZ=tos[index] if result[0] == -1 else result[3], 
                    lineColorRGB=self.MISS_COLOR if result[0] == -1 else self.HIT_COLOR, 
                    lineWidth=self.RAY_DEBUG_LINE_WIDTH, 
                    replaceItemUniqueId=self.rayDebugLineIds[index]
                )

        return np.array(state), reward, done, info
    
    def reset(self):
        """
            what you need do here:
            - reset scene items
            - reload robot
        """
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        p.setRealTimeSimulation(0)
        self.step_num = 0
        # print("\033[32m enter \033[0m")
        self.collision_num = 0
        self.pre_dis = self.__distance(self.DEPART_POS, self.TARGET_POS)                    # previous distance between robot and target
        self.depart_target_distance = self.__distance(self.DEPART_POS, self.TARGET_POS)     # distance between depart pos and target pos

        self.robot = Robot(
            basePos=self.DEPART_POS,
            baseOri=p.getQuaternionFromEuler(self.DEPART_EULER),
            physicsClientId=self._physics_client_id
        )
        if self.scene_name == "mix":
            self.scene_name = np.random.choice(self.all_scene)
        self.scene = self._register_scene.construct(scene_name=self.scene_name)
        state = self.robot.get_observation(targetPos=self.TARGET_POS)

        
       # add debug items to the target pos
        if self._evaluate:
            self.target_line = p.addUserDebugLine(
                lineFromXYZ=[self.TARGET_POS[0], self.TARGET_POS[1], 0.],
                lineToXYZ=[self.TARGET_POS[0], self.TARGET_POS[1], 5.],
                lineColorRGB=[1., 1., 0.2]
            )
            self.rayDebugLineIds = []
            froms, tos, results = rayTest(self.robot.robot, ray_length=self.LASER_LENGTH, ray_num=self.LASER_NUM)
            for index, result in enumerate(results):
                color = self.MISS_COLOR if result[0] == -1 else self.HIT_COLOR
                self.rayDebugLineIds.append(p.addUserDebugLine(froms[index], tos[index], color, self.RAY_DEBUG_LINE_WIDTH))

        return np.array(state)
    
    def render(self, mode='human'):
        pass
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1
    

if __name__ == "__main__":
    env = MaplessNaviEnv()
    # pprint([attr for attr in dir(env) if attr[:2] != "__"])

    from stable_baselines3.common.env_checker import check_env
    check_env(env)
