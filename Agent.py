import paddle
import parl
from parl.core.fluid import layers
import paddle.fluid as fluid
import numpy as np

class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.alg.sync_target(decay=0)

    # def build_program(self):
    #     self.pred_program = fluid.Program()
    #     self.learn_program = fluid.Program()

    #     with fluid.program_guard(self.pred_program):
    #         obs = layers.data(name="obs", shape=[self.obs_dim], dtype="float32")
    #         self.pred_act = self.alg.predict(obs)

    #     with fluid.program_guard(self.learn_program):
    #         obs = layers.data(name="obs", shape=[self.obs_dim], dtype="float32")
    #         act = layers.data(name="act", shape=[self.act_dim], dtype="float32")
    #         reward = layers.data(name="reward", shape=[], dtype="float32")
    #         next_obs = layers.data(name="next_obs", shape=[self.obs_dim], dtype="float32")
    #         terminal = layers.data(name="terminal", shape=[], dtype="bool")
    #         _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs, terminal)

    def predict(self, obs):
        # obs = np.expand_dims(obs, axis=0)
        # act = self.fluid_executor.run(
        #     self.pred_program,
        #     feed = {"obs":obs.astype("float32")},
        #     fetch_list = [self.pred_act]
        # )[0]
        
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        
        return action_numpy

    def learn(self, obs, act, reward, next_obs, terminal):
        # feed = {
        #     "obs":obs,
        #     "act":act,
        #     "reward":reward,
        #     "next_obs":next_obs,
        #     "terminal":terminal
        # }
        # critic_cost = self.fluid_executor.run(
        #     self.learn_program,
        #     feed = feed,
        #     fetch_list = [self.critic_cost]
        # )[0]
        # self.alg.sync_target()
        # return critic_cost
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(act, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        # critic_loss = self.alg.learn(obs, action, reward, next_obs,
        #                                          terminal)
        # return critic_loss
        critic_loss, actor_loss, Q_value = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss, Q_value
    
    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
        """
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, 1.0, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action