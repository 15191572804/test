#------------------------------------------------------ DDPG ------------------------------------------------------
import paddle
import parl
from paddle import nn
from paddle.nn import functional as F
# actor
# class Actor(parl.Model):
#     def __init__(self, obs_dim, act_dim):
#         super(Actor, self).__init__()
#         # --------------------------------------------
#         # for baseline
#         # self.fc1 = layers.fc(size=512, act="relu")
#         # self.fc2 = layers.fc(size=512, act="relu")
#         # self.fc3 = layers.fc(size=512, act="relu")
#         # self.fc4 = layers.fc(size=act_dim, act="tanh") 
#         # --------------------------------------------
        
#         # --------------------------------------------
#         # our method
#         self.fc1 = nn.Linear(in_features=obs_dim, out_features=32)
#         self.fc2 = nn.Linear(in_features=32, out_features=64)
#         self.fc3 = nn.Linear(in_features=64, out_features=128)
#         self.fc4 = nn.Linear(in_features=128, out_features=32)
#         self.fc5 = nn.Linear(in_features=32, out_features=act_dim)
#         self.res_fc = nn.Linear(in_features=obs_dim, out_features=act_dim)
#         self.bn = nn.BatchNorm(act_dim)
#         # --------------------------------------------



#     def policy(self, obs):
#         out = F.relu(self.fc1(obs))
#         out = self.fc2(out)
#         out = self.fc3(out)
#         out = self.fc4(out)
#         out = self.fc5(out)
#         out = out + self.res_fc(obs)
#         out = self.bn(out)
#         return paddle.tanh(out)
# # critic
# class Critic(parl.Model):
#     def __init__(self, obs_dim, act_dim):
#         super(Critic, self).__init__()
#         # --------------------------------------------
#         # baseline
#         # self.fc1 = layers.fc(size=512, act="relu")
#         # self.fc2 = layers.fc(size=512, act="relu")
#         # self.fc3 = layers.fc(size=512, act="relu")
#         # self.fc4 = layers.fc(size=512, act="relu")
#         # self.fc5 = layers.fc(size=1, act=None)
#         # --------------------------------------------

#         self.obs_fc1 = nn.Linear(in_features=obs_dim, out_features=32)
#         self.obs_fc2 = nn.Linear(in_features=32, out_features=64)
#         self.obs_fc3 = nn.Linear(in_features=64, out_features=128)

#         self.act_fc1 = nn.Linear(in_features=act_dim, out_features=32)
#         self.act_fc2 = nn.Linear(in_features=32, out_features=64)
#         self.act_fc3 = nn.Linear(in_features=64, out_features=128)

#         self.total_fc1 = nn.Linear(in_features=obs_dim + act_dim, out_features=16)
#         self.total_fc2 = nn.Linear(in_features=16, out_features=64)
#         self.total_fc3 = nn.Linear(in_features=64, out_features=128)

#         self.re_fc1 = nn.Linear(in_features=128 * 3, out_features=128)
#         self.re_fc2 = nn.Linear(in_features=128, out_features=256)
#         self.re_fc3 = nn.Linear(in_features=256, out_features=128)
#         self.re_fc4 = nn.Linear(in_features=128, out_features=1)
#         # self.re_fc1 = layers.fc(size=128 * 3, act="tanh")
# #         self.re_fc2 = layers.fc(size=256, act="tanh")
# #         self.re_fc3 = layers.fc(size=128, act="relu")
# #         self.re_fc4 = layers.fc(size=1, act="tanh")
        
#     def value(self, obs, act):
#         obs_out = F.relu(self.obs_fc1(obs))
#         obs_out = self.obs_fc2(obs_out)
#         obs_out = self.obs_fc3(obs_out)

#         act_out = F.relu(self.act_fc1(act))
#         act_out = paddle.tanh(self.act_fc2(act_out))
#         act_out = paddle.tanh(self.act_fc3(act_out))
        
#         total_out = paddle.concat([obs, act], axis=1)
#         total_out = F.relu(self.total_fc1(total_out))
#         total_out = self.total_fc2(total_out)
#         total_out = self.total_fc3(total_out)

#         re_out = paddle.concat([obs_out, act_out, total_out], axis=1)
#         re_out = paddle.tanh(self.re_fc1(re_out))
#         re_out = paddle.tanh(self.re_fc2(re_out))
#         re_out = F.relu(self.re_fc3(re_out))
#         re_out = paddle.tanh(self.re_fc4(re_out))
#         return paddle.squeeze(re_out, axis=[1])

# # integate actor net and critic net together
# class Model(parl.Model):
#     def __init__(self, obs_dim, act_dim):
#         super(Model, self).__init__()
#         self.actor_model = Actor(obs_dim, act_dim)
#         self.critic_model = Critic(obs_dim, act_dim)

#     def policy(self, obs):
#         return self.actor_model.policy(obs)

#     def value(self, obs, act):
#         return self.critic_model.value(obs, act)

#     # get actor's parameter
#     def get_actor_params(self):
#         return self.actor_model.parameters()
    
#     # get actor's parameter
#     def get_critic_params(self):
#         return self.critic_model.parameters()



import paddle
import parl
from paddle import nn
from paddle.nn import functional as F

# actor
class Actor(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        # --------------------------------------------
        # for baseline
        # self.fc1 = layers.fc(size=512, act="relu")
        # self.fc2 = layers.fc(size=512, act="relu")
        # self.fc3 = layers.fc(size=512, act="relu")
        # self.fc4 = layers.fc(size=act_dim, act="tanh") 
        # --------------------------------------------
        
        # --------------------------------------------
        # our method
        self.fc1 = nn.Linear(in_features=obs_dim, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=act_dim)
        self.bn = nn.BatchNorm(act_dim)
        # --------------------------------------------



    def policy(self, obs):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.bn(out)
        return paddle.tanh(out)
# critic
class Critic(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        # --------------------------------------------
        # baseline
        # self.fc1 = layers.fc(size=512, act="relu")
        # self.fc2 = layers.fc(size=512, act="relu")
        # self.fc3 = layers.fc(size=512, act="relu")
        # self.fc4 = layers.fc(size=512, act="relu")
        # self.fc5 = layers.fc(size=1, act=None)
        # --------------------------------------------
        self.fc1 = nn.Linear(in_features=obs_dim, out_features=512)
        self.fc2 = nn.Linear(in_features=512 + act_dim, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=512)
        
    def value(self, obs, act):
        out = F.relu(self.fc1(obs))
        out = paddle.concat([out, act], axis=1)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# integate actor net and critic net together
class Model(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.actor_model = Actor(obs_dim, act_dim)
        self.critic_model = Critic(obs_dim, act_dim)

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    # get actor's parameter
    def get_actor_params(self):
        return self.actor_model.parameters()
    
    # get actor's parameter
    def get_critic_params(self):
        return self.critic_model.parameters()


#------------------------------------------------------ TD3 ------------------------------------------------------
# import parl
# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F


# class Model(parl.Model):
#     def __init__(self, obs_dim, action_dim):
#         super(Model, self).__init__()
#         self.actor_model = Actor(obs_dim, action_dim)
#         self.critic_model = Critic(obs_dim, action_dim)

#     def policy(self, obs):
#         return self.actor_model(obs)

#     def value(self, obs, action):
#         return self.critic_model(obs, action)

#     def Q1(self, obs, action):
#         return self.critic_model.Q1(obs, action)

#     def get_actor_params(self):
#         return self.actor_model.parameters()

#     def get_critic_params(self):
#         return self.critic_model.parameters()


# class Actor(parl.Model):
#     def __init__(self, obs_dim, action_dim):
#         super(Actor, self).__init__()

#         self.l1 = nn.Linear(obs_dim, 512)
#         self.l2 = nn.Linear(512, 256)
#         self.l3 = nn.Linear(256, action_dim)
#         self.bn = nn.BatchNorm(action_dim)

#     def forward(self, obs):
#         x = F.relu(self.l1(obs))
#         x = F.relu(self.l2(x))
#         x = self.l3(x)
#         action = paddle.tanh(self.bn(x))
#         return action


# class Critic(parl.Model):
#     def __init__(self, obs_dim, action_dim):
#         super(Critic, self).__init__()

#         # Q1 architecture
#         self.l1 = nn.Linear(obs_dim + action_dim, 512)
#         self.l2 = nn.Linear(512, 256)
#         self.l3 = nn.Linear(256, 1)

#         # Q2 architecture
#         self.l4 = nn.Linear(obs_dim + action_dim, 512)
#         self.l5 = nn.Linear(512, 256)
#         self.l6 = nn.Linear(256, 1)

#     def forward(self, obs, action):
#         sa = paddle.concat([obs, action], 1)

#         q1 = F.relu(self.l1(sa))
#         q1 = F.relu(self.l2(q1))
#         q1 = self.l3(q1)

#         q2 = F.relu(self.l4(sa))
#         q2 = F.relu(self.l5(q2))
#         q2 = self.l6(q2)
#         return q1, q2

#     def Q1(self, obs, action):
#         sa = paddle.concat([obs, action], 1)

#         q1 = F.relu(self.l1(sa))
#         q1 = F.relu(self.l2(q1))
#         q1 = self.l3(q1)
#         return q1



#------------------------------------------------------ SAC ------------------------------------------------------

# import parl
# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F

# # clamp bounds for Std of action_log
# LOG_SIG_MAX = 2.0
# LOG_SIG_MIN = -20.0


# class Model(parl.Model):
#     def __init__(self, obs_dim, action_dim):
#         super(Model, self).__init__()
#         self.actor_model = Actor(obs_dim, action_dim)
#         self.critic_model = Critic(obs_dim, action_dim)

#     def policy(self, obs):
#         return self.actor_model(obs)

#     def value(self, obs, action):
#         return self.critic_model(obs, action)

#     def get_actor_params(self):
#         return self.actor_model.parameters()

#     def get_critic_params(self):
#         return self.critic_model.parameters()


# class Actor(parl.Model):
#     def __init__(self, obs_dim, action_dim):
#         super(Actor, self).__init__()

#         self.l1 = nn.Linear(obs_dim, 512)
#         self.l2 = nn.Linear(512, 256)
#         self.mean_linear = nn.Linear(256, action_dim)
#         self.std_linear = nn.Linear(256, action_dim)
#         self.bn = nn.BatchNorm(action_dim)

#     def forward(self, obs):
#         x = F.relu(self.l1(obs))
#         x = F.relu(self.l2(x))

#         act_mean = self.bn(self.mean_linear(x))
#         act_std = self.bn(self.std_linear(x))
#         act_log_std = paddle.clip(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#         return act_mean, act_log_std


# class Critic(parl.Model):
#     def __init__(self, obs_dim, action_dim):
#         super(Critic, self).__init__()

#         # Q1 network
#         self.l1 = nn.Linear(obs_dim + action_dim, 512)
#         self.l2 = nn.Linear(512, 256)
#         self.l3 = nn.Linear(256, 1)

#         # Q2 network
#         self.l4 = nn.Linear(obs_dim + action_dim, 512)
#         self.l5 = nn.Linear(512, 256)
#         self.l6 = nn.Linear(256, 1)

#     def forward(self, obs, action):
#         x = paddle.concat([obs, action], 1)

#         # Q1
#         q1 = F.relu(self.l1(x))
#         q1 = F.relu(self.l2(q1))
#         q1 = self.l3(q1)

#         # Q2
#         q2 = F.relu(self.l4(x))
#         q2 = F.relu(self.l5(q2))
#         q2 = self.l6(q2)
#         return q1, q2