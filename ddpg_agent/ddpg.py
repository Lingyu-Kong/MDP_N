import sys 
sys.path.append("..") 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from utils.tensor_utils import to_tensor
from bfgs.bfgs_utils import relax,compute,batch_relax
from ddpg_agent.actor import Actor
from ddpg_agent.critic import Critic

import wandb
import matplotlib.pyplot as plt

# state => 构象
# action => 坐标变化
# transaction => state_next = relax(state+action)
# reward => -(energy(state_next)-energy(state))
# q(s,a) => GNN(s+a)
# p(s) => GNN(s)

class DDPG(object):
    def __init__(
        self,
        actor_gnn_params: dict,
        critic_gnn_params: dict,
        replay_buffer_params: dict,
        num_atoms: int,
        pos_scale: float,
        if_relax: bool,
        max_relax_steps: int,
        latent_size: int,
        mlp_hidden_size: int,
        mlp_layers: int,
        threshold: float,
        gamma: float,
        tau: float,
        lr: float,
        decay_steps: int,
        decay_rate: float,
        if_soft_update: bool,
        if_enhanced_sample: bool,
        if_greedy: bool,
        device:torch.device,
    ):
        self.actor=Actor(actor_gnn_params,pos_scale,latent_size,mlp_hidden_size,mlp_layers,lr,device)
        self.actor_target=Actor(actor_gnn_params,pos_scale,latent_size,mlp_hidden_size,mlp_layers,lr,device)
        self.critic=Critic(critic_gnn_params,latent_size,mlp_hidden_size,mlp_layers,lr,device)
        self.critic_target=Critic(critic_gnn_params,latent_size,mlp_hidden_size,mlp_layers,lr,device)
        self.replay_buffer=ReplayBuffer(**replay_buffer_params)
        self.hard_update(self.actor_target,self.actor)
        self.hard_update(self.critic_target,self.critic)
        self.num_atoms=num_atoms
        self.pos_scale=pos_scale
        self.if_relax=if_relax
        self.max_relax_steps=max_relax_steps
        self.threshold=threshold
        self.gamma=gamma
        self.tau=tau
        self.if_soft_update=if_soft_update
        self.if_enhanced_sample=if_enhanced_sample
        self.if_greedy=if_greedy
        self.device=device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.scheduler_actor=StepLR(self.actor.optimizer,decay_steps,decay_rate)
        self.scheduler_critic=StepLR(self.critic.optimizer,decay_steps,decay_rate)
    
    def update_single_iter(self,batch_size):
        state_batch,action_batch,reward_batch,next_state_batch=self.replay_buffer.batch_read(batch_size)  ## get np data from replay buffer
        ## critic network update
        if self.if_greedy:
            target_q_batch=to_tensor(reward_batch)
        else:
            next_reward_batch=self.critic_target(to_tensor(next_state_batch),self.actor_target.sample_action(to_tensor(next_state_batch)))
            target_q_batch=to_tensor(reward_batch)+self.gamma*next_reward_batch
        q_batch=self.critic(to_tensor(state_batch),to_tensor(action_batch))
        critic_loss=F.mse_loss(q_batch,target_q_batch)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        ## actor network update
        # action_batch_pred=self.actor(to_tensor(state_batch))
        # ## TODO: 下面这个 actor_loss 应不应该乘 -1 ？
        # action_result=batch_relax(to_tensor(state_batch)+action_batch_pred,self.max_relax_steps,batch_size)
        # actor_loss= self.critic(action_result).mean()
        log_prob=self.actor(to_tensor(state_batch),to_tensor(action_batch)).squeeze(-1)
        actor_loss=(log_prob*(to_tensor(reward_batch).squeeze(-1))).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        ## lr update 
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        return critic_loss.item(),actor_loss.item()

    def train(self,num_steps, warmup_steps,batch_size):
        wandb.watch(self.actor.mu_net,log="all",log_freq=1,idx=1)
        wandb.watch(self.actor.sigma_net,log="all",log_freq=1,idx=2)
        wandb.watch(self.actor.gnn,log="all",log_freq=1,idx=3)
        for i in range(num_steps):
            ## sample and fill into replay buffer
            start_time=time.time()
            if self.if_enhanced_sample:
                state,action_1,reward_1,next_state_1,action_2,reward_2,next_state_2=self.enhanced_sample_mdp_quaternion(i,warmup_steps)
                self.replay_buffer.store(state,action_1,reward_1,next_state_1)
                self.replay_buffer.store(state,action_2,reward_2,next_state_2)
            else:
                state,action,reward,next_state=self.sample_mdp_quaternion(i,warmup_steps)
                self.replay_buffer.store(state,action,reward,next_state)
            ## if warmup finished then update network
            if i > warmup_steps:
                critic_loss,actor_loss=self.update_single_iter(batch_size)
                ## update target network
                if self.if_soft_update:
                    self.soft_update(self.actor_target,self.actor,self.tau)
                    self.soft_update(self.critic_target,self.critic,self.tau)
                else:
                    if i%100 ==99:
                        self.hard_update(self.actor_target,self.actor)
                        self.hard_update(self.critic_target,self.critic)
                ## log
                wandb.log({"critic_loss":critic_loss,
                            "actor_loss":actor_loss,})
                if (i-200) % 200 == 199:
                    state_batch,action_batch,reward_batch,_=self.replay_buffer.batch_read(batch_size)
                    ground_truth=reward_batch.squeeze(-1).tolist()
                    predict=self.critic(to_tensor(state_batch),to_tensor(action_batch)).squeeze(-1).tolist()
                    plt.figure()
                    plt.plot(ground_truth,label="ground_truth",color="red")
                    plt.plot(predict,label="predict",color="blue")
                    wandb.log({"critic performance_"+i.__str__():plt})
                    self.actor.save_model("./model_save/actor_"+i.__str__()+".pt")
                    self.critic.save_model("./model_save/critic_"+i.__str__()+".pt")

            end_time=time.time()
            print("=======================================================================")
            print("step ", i, " : finished,    time cost : ",
                end_time-start_time, "s")
            print("=======================================================================")
    
    def test(self,warmup_steps,test_steps):
        for i in range(warmup_steps):
            if self.if_enhanced_sample:
                state,action_1,reward_1,next_state_1,action_2,reward_2,next_state_2=self.enhanced_sample_mdp_quaternion(i,warmup_steps)
                self.replay_buffer.store(state,action_1,reward_1,next_state_1)
                self.replay_buffer.store(state,action_2,reward_2,next_state_2)
            else:
                state,action,reward,next_state=self.sample_mdp_quaternion(i,warmup_steps)
                self.replay_buffer.store(state,action,reward,next_state)
        pred_norms=[]
        real_norms=[]
        res_norm=[]
        for i in range(test_steps):
            state,action,reward,next_state=self.replay_buffer.batch_read(1)
            action_pred,mu,sigma=self.actor.sample_action(to_tensor(state))
            action_pred=action_pred.squeeze(0)
            pred_norm=action_pred.norm().item()
            real_norm=to_tensor(action).norm().item()
            res_norm.append(pred_norm-real_norm)
            pred_norms.append(pred_norm)
            real_norms.append(real_norm)
            print("mu:",mu)
            print("real_action",to_tensor(action))
            # print("sigma:",sigma)
        plt.figure()
        plt.plot(pred_norms,label="pred_norms",color="red")
        plt.plot(real_norms,label="real_norms",color="blue")
        plt.plot(res_norm,label="res_norm",color="green")
        wandb.log({"test_norms":plt})


    def sample_mdp_quaternion(self,current_steps,warmup_steps):
        ## random sample and relax to get a local optimal state
        pos=np.zeros((self.num_atoms,3))
        for i in range(self.num_atoms):
            if_continue=True
            while if_continue:
                new_pos=np.random.rand(3)*2*self.pos_scale-self.pos_scale
                if_continue=False
                for j in range(i):
                    distance=np.linalg.norm(new_pos-pos[j],ord=2)
                    if distance<self.threshold:
                        if_continue=True
                        break
            pos[i,:]=new_pos
        _,energy_0,state=relax(pos.tolist(),self.max_relax_steps)
        state=np.array(state,dtype=np.float64)
        ## sample action:
        if current_steps<warmup_steps:
            action=np.random.rand(self.num_atoms,3)*2*self.pos_scale-self.pos_scale
        else:
            action=self.actor(to_tensor(pos).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        next_state=np.add(state,action)
        if self.if_relax:
            _,energy_1,next_state=relax(next_state.tolist(),self.max_relax_steps)
            next_state=np.array(next_state,dtype=np.float64)
        else:
            energy_1=compute(next_state.tolist())
        reward=energy_0-energy_1
        # print("reward: ",reward)
        return state,action,reward,next_state

    def enhanced_sample_mdp_quaternion(self,current_steps,warmup_steps):
        ## sample state
        pos=np.zeros((self.num_atoms,3))
        for i in range(self.num_atoms):
            if_continue=True
            while if_continue:
                new_pos=np.random.rand(3)*2*self.pos_scale-self.pos_scale
                if_continue=False
                for j in range(i):
                    distance=np.linalg.norm(new_pos-pos[j],ord=2)
                    if distance<self.threshold:
                        if_continue=True
                        break
            pos[i,:]=new_pos
        _,energy_0,state=relax(pos.tolist(),self.max_relax_steps)
        state=np.array(state,dtype=np.float64)
        ## sample action:
        if current_steps<warmup_steps:
            action=np.random.rand(self.num_atoms,3)*2*self.pos_scale-self.pos_scale
        else:
            action=self.actor.sample_action(to_tensor(pos).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        next_state_1=np.add(state,action)
        energy_1=compute(next_state_1.tolist())
        _,energy_2,next_state_2=relax(next_state_1.tolist(),self.max_relax_steps)
        next_state_2=np.array(next_state_2,dtype=np.float64)
        action_1=action
        action_2=np.subtract(next_state_2,state)
        return state,action_1,energy_1,next_state_1,action_2,energy_2,next_state_2

        
    def hard_update(self,target,source):
        for target_param,source_param in zip(target.parameters(),source.parameters()):
            target_param.data.copy_(source_param.data)
    
    def soft_update(self,target,source,tau):
        for target_param,source_param in zip(target.parameters(),source.parameters()):
            target_param.data.copy_(
                target_param.data*(1-tau) + source_param.data*tau
            )

class ReplayBuffer(object):
    def __init__(self,
                 buffer_size:int,
                 num_atoms:int,
                 ):
        self.buffer_size=buffer_size
        self.num_atoms=num_atoms
        self.buffer_top=0
        self.state_buffer=np.zeros((buffer_size,num_atoms,3))
        self.action_buffer=np.zeros((buffer_size,num_atoms,3))
        self.reward_buffer=np.zeros((buffer_size,1))
        self.next_state_buffer=np.zeros((buffer_size,num_atoms,3))

    def store(self,state,action,reward,next_state):
        self.state_buffer[self.buffer_top%self.buffer_size,:,:]=state
        self.action_buffer[self.buffer_top%self.buffer_size,:,:]=action
        self.reward_buffer[self.buffer_top%self.buffer_size,:]=reward
        self.next_state_buffer[self.buffer_top%self.buffer_size,:,:]=next_state
        self.buffer_top=self.buffer_top+1

    def batch_read(self,batch_size):
        choices=np.random.choice(min(self.buffer_size,self.buffer_top),batch_size)
        return self.state_buffer[choices,:,:],self.action_buffer[choices,:,:],self.reward_buffer[choices,:],self.next_state_buffer[choices,:,:]
