import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as Dist
from model.metalayer import MLPwoLastAct
from model.dmcg_nn import DMCG_NN_Pos_Fixed

"""
使用 Gaussian Distribution 定义 Actor
"""


class Actor(nn.Module):
    def __init__(self,gnn_params,pos_scale,latent_size,mlp_hidden_size,mlp_layers,lr,device):
        super().__init__()
        self.pos_scale=pos_scale
        self.gnn=DMCG_NN_Pos_Fixed(**gnn_params)
        self.mu_net=MLPwoLastAct(
            input_size=latent_size,
            output_sizes=[mlp_hidden_size]*mlp_layers+[3],
            use_layer_norm=False,
            activation=nn.ReLU,
            dropout=0.0,
            layernorm_before=False,
            use_bn=False,
        )
        self.mu_last_activate=nn.Tanh()
        self.sigma_net=MLPwoLastAct(
            input_size=latent_size,
            output_sizes=[mlp_hidden_size]*mlp_layers+[3],
            use_layer_norm=False,
            activation=nn.ReLU,
            dropout=0.0,
            layernorm_before=False,
            use_bn=False,
        )
        self.sigma_last_activate=nn.Sigmoid()
        self.device=device
        self.mu_net.to(self.device)
        self.sigma_net.to(self.device)
        self.optimizer=optim.Adam(self.parameters(),lr=lr)
        self.weight_init(self.mu_net)
        self.weight_init(self.sigma_net)
        self.weight_init(self.gnn)

    def forward(self,state_batch,action_batch):
        """
        state_batch: [batch_size,num_atoms,3]
        action_batch: [batch_size,num_atoms,3]
        """
        node_attr,_,_=self.gnn(state_batch)
        mu=self.mu_net(node_attr)
        mu=self.mu_last_activate(mu)*self.pos_scale
        sigma=self.sigma_net(node_attr)
        sigma=(self.sigma_last_activate(sigma)+1e-8)*self.pos_scale
        normal=Dist.Normal(mu,sigma)
        log_prob=normal.log_prob(action_batch.view(-1,3))
        log_prob=log_prob.view(state_batch.shape[0],state_batch.shape[1]*3).sum(dim=1)
        # prob=log_prob.exp()
        return log_prob
    
    def sample_action(self,state_batch):
        """
        state_batch: [batch_size,num_atoms,3]
        """
        node_attr,_,_=self.gnn(state_batch)
        mu=self.mu_net(node_attr)
        mu=self.mu_last_activate(mu)*self.pos_scale
        sigma=self.sigma_net(node_attr)
        sigma=(self.sigma_last_activate(sigma)+1e-8)*self.pos_scale
        normal=Dist.Normal(mu,sigma)
        action=normal.sample()
        return action.view(state_batch.shape[0],state_batch.shape[1],3)

    def relax(self,xyz,max_steps):
        for i in range(max_steps):
            action=self.forward(xyz.unsqueeze(0)).squeeze(0)
            xyz=xyz+action
        return xyz

    def save_model(self,path):
        torch.save(self.state_dict(),path)

    def load_model(self,path):
        self.load_state_dict(torch.load(path))

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        
        