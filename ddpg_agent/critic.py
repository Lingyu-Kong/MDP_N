import torch
import torch.nn as nn
import torch.optim as optim
from model.dmcg_nn import DMCG_NN_Pos_Fixed
from model.metalayer import MLPwoLastAct

class Critic(nn.Module):
    def __init__(self,gnn_params,latent_size,mlp_hidden_size,mlp_layers,lr,device):
        super().__init__()
        self.gnn=DMCG_NN_Pos_Fixed(**gnn_params)
        self.final_global=MLPwoLastAct(
            input_size=latent_size,
            output_sizes=[mlp_hidden_size]*mlp_layers+[1],
            use_layer_norm=False,
            activation=nn.ReLU,
            dropout=0.0,
            layernorm_before=False,
            use_bn=False,
        )
        self.device=device
        self.final_global.to(self.device)
        self.optimizer=optim.Adam(self.parameters(),lr=lr)
        self.weight_init(self.final_global)
        self.weight_init(self.gnn)
    
    def forward(self,state_batch,action_batch):
        conforms=state_batch+action_batch
        _,_,global_attr=self.gnn(conforms)
        final_global=self.final_global(global_attr)
        return final_global

    def evaluate(self,conforms):
        _,_,global_attr=self.gnn(conforms)
        final_global=self.final_global(global_attr)
        return final_global

    def save_model(self,path):
        torch.save(self.state_dict(),path)
    
    def load_model(self,path):
        self.load_state_dict(torch.load(path))
    
    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
