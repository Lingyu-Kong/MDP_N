import sys 
sys.path.append("..") 
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
import torch.nn.functional as F
from model.metalayer import MetaLayer,MLP,DropoutIfTraining,MLPwoLastAct
from utils.graph_utils import get_edges_batch

_REDUCER_NAMES = {
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}

class DMCG_NN_Pos_Fixed(nn.Module):
    def __init__(
        self,
        device: torch.device,
        lr: float,
        num_atoms: int,
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        latent_size: int = 256,
        use_layer_norm: bool = False,
        num_message_passing_steps: int = 12,
        global_reducer: str = "sum",
        node_reducer: str = "sum",
        dropedge_rate: float = 0.1,
        dropnode_rate: float = 0.1,
        dropout: float = 0.1,
        layernorm_before: bool = False,
        use_bn: bool = False,
        cycle: int = 1,
        node_attn: bool = False,
        global_attn: bool = False,
    ):
        super().__init__()
        self.gnn=nn.ModuleList()
        for i in range(num_message_passing_steps):
            edge_model=DropoutIfTraining(
                dropout_rate=dropedge_rate,
                submodule=MLP(
                    latent_size*4,
                    [mlp_hidden_size]*mlp_layers+[latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=dropout,
                    use_bn=use_bn,
                ),
            )
            node_model=DropoutIfTraining(
                dropout_rate=dropnode_rate,
                submodule=MLP(
                    latent_size*4,
                    [mlp_hidden_size]*mlp_layers+[latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=dropout,
                    use_bn=use_bn,
                ),
            )
            if i == num_message_passing_steps - 1:
                global_model=MLP(
                    latent_size*3,
                    [mlp_hidden_size]*mlp_layers+[latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=dropout,
                    use_bn=use_bn,
                )
            else:
                global_model=MLP(
                    latent_size*3,
                    [mlp_hidden_size]*mlp_layers+[latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=dropout,
                    use_bn=use_bn,
                )
            self.gnn.append(
                MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    node_attn=node_attn,
                    emb_dim=latent_size,
                    global_attn=global_attn,
                )
            )
        self.PosEN=MLP(
            num_atoms-1,
            [mlp_hidden_size]*mlp_layers+[latent_size]
        )
        self.PosEE=MLP(
            1,
            [mlp_hidden_size]*mlp_layers+[latent_size]
        )
        self.PosEG=MLP(
            num_atoms*(num_atoms-1),
            [mlp_hidden_size]*mlp_layers+[latent_size]
        )

        # self.final_global_model=MLP(
        #     latent_size,
        #     [mlp_hidden_size]*mlp_layers+[1],
        # )
        self.latent_size=latent_size
        self.dropout=dropout ## used in residual update
        self.cycle=cycle
        self.device=device
        self.to(device)


    def forward(self,conforms):
        """
        conforms: [batch_size,num_atoms,3]
        """
        node_attr,edge_attr,global_attr,edge_index=self.attr_init(conforms)
        pos=conforms.view(-1,3).to(self.device)
        node_batch=torch.repeat_interleave(torch.arange(conforms.shape[0]),conforms.shape[1]).to(self.device)
        edge_batch=torch.repeat_interleave(torch.arange(conforms.shape[0]),conforms.shape[1]*(conforms.shape[1]-1)).to(self.device)
        for i, layer in enumerate(self.gnn):
            if i == len(self.gnn) -1 :
                cycle=self.cycle
            else:
                cycle=1
            for _ in range(cycle):
                # node_attr_ext,edge_attr_ext,global_attr_ext,_=self.attr_init(conforms)
                # node_attr=node_attr+node_attr_ext
                # edge_attr=edge_attr+edge_attr_ext
                # global_attr=global_attr+global_attr_ext
                node_attr_1,edge_attr_1,global_attr_1=layer(
                    node_attr,edge_attr,global_attr,edge_index,node_batch,edge_batch,conforms.shape[1],conforms.shape[1]*(conforms.shape[1]-1))
                node_attr=node_attr+F.dropout(node_attr_1,p=self.dropout,training=self.training)
                edge_attr=edge_attr+F.dropout(edge_attr_1,p=self.dropout,training=self.training)
                global_attr=global_attr+F.dropout(global_attr_1,p=self.dropout,training=self.training)
        return node_attr,edge_attr,global_attr ## [batch_size*num_atoms,latent_size],[batch_size*num_edges,latent_size],[batch_size,latent_size]

        
    
    def attr_init(self,conforms):
        """
        conforms: [batch_size,num_atoms,3]
        distance_node=[batch_size*num_atoms,num_atoms-1]
        distance_edge=[batch_size*num_edges,1]
        node_attr=[batch_size*num_atoms,latent_size]
        edge_attr=[batch_size*num_edges,latent_size]
        global_attr=[batch_size,latent_size]
        """
        # print(conforms.shape)
        edge_index=get_edges_batch(conforms.shape[1],conforms.shape[0])
        pos=conforms.view(-1,3)
        distance_edge=torch.norm(pos[edge_index[0]]-pos[edge_index[1]],p=2,dim=1)
        distance_node=torch.zeros((conforms.shape[0]*conforms.shape[1],conforms.shape[1]-1))
        for i in range(conforms.shape[0]*conforms.shape[1]):
            distance_node[i]=torch.FloatTensor([distance_edge[j] for j in range((i*(conforms.shape[1]-1)),((i+1)*(conforms.shape[1]-1)))])
        distance_node=distance_node.to(self.device)
        distance_edge=distance_edge.to(self.device)
        node_attr=self.PosEN(distance_node)
        edge_attr=self.PosEE(distance_edge.unsqueeze(-1))
        global_attr=self.PosEG(distance_edge.reshape(conforms.shape[0],(conforms.shape[1]*(conforms.shape[1]-1))))
        return node_attr.to(self.device),edge_attr.to(edge_attr),global_attr.to(self.device),edge_index.to(self.device)

class DMCG_NN_Pos_Changed(nn.Module):
    def __init__(
        self,
        device: torch.device,
        lr: float,
        num_atoms: int,
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        latent_size: int = 256,
        use_layer_norm: bool = False,
        num_message_passing_steps: int = 12,
        global_reducer: str = "sum",
        node_reducer: str = "sum",
        dropedge_rate: float = 0.1,
        dropnode_rate: float = 0.1,
        dropout: float = 0.1,
        layernorm_before: bool = False,
        use_bn: bool = False,
        cycle: int = 1,
        node_attn: bool = False,
        global_attn: bool = False,
    ):
        super().__init__()
        self.gnn=nn.ModuleList()
        self.pos_update=nn.ModuleList()
        for i in range(num_message_passing_steps):
            edge_model=DropoutIfTraining(
                dropout_rate=dropedge_rate,
                submodule=MLP(
                    latent_size*4,
                    [mlp_hidden_size]*mlp_layers+[latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=dropout,
                    use_bn=use_bn,
                ),
            )
            node_model=DropoutIfTraining(
                dropout_rate=dropnode_rate,
                submodule=MLP(
                    latent_size*4,
                    [mlp_hidden_size]*mlp_layers+[latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=dropout,
                    use_bn=use_bn,
                ),
            )
            if i == num_message_passing_steps - 1:
                global_model=MLP(
                    latent_size*3,
                    [mlp_hidden_size]*mlp_layers+[latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=dropout,
                    use_bn=use_bn,
                )
            else:
                global_model=MLP(
                    latent_size*3,
                    [mlp_hidden_size]*mlp_layers+[latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=dropout,
                    use_bn=use_bn,
                )
            self.gnn.append(
                MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    node_attn=node_attn,
                    emb_dim=latent_size,
                    global_attn=global_attn,
                )
            )
            self.pos_update.append(
                MLPwoLastAct(latent_size,[latent_size,3])
            )
        self.PosEN=MLP(
            num_atoms-1,
            [mlp_hidden_size]*mlp_layers+[latent_size]
        )
        self.PosEE=MLP(
            1,
            [mlp_hidden_size]*mlp_layers+[latent_size]
        )
        self.PosEG=MLP(
            num_atoms*(num_atoms-1),
            [mlp_hidden_size]*mlp_layers+[latent_size]
        )

        self.final_global_model=MLP(
            latent_size,
            [mlp_hidden_size]*mlp_layers+[1],
        )
        self.latent_size=latent_size
        self.dropout=dropout ## used in residual update
        self.cycle=cycle
        # self.optimizer=optim.Adam(self.parameters(),lr=lr)
        self.device=device
        self.to(device)


    def forward(self,conforms):
        """
        conforms: [batch_size,num_atoms,3]
        """
        node_attr,edge_attr,global_attr,edge_index=self.attr_init(conforms)
        pos=conforms.view(-1,3).to(self.device)
        node_batch=torch.repeat_interleave(torch.arange(conforms.shape[0]),conforms.shape[1]).to(self.device)
        edge_batch=torch.repeat_interleave(torch.arange(conforms.shape[0]),conforms.shape[1]*(conforms.shape[1]-1)).to(self.device)
        for i, layer in enumerate(self.gnn):
            if i == len(self.gnn) -1 :
                cycle=self.cycle
            else:
                cycle=1
            for _ in range(cycle):
                node_attr_ext,edge_attr_ext,global_attr_ext=self.attr_init(conforms)
                node_attr=node_attr+node_attr_ext
                edge_attr=edge_attr+edge_attr_ext
                global_attr=global_attr+global_attr_ext
                node_attr_1,edge_attr_1,global_attr_1=layer(
                    node_attr,edge_attr,global_attr,edge_index,node_batch,edge_batch,conforms.shape[1],conforms.shape[1]*(conforms.shape[1]-1))
                node_attr=node_attr+F.dropout(node_attr_1,p=self.dropout,training=self.training)
                edge_attr=edge_attr+F.dropout(edge_attr_1,p=self.dropout,training=self.training)
                global_attr=global_attr+F.dropout(global_attr_1,p=self.dropout,training=self.training)
                pos=self.move2origin(pos+self.pos_update[i](node_attr),node_batch,conforms.shape[1])
                conforms=pos.view(conforms.shape[0],conforms.shape[1],3)
        global_attr=self.final_global_model(global_attr)
        return global_attr ## [batch_size,1]

    def move2origin(self,pos,node_batch,num_atoms):
        pos_mean=global_mean_pool(pos,node_batch)
        pos_mean=torch.repeat_interleave(pos_mean,num_atoms,dim=0)
        return pos-pos_mean  
    
    def attr_init(self,conforms):
        """
        conforms: [batch_size,num_atoms,3]
        distance_node=[batch_size*num_atoms,num_atoms-1]
        distance_edge=[batch_size*num_edges,1]
        node_attr=[batch_size*num_atoms,latent_size]
        edge_attr=[batch_size*num_edges,latent_size]
        global_attr=[batch_size,latent_size]
        """
        edge_index=get_edges_batch(conforms.shape[1],conforms.shape[0])
        pos=conforms.view(-1,3)
        distance_edge=torch.norm(pos[edge_index[0]]-pos[edge_index[1]],p=2,dim=1)
        distance_node=torch.zeros((conforms.shape[0]*conforms.shape[1],conforms.shape[1]-1))
        for i in range(conforms.shape[0]*conforms.shape[1]):
            distance_node[i]=torch.FloatTensor([distance_edge[j] for j in range((i*(conforms.shape[1]-1)),((i+1)*(conforms.shape[1]-1)))])
        distance_node=distance_node.to(self.device)
        distance_edge=distance_edge.to(self.device)
        node_attr=self.PosEN(distance_node)
        edge_attr=self.PosEE(distance_edge.unsqueeze(-1))
        global_attr=self.PosEG(distance_edge.reshape(conforms.shape[0],(conforms.shape[1]*(conforms.shape[1]-1))))
        return node_attr.to(self.device),edge_attr.to(edge_attr),global_attr.to(self.device),edge_index.to(self.device)