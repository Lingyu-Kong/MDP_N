import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import softmax
from torch_scatter import scatter


class MetaLayer(nn.Module):
    def __init__(
        self,
        edge_model,
        node_model,
        global_model,
        aggregate_edges_for_node_fn=None,
        aggregate_edges_for_globals_fn=None,
        aggregate_nodes_for_globals_fn=None,
        node_attn=True,
        emb_dim=None,
        global_attn=False,
    ):
        super().__init__()
        self.edge_model=edge_model
        self.node_model=node_model
        self.global_model=global_model
        self.aggregate_edges_for_node_fn=aggregate_edges_for_node_fn
        self.aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn
        self.aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn
        if node_attn:
            self.node_attn=NodeAttn(emb_dim, num_heads=None)
        else:
            self.node_attn=None
        if global_attn and global_model is not None:
            self.global_node_attn=GlobalAttn(emb_dim, num_heads=None)
            self.global_edge_attn=GlobalAttn(emb_dim, num_heads=None)
        else:
            self.global_node_attn=None
            self.global_edge_attn=None

    def forward(
        self,
        node_attr: torch.Tensor,  ## [batch_size*num_nodes, node_attr_dim]
        edge_attr: torch.Tensor,  ## [batch_size*num_edges, edge_attr_dim]
        global_attr: torch.Tensor,## [batch_size, global_attr_dim]
        edge_index: torch.Tensor, ## [2, batch_size*num_edges]
        node_batch: torch.Tensor, ## [batch_size*num_nodes]
        edge_batch: torch.Tensor, ## [batch_size*num_edges]
        num_nodes: None,
        num_edges: None,
    ):
        row=edge_index[0]
        col=edge_index[1]
        if self.edge_model is not None:
            sent_attributes=node_attr[row]
            receive_attributes=edge_attr[col]
            global_edges=torch.repeat_interleave(global_attr, num_edges, dim=0)
            concat_feat=torch.cat([edge_attr,sent_attributes,receive_attributes,global_edges], dim=1) ## [batch_size*num_edges, edge_attr_dim+node_attr_dim+node_attr_dim+global_attr_dim]
            edge_attr=self.edge_model(concat_feat) ## [batch_size*num_edges, edge_attr_dim]
        
        if self.node_model is not None and self.node_attn is None:
            sent_attributes=self.aggregate_edges_for_node_fn(edge_attr, row, size=node_attr.size(0))
            receive_attributes=self.aggregate_edges_for_node_fn(edge_attr, col, size=node_attr.size(0))
            global_nodes=torch.repeat_interleave(global_attr, num_nodes, dim=0)
            concat_feat=torch.cat([node_attr,sent_attributes,receive_attributes,global_nodes], dim=1) ## [batch_size*num_nodes, node_attr_dim+node_attr_dim+node_attr_dim+global_attr_dim]
            node_attr=self.node_model(concat_feat) ## [batch_size*num_nodes, node_attr_dim]
        elif self.node_model is not None:
            sent_attributes=self.node_attn(node_attr[row],node_attr[col],edge_attr,row,node_attr.size(0))
            receive_attributes=self.node_attn(node_attr[col],node_attr[row],edge_attr,col,node_attr.size(0))
            global_nodes=torch.repeat_interleave(global_attr, num_nodes, dim=0)
            concat_feat=torch.cat([node_attr,sent_attributes,receive_attributes,global_nodes], dim=1) ## [batch_size*num_nodes, node_attr_dim+node_attr_dim+node_attr_dim+global_attr_dim]
            node_attr=self.node_model(concat_feat) ## [batch_size*num_nodes, node_attr_dim]

        if self.global_model is not None and self.global_node_attn is None:
            node_attributes=self.aggregate_nodes_for_globals_fn(node_attr, node_batch, size=global_attr.size(0))
            edge_attributes=self.aggregate_edges_for_globals_fn(edge_attr, edge_batch, size=global_attr.size(0))
            concat_feat=torch.cat([global_attr,node_attributes,edge_attributes], dim=-1) ## [batch_size, global_attr_dim+node_attr_dim+edge_attr_dim]
            global_attr=self.global_model(concat_feat) ## [batch_size, global_attr_dim]
        elif self.global_model is not None:
            node_attributes=self.global_node_attn(torch.repeat_interleave(global_attr, num_nodes, dim=0),node_attr,node_batch,global_attr.size(0))
            edge_attributes=self.global_edge_attn(torch.repeat_interleave(global_attr, num_edges, dim=0),edge_attr,edge_batch,global_attr.size(0))
            concat_feat=torch.cat([global_attr,node_attributes,edge_attributes], dim=-1)
            global_attr=self.global_model(concat_feat)
        
        return node_attr, edge_attr, global_attr



class DropoutIfTraining(nn.Module):
    def __init__(self, dropout_rate=0.0, submodule=None):
        super().__init__()
        assert dropout_rate>=0.0 and dropout_rate<=1.0
        self.dropout_rate=dropout_rate
        self.submodule=submodule if submodule is not None else nn.Identity()

    def forward(self, x):
        x = self.submodule(x)
        ones=x.new_ones((x.size(0),1))
        ones=F.dropout(ones, p=self.dropout_rate, training=self.training)
        return x*ones

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activate=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list=[]
        if not use_bn:
            if layernorm_before:
                module_list.appen(nn.LayerNorm(input_size))
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(activate())
                    if dropout>0.0:
                        module_list.append(nn.Dropout(dropout))
                input_size=size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activate())
                input_size=size
        self.module_list=nn.Sequential(*module_list)
        self.reset_parameters()
    
    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x

class MLPwoLastAct(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))

            if dropout > 0:
                module_list.append(nn.Dropout(dropout))
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(activation())
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x

class NodeAttn(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0

        self.w1=nn.Linear(3*emb_dim, emb_dim)
        self.w2=nn.Parameter(torch.zeros((self.num_heads, self.emb_dim // self.num_heads)))
        self.w3=nn.Linear(2*emb_dim, emb_dim)
        self.head_dim=self.emb_dim // self.num_heads
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w3.weight, gain=1 / math.sqrt(2))

    def forward(self, q, k_v, k_e, index, nnode):
        x = torch.cat([q, k_v, k_e], dim=1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)

        v = torch.cat([k_v, k_e], dim=1)
        v = self.w3(v).view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=nnode)
        return x

class GlobalAttn(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0
        self.w1 = nn.Linear(2 * emb_dim, emb_dim)
        self.w2 = nn.Parameter(torch.zeros(self.num_heads, self.emb_dim // self.num_heads))
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))

    def forward(self, q, k, index, dim_size):
        x = torch.cat([q, k], dim=1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)

        v = k.view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=dim_size)
        return x

class MLPwoLastAct(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))

            if dropout > 0:
                module_list.append(nn.Dropout(dropout))
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(activation())
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x