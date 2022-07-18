import argparse
import torch
from ddpg_agent.ddpg import DDPG
import wandb

wandb.login()
wandb.init(project="MDP_N", entity="kly20")

parser = argparse.ArgumentParser()

## shared arguments
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--lr",type=float,default=5e-4)
parser.add_argument("--decay_steps", type=int, default=200)
parser.add_argument("--decay_rate", type=float, default=0.9)
parser.add_argument('--pos_scale', type=float, default=2.0)
parser.add_argument("--num_atoms", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--threshold", type=float, default=1)
parser.add_argument("--cuda", type=bool, default=True)
## GNN arguments
parser.add_argument("--num_message_passing_steps", type=int, default=12)
parser.add_argument("--mlp_hidden_size",type=int,default=512)
parser.add_argument("--mlp_layers",type=int,default=2)
parser.add_argument("--latent_size",type=int,default=256)
parser.add_argument("--use_layer_norm",type=bool,default=False)
parser.add_argument("--global_reducer",type=str,default="sum")
parser.add_argument("--node_reducer",type=str,default="sum")
parser.add_argument("--dropedge_rate",type=float,default=0.1)
parser.add_argument("--dropnode_rate",type=float,default=0.1)
parser.add_argument("--dropout",type=float,default=0.1)
parser.add_argument("--layernorm_before",type=bool,default=False)
parser.add_argument("--use_bn",type=bool,default=False)
parser.add_argument("--cycle",type=int,default=1)
parser.add_argument("--node_attn",type=bool,default=True)
parser.add_argument("--global_attn",type=bool,default=True)
## Replay Buffer arguments
parser.add_argument("--buffer_size",type=int,default=2000)
## BFGS arguments
parser.add_argument("--max_relax_steps",type=int,default=200)
parser.add_argument("--if_relax",type=bool,default=True)
## DDPG arguments
parser.add_argument("--gamma",type=int,default=0.99)
parser.add_argument("--tau",type=float,default=0.001)
parser.add_argument("--warmup_steps",type=int,default=200)
parser.add_argument("--if_soft_update",type=bool,default=True)
parser.add_argument("--if_enhanced_sample",type=bool,default=True)
parser.add_argument("--if_greedy",type=bool,default=True)

args=parser.parse_args()

device=torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")

wandb.config = {
    "network structure":"DMCG",
    "num_atoms":args.num_atoms,
    "pos_scale": args.pos_scale,
    "training_steps": args.num_steps,
    "batch_size": args.batch_size,
}

actor_gnn_params={
    "device":device,
    "lr":args.lr,
    "num_atoms":args.num_atoms,
    "mlp_hidden_size":args.mlp_hidden_size,
    "mlp_layers":args.mlp_layers,
    "latent_size":args.latent_size,
    "use_layer_norm":args.use_layer_norm,
    "num_message_passing_steps":args.num_message_passing_steps,
    "global_reducer":args.global_reducer,
    "node_reducer":args.node_reducer,
    "dropedge_rate":args.dropedge_rate,
    "dropnode_rate":args.dropnode_rate,
    "dropout":args.dropout,
    "layernorm_before":args.layernorm_before,
    "use_bn":args.use_bn,
    "cycle":args.cycle,
    "node_attn":args.node_attn,
    "global_attn":args.global_attn
}

critic_gnn_params={
    "device":device,
    "lr":args.lr,
    "num_atoms":args.num_atoms,
    "mlp_hidden_size":args.mlp_hidden_size,
    "mlp_layers":args.mlp_layers,
    "latent_size":args.latent_size,
    "use_layer_norm":args.use_layer_norm,
    "num_message_passing_steps":args.num_message_passing_steps,
    "global_reducer":args.global_reducer,
    "node_reducer":args.node_reducer,
    "dropedge_rate":args.dropedge_rate,
    "dropnode_rate":args.dropnode_rate,
    "dropout":args.dropout,
    "layernorm_before":args.layernorm_before,
    "use_bn":args.use_bn,
    "cycle":args.cycle,
    "node_attn":args.node_attn,
    "global_attn":args.global_attn
}

replay_buffer_params={
    "buffer_size":args.buffer_size,
    "num_atoms":args.num_atoms,
}

ddpg_agent_params={
    "actor_gnn_params":actor_gnn_params,
    "critic_gnn_params":critic_gnn_params,
    "replay_buffer_params":replay_buffer_params,
    "num_atoms":args.num_atoms,
    "pos_scale":args.pos_scale,
    "if_relax":args.if_relax,
    "max_relax_steps":args.max_relax_steps,
    "latent_size":args.latent_size,
    "mlp_hidden_size":args.mlp_hidden_size,
    "mlp_layers":args.mlp_layers,
    "threshold":args.threshold,
    "gamma":args.gamma,
    "tau":args.tau,
    "lr":args.lr,
    "decay_steps":args.decay_steps,
    "decay_rate":args.decay_rate,
    "if_soft_update":args.if_soft_update,
    "if_enhanced_sample":args.if_enhanced_sample,
    "if_greedy":args.if_greedy,
    "device":device,
}

if __name__=="__main__":
    agent=DDPG(**ddpg_agent_params)
    agent.train(args.num_steps,args.warmup_steps,args.batch_size)
    agent.actor.save_model("./model_save/actor_final.pt")
    agent.critic.save_model("./model_save/critic_final.pt")
    # agent.actor.load_model("./model_save/actor_final.pt")
    # agent.critic.load_model("./model_save/critic_final.pt")
    # agent.test(args.warmup_steps,200)
