import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GATv2Conv, ResGatedGraphConv
from torch_geometric.nn import Sequential as PyGSequential
from torch_geometric.nn.norm import MeanSubtractionNorm
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, select
from typing import Tuple, Union, List
from gym_utils.gym_wrapper import owner_one_hot_encoding
from gym_utils.distributions import MaskedCategorical, SigmoidTransformedDistribution

import numpy as np
from gymnasium.spaces import GraphInstance

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init_gat(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.lin_l.weight, std)
    torch.nn.init.constant_(layer.lin_l.bias, bias_const)
    torch.nn.init.orthogonal_(layer.lin_r.weight, std)
    torch.nn.init.constant_(layer.lin_r.bias, bias_const)
    if layer.lin_edge is not None:
        torch.nn.init.orthogonal_(layer.lin_edge.weight, std)
        # torch.nn.init.constant_(layer.lin_edge.bias, bias_const) # PyG GATv2Conv does not have bias in edge linear layer
    torch.nn.init.constant_(layer.bias, bias_const)
    torch.nn.init.orthogonal_(layer.att, std)
    return layer

def GraphInstanceToPyG(graph_instance):
    """Convert a GraphInstance to PyTorch Geometric Data format"""
    if isinstance(graph_instance, GraphInstance):
        node_features = graph_instance.nodes
        edge_index = graph_instance.edge_links.permute(1, 0).long()  # Convert to edge index format
        return Data(x=node_features, edge_index=edge_index)
    else:
        raise ValueError("Input must be a GraphInstance")


class PlanetWarsAgentGNN(nn.Module):
    """Graph Neural Network agent with action masking"""
    
    def __init__(self, args, player_id=1):
        super().__init__()
        self.args = args
        self.player_id = player_id
        # self.edge_index = torch.Tensor([[i, j] for i in range(args.num_planets) for j in range(args.num_planets) if i != j]).long().permute(1, 0)

        # Node feature dimension from gym wrapper
        self.node_feature_dim = args.node_feature_dim + 1 # +2 for planet owner one-hot encodings -1 for hasTransporter (we only use it to avoid calculating the transporter from edges)
        self.hidden_dim = args.hidden_dim if hasattr(args, 'hidden_dim') else 128  # Default hidden dimension
        
        # Graph Attention Network layers (edge features)
        self.a_gnn = PyGSequential('x, edge_index, edge_attr, batch', [
            (layer_init_gat(GATv2Conv(self.node_feature_dim, self.hidden_dim, heads=4, concat=True, edge_dim=5)), 'x, edge_index, edge_attr -> x'),
            (MeanSubtractionNorm(), 'x, batch -> x'),
            nn.ReLU(),
            # (layer_init_gat(GATv2Conv(self.hidden_dim*4, self.hidden_dim, heads=4, concat=True, edge_dim=5)), 'x, edge_index, edge_attr -> x'),
            # (MeanSubtractionNorm(), 'x, batch -> x'),
            # nn.ReLU(),
            (layer_init_gat(GATv2Conv(self.hidden_dim*4, self.hidden_dim, heads=1, concat=False, edge_dim=5)), 'x, edge_index, edge_attr -> x'),
        ])

        self.v_gnn = PyGSequential('x, edge_index, edge_attr, batch', [
            (layer_init_gat(GATv2Conv(self.node_feature_dim, self.hidden_dim, heads=4, concat=True, edge_dim=5)), 'x, edge_index, edge_attr -> x'),
            (MeanSubtractionNorm(), 'x, batch -> x'),
            nn.ReLU(),
            # (layer_init_gat(GATv2Conv(self.hidden_dim*4, self.hidden_dim, heads=4, concat=True, edge_dim=5)), 'x, edge_index, edge_attr -> x'),
            # (MeanSubtractionNorm(), 'x, batch -> x'),
            # nn.ReLU(),
            (layer_init_gat(GATv2Conv(self.hidden_dim*4, self.hidden_dim, heads=1, concat=False, edge_dim=5)), 'x, edge_index, edge_attr  -> x'),
        ])

        #Residual Gated Graph Conv layers
        # self.v_gnn = PyGSequential('x, edge_index, edge_attr, batch', [
        #     (ResGatedGraphConv(self.node_feature_dim, self.hidden_dim, edge_dim=5), 'x, edge_index, edge_attr -> x'),
        #     nn.ReLU(),
        #     (ResGatedGraphConv(self.hidden_dim, self.hidden_dim, edge_dim=5), 'x, edge_index, edge_attr -> x'),
        #     nn.ReLU(),
        #     (ResGatedGraphConv(self.hidden_dim, self.hidden_dim, edge_dim=5), 'x, edge_index, edge_attr -> x'),
        # ])
        # self.a_gnn = PyGSequential('x, edge_index, edge_attr, batch', [
        #     (ResGatedGraphConv(self.node_feature_dim, self.hidden_dim, edge_dim=5), 'x, edge_index, edge_attr -> x'),
        #     (MeanSubtractionNorm(), 'x, batch -> x'),
        #     nn.ReLU(),
        #     (ResGatedGraphConv(self.hidden_dim, self.hidden_dim, edge_dim=5), 'x, edge_index, edge_attr -> x'),
        #     (MeanSubtractionNorm(), 'x, batch -> x'),
        #     nn.ReLU(),
        #     (ResGatedGraphConv(self.hidden_dim, self.hidden_dim, edge_dim=5), 'x, edge_index, edge_attr -> x'),
        # ])

        # Global graph aggregation
        self.global_pool = global_mean_pool
        
        # Value head - uses global features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
        )
        
        # Policy heads - use per-node features for source/target selection
        self.source_actor = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=0.01),  # Per-node logit
        )
        
        self.target_actor = nn.Sequential(
            layer_init(nn.Linear(2*self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=0.01),  # Per-node logit
        )

        # No-op Policy Head - uses global features
        self.noop_actor = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=0.01)
        )
        
        # Ship ratio (continuous) - uses global features
        if args.discretized_ratio_bins == 0:
            self.ratio_actor_mean = nn.Sequential(
                layer_init(nn.Linear(3*self.hidden_dim, 32)),
                nn.ReLU(),
                layer_init(nn.Linear(32, 1), std=0.01),
            )
            self.ratio_actor_logstd = nn.Parameter(torch.zeros(1))
        else:
            #Discretized ratio actor
            self.ratio_actor = nn.Sequential(
                layer_init(nn.Linear(3*self.hidden_dim, 32)),
                nn.ReLU(),
                layer_init(nn.Linear(32, self.args.discretized_ratio_bins), std=0.01),
            )

    def forward_gnn(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through GNN layers"""
        # GNN forward pass
        h = self.a_gnn(x, edge_index, edge_attr, batch)

        # Per-node features
        # node_features = self.node_mlp(h)
        
        # Global features
        if batch is None:
            # Single graph case
            global_features = torch.mean(h, dim=0, keepdim=True)
        else:
            # Batch case
            global_features = self.global_pool(h, batch)
        
        # global_features = self.a_global_mlp(global_features)
        
        return h, global_features
    
    def forward_value_gnn(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through GNN layers for value estimation"""
        # GNN forward pass
        h = self.v_gnn(x, edge_index, edge_attr, batch)

        # Global features
        if batch is None:
            # Single graph case
            global_features = torch.mean(h, dim=0, keepdim=True)
        else:
            # Batch case
            global_features = self.global_pool(h, batch)

        value = self.critic(global_features)

        return value


    def get_value(self, obs):
        """Get state value"""
        if isinstance(obs, Union[Tuple, List]):
            obs = Batch.from_data_list(obs)
        data, batch = obs, obs.batch
        planet_owners = data.x[:, 0]
        transporter_owners_per_edge = data.edge_attr[:, 0]
        x = torch.cat((owner_one_hot_encoding(planet_owners, self.player_id),
                       data.x[:, 1:-1]), dim=-1)
        edge_attr=torch.cat((owner_one_hot_encoding(transporter_owners_per_edge, self.player_id),
                            data.edge_attr[:, 1:]), dim=-1)

        value = self.forward_value_gnn(x, data.edge_index, edge_attr, batch)
        return value


    def get_action_and_value(self, obs, action=None):
        """Get action probabilities and value"""
        if isinstance(obs, Union[Tuple, List]):
            obs = Batch.from_data_list(obs)
        elif isinstance(obs, Data):
            obs = Batch.from_data_list([obs])  # Ensure obs is a Batch object
        data, batch = obs, obs.batch
        batch_size = batch.max().item() + 1
        # Get number of nodes in each sample
        num_nodes = torch.scatter_reduce(torch.zeros(batch_size,dtype=int).to(batch.device),0,batch,torch.ones_like(batch),reduce='sum')
    
        # Get planet owners
        planet_owners = data.x[:, 0]
        transporter_owners_per_edge = data.edge_attr[:, 0]
        transporter_owners = data.x[:, 3]

        #one-hot encode planet owners and transporter owners
        data.x = torch.cat((owner_one_hot_encoding(planet_owners, self.player_id),
                           data.x[:, 1:-1]),
                          dim=-1)
        data.edge_attr = torch.cat((owner_one_hot_encoding(transporter_owners_per_edge, self.player_id),
                                   data.edge_attr[:, 1:]), dim=-1)
        
        # Forward pass through GNN
        node_features, global_features = self.forward_gnn(data.x, data.edge_index, data.edge_attr, batch)
        # Get value from GNN
        value = self.forward_value_gnn(data.x, data.edge_index, data.edge_attr, batch)

        # Get per-node logits for source selection
        source_node_logits = self.source_actor(node_features).squeeze(-1)  # [num_nodes]
        source_logits = to_dense_batch(source_node_logits, batch, fill_value=-1e8)[0]  # [batch_size, max_num_planets]
        
        # Get no-op action logits
        noop_logits = self.noop_actor(global_features)  # [batch_size]
        # Concatenate no-op logits with source and logits
        source_logits = torch.cat((noop_logits, source_logits), dim=1)  # [batch_size, num_planets + 1]

        # Create masks
        dense_planet_owners = to_dense_batch(planet_owners, batch, fill_value=-1)[0]
        dense_transporter_owners = to_dense_batch(transporter_owners, batch, fill_value=-1)[0]
        source_mask = torch.logical_and(dense_planet_owners == 1, dense_transporter_owners == 0) 
        source_mask = torch.cat((torch.ones(batch_size, 1, dtype=torch.bool, device=source_mask.device), source_mask), dim=1)  # Add no-op mask

        # Create masked distributions for source selection
        source_probs = MaskedCategorical(logits=source_logits, mask=source_mask)
        
        #Initialize logprobs and entropy
        target_logprob = torch.zeros(batch_size, device=source_logits.device)
        target_entropy = torch.zeros(batch_size, device=source_logits.device)
        ratio_logprob = torch.zeros(batch_size, device=source_logits.device)
        ratio_entropy = torch.zeros(batch_size, device=source_logits.device)

        if action is None:
            # Sample actions
            source_action = source_probs.sample()
            source_action = torch.clamp(source_action, max=num_nodes)  # Clamp to valid range just in case a padding node is selected. Note no -1 because we cat co-op
            target_action = torch.zeros(batch_size, dtype=torch.long, device=source_action.device)
            ratio_action = torch.zeros(batch_size, 1, dtype=torch.float, device=source_action.device)
        else:
            # Use provided action
            source_action = action[:, 0].long()  # Source action
            target_action = action[:, 1].long()  # Target action
            ratio_action = action[:, 2].unsqueeze(-1)  # Ratio action

        #Only sample target and ratio actions if source action is non-null
        valid_action_idx = source_action != 0
        if valid_action_idx.any():
            # Get node mask for valid actions. This returns a mask of nodes belonging to sample observations were the sampled source action is non-null
            valid_batch = valid_action_idx[batch]

            cumulative_nodes = torch.cumsum(num_nodes, dim=0)
            cumulative_nodes = torch.cat((torch.zeros(1, dtype=cumulative_nodes.dtype, device=cumulative_nodes.device), cumulative_nodes[:-1]), dim=0)
            # Get node features for selected valid actions
            source_idx = cumulative_nodes[valid_action_idx] + source_action[valid_action_idx] - 1

            #Concatenate sampled action to target input
            source_features = node_features[source_idx].repeat_interleave(num_nodes[valid_action_idx], dim=0)  # [batches with valid actions, num_planets, node_feature_dim]
            target_features = select(node_features, valid_batch, dim=0)
            target_features = torch.cat((source_features, target_features), dim=-1)

            target_logits = self.target_actor(target_features).squeeze(-1)  # [num_nodes]

            # Create target mask (opponent planets + neutral, but not source planet)
            target_mask = torch.ones_like(planet_owners[valid_batch], dtype=torch.bool)  # All planets
            # target_mask = (planet_owners[valid_batch] != self.player_id).float()  # Not our planets
            # target_mask = (planet_owners[valid_batch] != 0).float()  # Not neutrals
            # target_mask = (planet_owners[valid_batch] == 3-self.player_id).float()  #Currently targetting only opponent planets yields better results

            dense_valid_batch_idx = torch.arange(valid_action_idx.sum(), device=source_logits.device).repeat_interleave(num_nodes[valid_action_idx])
            dense_target_logits = to_dense_batch(target_logits, dense_valid_batch_idx, fill_value=-1e8)[0]  # [batch_size, max_num_planets]
            
            # Prevent sending to self
            target_mask = to_dense_batch(target_mask, dense_valid_batch_idx, fill_value=0.0)[0]  # [batch_size, max_num_planets]
            target_mask[torch.arange(valid_action_idx.sum()), source_action[valid_action_idx]-1] = 0

            target_probs = MaskedCategorical(logits=dense_target_logits, mask=target_mask)

            if action is None:
                valid_target_action = target_probs.sample()
                valid_target_action = torch.clamp(valid_target_action, max=num_nodes[valid_action_idx]-1)  # Clamp to valid range just in case a padding node is selected.
                target_action[valid_action_idx] = valid_target_action
            else:
                valid_target_action = target_action[valid_action_idx]

            # Get ship ratio distribution, we use sampled source and target node features and global features
            # print(f'Target index: {cumulative_nodes[valid_action_idx] + valid_target_action}')
            # print(f'Node features shape: {node_features.shape}')
            ratio_input = torch.cat((global_features[valid_action_idx], #Global features
                                    node_features[source_idx], # Source node features
                                    node_features[cumulative_nodes[valid_action_idx] + valid_target_action] # Target node features
                                    ), dim=-1) 
            if self.args.discretized_ratio_bins == 0:
                ratio_mean = self.ratio_actor_mean(ratio_input)
                ratio_std = self.ratio_actor_logstd.exp()
                ratio_std = torch.clamp(ratio_std, max=10.0)  # Clamp to avoid extreme values
                ratio_probs = SigmoidTransformedDistribution(ratio_mean, ratio_std)
            else:
                ratio_probs = Categorical(logits=self.ratio_actor(ratio_input))
                
            if self.args.discretized_ratio_bins == 0:
                if action is None:
                    ratio_sample = ratio_probs.sample()
                    ratio_action[valid_action_idx] = ratio_sample
                    ratio_logprob[valid_action_idx] = ratio_probs.log_prob(ratio_sample).squeeze(-1)
                else:
                    ratio_logprob[valid_action_idx] = ratio_probs.log_prob(ratio_action[valid_action_idx]).squeeze(-1)
            
            if self.args.discretized_ratio_bins > 0:
                if action is None:
                    ratio_action_bins = ratio_probs.sample().unsqueeze(-1)
                    ratio_logprob[valid_action_idx] = ratio_probs.log_prob(ratio_action_bins.squeeze(-1))
                    ratio_action[valid_action_idx] = ratio_action_bins / (self.args.discretized_ratio_bins-1)  # Scale to [0,1]
                else:
                    discrete_ratio_action = ratio_action[valid_action_idx] * (self.args.discretized_ratio_bins-1)
                    discrete_ratio_action = discrete_ratio_action.long()
                    ratio_logprob[valid_action_idx] = ratio_probs.log_prob(discrete_ratio_action.squeeze(-1))

            target_entropy[valid_action_idx] = target_probs.entropy() 
            target_logprob[valid_action_idx] = target_probs.log_prob(valid_target_action)
            ratio_entropy[valid_action_idx] = ratio_probs.entropy().squeeze(-1)


        action = torch.stack([
            source_action.float(),
            target_action.float(),
            ratio_action.squeeze(-1)
        ], dim=-1)
    
        # Calculate log probabilities
        source_logprob = source_probs.log_prob(action[:, 0]) 

        # Combined log probability
        total_logprob = source_logprob + target_logprob + ratio_logprob
        
        # Combined entropy
        total_entropy = source_probs.entropy() + target_entropy + ratio_entropy #According to cleanrl, entropy does not help in continuous actions
        
        return action, total_logprob, total_entropy, value

    def get_action(self, data, exploit=True):
        with torch.no_grad():
            # Get masks from node features (owner is first feature)
            num_planets = data.x.size(0)
            planet_owners = data.x[:, 0].unsqueeze(0)  # [1, num_planets]
            transporter_owners_per_edge = data.edge_attr[:, 0].view(num_planets,num_planets-1).unsqueeze(0) # [1, num_planets, num_planets-1]
            transporter_owners = torch.sum(transporter_owners_per_edge, dim=2) > 0  # [1, num_planets]

            #one-hot encode planet owners and transporter owners
            data.x = torch.cat((owner_one_hot_encoding(planet_owners.view(-1), self.player_id),
                            data.x[:, 1:]),
                            dim=-1)
            data.edge_attr = torch.cat((owner_one_hot_encoding(transporter_owners_per_edge.view(-1), self.player_id),
                                    data.edge_attr[:, 1:]), dim=-1)

            node_features, global_features = self.forward_gnn(data.x, data.edge_index, data.edge_attr)

            # Get per-node logits for source selection
            source_node_logits = self.source_actor(node_features).squeeze(-1)  # [num_nodes]
            source_logits = source_node_logits.unsqueeze(0)  # [1, num_planets]
            
            # Get no-op action logits
            noop_logits = self.noop_actor(global_features)  # [1]
            # Concatenate no-op logits with source logits
            source_logits = torch.cat((noop_logits, source_logits), dim=1)  # [1, num_planets + 1]

            # Create masks - same as get_action_and_value
            source_mask = torch.logical_and(planet_owners == self.player_id, transporter_owners == 0)
            source_mask = torch.cat((torch.ones(1, 1, dtype=torch.bool, device=source_mask.device), source_mask), dim=1)  # Add no-op mask

            # Create masked distributions for source selection
            source_probs = MaskedCategorical(logits=source_logits, mask=source_mask)

            # Exploit or explore
            if exploit:
                source_action = source_probs.probs.argmax(dim=-1)  # [1]
            else:
                source_action = source_probs.sample()  # [1]

            # Initialize target and ratio actions
            target_action = torch.zeros(1, dtype=torch.long, device=source_action.device)
            ratio_action = torch.zeros(1, 1, dtype=torch.float, device=source_action.device)
            
            # Only sample target and ratio actions if source action is non-null (not no-op)
            valid_action_idx = source_action != 0
            if valid_action_idx.any():
                # Get target logits for valid actions
                source_features = node_features[source_action[valid_action_idx]-1].expand(num_planets, -1)  # [batches with valid actions, num_planets, node_feature_dim]
                target_features = torch.cat((source_features, node_features), dim=-1)
                target_logits = self.target_actor(target_features).squeeze(-1).unsqueeze(0)  # [1, num_planets]

                # Create target mask (opponent planets only, same as get_action_and_value)
                target_mask = torch.ones_like(planet_owners, dtype=torch.bool)  # All planets
                # target_mask = (planet_owners != self.player_id).float()  # Not our planets
                # target_mask = (planet_owners != 0).float()  # Not neutrals
                # target_mask = (planet_owners == 3-self.player_id).float()  # Only opponent planets
                
                # Prevent sending to self (source_action - 1 because source_action includes no-op offset)
                target_mask[0, source_action[0] - 1] = 0
                
                target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)
                if exploit:
                    target_action = target_probs.probs.argmax(dim=-1)  # [1]
                else:
                    target_action = target_probs.sample()  # [1]

                # Get ship ratio distribution using source, target, and global features
                ratio_input = torch.cat((global_features,
                                    node_features[source_action[0] - 1].unsqueeze(0),  # -1 for no-op offset
                                    node_features[target_action[0]].unsqueeze(0)), dim=-1)
                if self.args.discretized_ratio_bins == 0:
                    ratio_mean = torch.sigmoid(self.ratio_actor_mean(ratio_input))
                    if exploit:
                        ratio_action = ratio_mean
                    else:
                        ratio_action = SigmoidTransformedDistribution(ratio_mean, self.ratio_actor_logstd.exp()).sample()
                else:
                    if exploit:
                        ratio_action = torch.argmax(self.ratio_actor(ratio_input), dim=-1)
                    else:
                        ratio_action = Categorical(logits=self.ratio_actor(ratio_input)).sample()
                    ratio_action = ratio_action.float() / (self.args.discretized_ratio_bins-1)

            action = torch.cat([
                source_action.float()-1,
                target_action.float(),
                ratio_action.squeeze(-1)
            ], dim=-1)

            return action
        
    def copy(self):
        """Create a copy of the agent"""
        new_agent = PlanetWarsAgentGNN(self.args, self.player_id)
        new_agent.load_state_dict(self.state_dict())
        return new_agent
    
    def copy_as_opponent(self):
        """Create a copy of the agent as an opponent"""
        new_agent = PlanetWarsAgentGNN(self.args, self.player_id)
        new_agent.load_state_dict(self.state_dict())
        new_agent.player_id = 3 - self.player_id
        return new_agent



# Example usage showing how to integrate with the gym environment
if __name__ == "__main__":
    import argparse
    from util.gym_wrapper import PlanetWarsForwardModelEnv
    from core.game_state import Player
    
    # Mock args for testing
    class Args:
        def __init__(self):
            self.num_planets = 10
            self.node_feature_dim = 13
            self.use_adjacency_matrix = True
            self.num_envs = 1
            self.minibatch_size = 1
    
    args = Args()
    
    # Create environment
    env = PlanetWarsForwardModelEnv(
        controlled_player=Player.Player1,
        max_ticks=100,
        game_params={'numPlanets': args.num_planets}
    )
    
    # Create GNN agent
    agent = PlanetWarsAgentGNN(args, player_id=1)
    
    # Test with environment
    obs, info = env.reset()
    print(f"Observation shape: {obs.nodes.shape}")
    print(f"Adjacency matrix shape: {obs.adjacency_matrix.shape}")
    
    # Convert to dict format expected by agent
    obs_dict = {
        'node_features': obs.node_features,
        'adjacency_matrix': obs.adjacency_matrix
    }
    
    # Test agent
    with torch.no_grad():
        action, logprob, entropy, value = agent.get_action_and_value(obs_dict)
        print(f"Action: {action}")
        print(f"Log prob: {logprob}")
        print(f"Entropy: {entropy}")
        print(f"Value: {value}")
    
    env.close()
    print("GNN agent test completed!")