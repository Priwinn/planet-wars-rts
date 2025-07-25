import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GATv2Conv, ResGatedGraphConv
from torch_geometric.data import Data, Batch
from typing import Tuple, Union, List
from gym_utils.gym_wrapper import owner_one_hot_encoding

import numpy as np
from gymnasium.spaces import GraphInstance

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
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
        self.edge_index = torch.Tensor([[i, j] for i in range(args.num_planets) for j in range(args.num_planets) if i != j]).long().permute(1, 0)

        
        # Node feature dimension from gym wrapper
        self.node_feature_dim = args.node_feature_dim + 2 # +2 for planet owner one-hot encodings
        
        # GNN layers (edge weights)
        # self.conv1 = GCNConv(self.node_feature_dim, 64)
        # self.conv2 = GCNConv(64, 128)
        # self.conv3 = GCNConv(128, 64)
        
        # Graph Attention Network layers (edge features)
        # self.v_conv1 = GATv2Conv(self.node_feature_dim, 64, heads=4, concat=True, edge_dim=5)
        # self.v_conv2 = GATv2Conv(64*4, 64, heads=1, concat=False, edge_dim=5)
        # self.v_conv2 = GATv2Conv(64*4, 128, heads=4, concat=True, edge_dim=5)
        # self.v_conv3 = GATv2Conv(128*4, 64, heads=1, concat=False, edge_dim=5)

        # self.a_conv1 = GATv2Conv(self.node_feature_dim, 64, heads=4, concat=True, edge_dim=5)
        # self.a_conv2 = GATv2Conv(64*4, 64, heads=1, concat=False, edge_dim=5)
        # self.a_conv2 = GATv2Conv(64*4, 128, heads=4, concat=True, edge_dim=5)
        # self.a_conv3 = GATv2Conv(128*4, 64, heads=1, concat=False, edge_dim=5)

        #Residual Gated Graph Conv layers
        self.v_conv1 = ResGatedGraphConv(self.node_feature_dim, 128, edge_dim=5)
        self.v_conv2 = ResGatedGraphConv(128, 128, edge_dim=5)
        self.a_conv1 = ResGatedGraphConv(self.node_feature_dim, 128, edge_dim=5)
        self.a_conv2 = ResGatedGraphConv(128, 128, edge_dim=5)

        # Global graph aggregation
        self.global_pool = global_mean_pool
        
        # Node and global feature MLPs
        self.node_mlp = nn.Sequential(
            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        
        self.v_global_mlp = nn.Sequential(
            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )

        self.a_global_mlp = nn.Sequential(
            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        
        # Value head - uses global features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=1.0),
        )
        
        # Policy heads - use per-node features for source/target selection
        self.source_actor = nn.Sequential(
            layer_init(nn.Linear(128, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=0.01),  # Per-node logit
        )
        
        self.target_actor = nn.Sequential(
            layer_init(nn.Linear(2*128, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=0.01),  # Per-node logit
        )

        # No-op Policy Head - uses global features
        self.noop_actor = nn.Sequential(
            layer_init(nn.Linear(128, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=0.01)
        )
        
        # Ship ratio (continuous) - uses global features
        if args.discretized_ratio_bins == 0:
            self.ratio_actor_mean = nn.Sequential(
                layer_init(nn.Linear(3*128, 32)),
                nn.ReLU(),
                layer_init(nn.Linear(32, 1), std=0.01),
            )
            self.ratio_actor_logstd = nn.Parameter(torch.zeros(1))
        else:
            #Discretized ratio actor
            self.ratio_actor = nn.Sequential(
                layer_init(nn.Linear(3*128, 32)),
                nn.ReLU(),
                layer_init(nn.Linear(32, self.args.discretized_ratio_bins), std=0.01),
            )

    def forward_gnn(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through GNN layers"""
        # GNN forward pass
        h = self.a_conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.a_conv2(h, edge_index, edge_attr)
        # h = F.relu(h)
        # h = self.a_conv3(h, edge_index, edge_attr)

        

        # Per-node features
        node_features = self.node_mlp(h)
        
        # Global features
        if batch is None:
            # Single graph case
            global_features = torch.mean(h, dim=0, keepdim=True)
        else:
            # Batch case
            global_features = self.global_pool(h, batch)
        
        global_features = self.a_global_mlp(global_features)
        
        return node_features, global_features
    
    def forward_value_gnn(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through GNN layers for value estimation"""
        # GNN forward pass
        h = self.v_conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.v_conv2(h, edge_index, edge_attr)
        # h = F.relu(h)
        # h = self.v_conv3(h, edge_index, edge_attr)

        # Global features
        if batch is None:
            # Single graph case
            global_features = torch.mean(h, dim=0, keepdim=True)
        else:
            # Batch case
            global_features = self.global_pool(h, batch)

        global_features = self.v_global_mlp(global_features)

        return h, global_features


    def get_value(self, obs):
        """Get state value"""
        if isinstance(obs, Union[Tuple, List]):
            obs = Batch.from_data_list(obs)
        data, batch = obs, obs.batch
        planet_owners = data.x[:, 0]
        transporter_owners_per_edge = data.edge_attr[:, 0]
        x = torch.cat((owner_one_hot_encoding(planet_owners, self.player_id),
                       data.x[:, 1:]), dim=-1)
        edge_attr=torch.cat((owner_one_hot_encoding(transporter_owners_per_edge, self.player_id),
                            data.edge_attr[:, 1:]), dim=-1)

        _, global_features = self.forward_value_gnn(x, data.edge_index, edge_attr, batch)
        return self.critic(global_features)
    

    def get_action_and_value(self, obs, action=None):
        """Get action probabilities and value"""
        if isinstance(obs, Union[Tuple, List]):
            obs = Batch.from_data_list(obs)
        else:
            obs = Batch.from_data_list([obs])  # Ensure obs is a Batch object
        data, batch = obs, obs.batch
        batch_size = batch.max().item() + 1
        num_planets = self.args.num_planets
    
        # Get planet owners
        planet_owners = data.x[:, 0].view(batch_size, num_planets)
        transporter_owners_per_edge = data.edge_attr.view(batch_size, num_planets,num_planets-1,3) [:,:,:, 0]
        transporter_owners = torch.sum(transporter_owners_per_edge, dim=2) > 0

        #one-hot encode planet owners and transporter owners

        data.x = torch.cat((owner_one_hot_encoding(planet_owners.view(-1), self.player_id),
                           data.x[:, 1:]),
                          dim=-1)
        data.edge_attr = torch.cat((owner_one_hot_encoding(transporter_owners_per_edge.view(-1), self.player_id),
                                   data.edge_attr[:, 1:]), dim=-1)

        node_features, global_features = self.forward_gnn(data.x, data.edge_index, data.edge_attr, batch)
        _, v_global_features = self.forward_value_gnn(data.x, data.edge_index, data.edge_attr, batch)

        # Get per-node logits for source selection
        source_node_logits = self.source_actor(node_features).squeeze(-1)  # [num_nodes]
        source_logits = source_node_logits.view(batch_size, -1)
        
        # Get no-op action logits
        noop_logits = self.noop_actor(global_features)  # [batch_size]
        # Concatenate no-op logits with source and logits
        source_logits = torch.cat((noop_logits, source_logits), dim=1)  # [batch_size, num_planets + 1]

        # Create masks
        source_mask = torch.logical_and(planet_owners == 1, transporter_owners == 0) 
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
            #Concatenate sampled action to target input
            source_features = node_features.view(batch_size,self.args.num_planets, -1)[valid_action_idx,source_action[valid_action_idx]-1].unsqueeze(1).expand(-1, self.args.num_planets, -1)  # [batches with valid actions, num_planets, node_feature_dim]
            target_features = node_features.view(batch_size,self.args.num_planets, -1)[valid_action_idx]
            target_features = torch.cat((source_features, target_features), dim=-1)

            target_logits = self.target_actor(target_features).squeeze(-1)  # [num_nodes]

            # Create target mask (opponent planets + neutral, but not source planet)
            # target_mask = torch.ones_like(planet_owners[valid_action_idx], dtype=torch.bool)  # All planets
            # target_mask = (planet_owners[valid_action_idx] != self.player_id).float()  # Not our planets
            # target_mask = (planet_owners[valid_action_idx] != 0).float()  # Not neutrals
            target_mask = (planet_owners[valid_action_idx] == 3-self.player_id).float()  #Currently targetting only opponent planets yields better results

            # Prevent sending to self
            target_mask[torch.arange(valid_action_idx.sum()), source_action[valid_action_idx]-1] = 0
            
            target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)

            if action is None:
                valid_target_action = target_probs.sample()
                target_action[valid_action_idx] = valid_target_action
            else:
                valid_target_action = target_action[valid_action_idx]

            # Get ship ratio distribution, we use sampled source and target node features and global features
            ratio_input = torch.cat((global_features[valid_action_idx],
                                    node_features.view(batch_size,self.args.num_planets, -1)[valid_action_idx,source_action[valid_action_idx]-1],
                                    node_features.view(batch_size,self.args.num_planets, -1)[valid_action_idx,valid_target_action]), dim=-1)
            if self.args.discretized_ratio_bins == 0:
                ratio_mean = self.ratio_actor_mean(ratio_input)
                ratio_std = self.ratio_actor_logstd.exp()
                ratio_probs = Normal(ratio_mean, ratio_std)
            else:
                ratio_probs = Categorical(logits=self.ratio_actor(ratio_input))
                
            if self.args.discretized_ratio_bins == 0:
                if action is None:
                    raw_ratio = ratio_probs.sample()
                    ratio_action[valid_action_idx] = torch.sigmoid(raw_ratio) / 2 + 0.5
                    # Log prob calculation
                    ratio_logprob_raw = ratio_probs.log_prob(raw_ratio).squeeze(-1)
                    # Jacobian correction
                    jacobian = torch.sigmoid(raw_ratio) * (1 - torch.sigmoid(raw_ratio)) / 2
                    ratio_logprob[valid_action_idx] = ratio_logprob_raw #- torch.log(jacobian).squeeze(-1) 
                    # According to https://openreview.net/forum?id=nIAxjsniDzg it does not affect policy loss or KL but it does affect entropy
                else:
                    # Inverse transform provided action back to raw space
                    transformed_ratio = (ratio_action[valid_action_idx] - 0.5) * 2  # [0,1] range
                    raw_ratio_action = torch.logit(torch.clamp(transformed_ratio, 1e-6, 1-1e-6))  # Avoid inf
                    # Calculate log prob in raw space
                    ratio_logprob_raw = ratio_probs.log_prob(raw_ratio_action).squeeze(-1)
                    # Jacobian correction
                    jacobian = torch.sigmoid(raw_ratio_action) * (1 - torch.sigmoid(raw_ratio_action)) / 2
                    ratio_logprob[valid_action_idx] = ratio_logprob_raw #- torch.log(jacobian).squeeze(-1) 
                    # According to https://openreview.net/forum?id=nIAxjsniDzg it does not affect policy loss or KL but it does affect entropy
            
            if self.args.discretized_ratio_bins > 0:
                if action is None:
                    ratio_action[valid_action_idx] = ratio_probs.sample().unsqueeze(-1)
                    ratio_logprob[valid_action_idx] = ratio_probs.log_prob(ratio_action[valid_action_idx].squeeze(-1))
                    ratio_action[valid_action_idx] = ratio_action[valid_action_idx] / (self.args.discretized_ratio_bins-1)  # Scale to [0,1]
                else:
                    discrete_ratio_action = ratio_action[valid_action_idx] * (self.args.discretized_ratio_bins-1)
                    discrete_ratio_action = discrete_ratio_action.long()
                    ratio_logprob[valid_action_idx] = ratio_probs.log_prob(discrete_ratio_action).squeeze(-1)

            target_entropy[valid_action_idx] = target_probs.entropy() 
            target_logprob[valid_action_idx] = target_probs.log_prob(valid_target_action)
            if self.args.discretized_ratio_bins == 0:
                # Add log(jacobian) to entropy to match https://openreview.net/forum?id=nIAxjsniDzg
                ratio_entropy[valid_action_idx] = ratio_probs.entropy() + torch.log(jacobian).squeeze(-1)
            else:
                ratio_entropy[valid_action_idx] = ratio_probs.entropy()


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
        
        # Get value
        value = self.critic(v_global_features)


        
        return action, total_logprob, total_entropy, value

    def get_action(self, data):
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
            
            # Take highest probability source action (deterministic for inference)
            source_action = source_probs.probs.argmax(dim=-1)  # [1]
            
            # Initialize target and ratio actions
            target_action = torch.zeros(1, dtype=torch.long, device=source_action.device)
            ratio_action = torch.zeros(1, 1, dtype=torch.float, device=source_action.device)
            
            # Only sample target and ratio actions if source action is non-null (not no-op)
            valid_action_idx = source_action != 0
            if valid_action_idx.any():
                # Get target logits for valid actions
                target_logits = self.target_actor(node_features).squeeze(-1).unsqueeze(0)  # [1, num_planets]
                
                # Create target mask (opponent planets only, same as get_action_and_value)
                target_mask = (planet_owners == 3-self.player_id).float()  # Only opponent planets
                
                # Prevent sending to self (source_action - 1 because source_action includes no-op offset)
                target_mask[0, source_action[0] - 1] = 0
                
                target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)
                target_action = target_probs.probs.argmax(dim=-1)  # [1]
                
                # Get ship ratio distribution using source, target, and global features
                ratio_input = torch.cat((global_features,
                                    node_features[source_action[0] - 1].unsqueeze(0),  # -1 for no-op offset
                                    node_features[target_action[0]].unsqueeze(0)), dim=-1)
                ratio_mean = torch.sigmoid(self.ratio_actor_mean(ratio_input))/2+0.5  # Min 0.5, max 1.0
                
                # Take mean for test time (deterministic)
                ratio_action = torch.clamp(ratio_mean, 0.0, 1.0)
            
            action = torch.cat([
                source_action.float(),
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

class MaskedCategorical(Categorical):
    """Categorical distribution with action masking"""
    
    def __init__(self, logits=None, probs=None, mask=None):
        if mask is not None:
            # Set logits of invalid actions to very negative values
            if logits is not None:
                logits = torch.where(mask.bool(), logits, torch.tensor(-1e8, device=logits.device))
            elif probs is not None:
                probs = torch.where(mask.bool(), probs, torch.tensor(1e-8, device=probs.device))
        
        super().__init__(logits=logits, probs=probs)
        self.mask = mask
    
    def entropy(self):
        # Only calculate entropy for valid actions
        if self.mask is not None:
            # Get probabilities and zero out invalid actions
            p_log_p = self.logits * self.probs
            p_log_p = torch.where(self.mask.bool(), p_log_p, torch.tensor(0.0, device=p_log_p.device))
            return -p_log_p.sum(-1)
        return super().entropy()


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