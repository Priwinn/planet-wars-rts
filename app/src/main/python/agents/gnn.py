import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GATv2Conv
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
        self.v_conv1 = GATv2Conv(self.node_feature_dim, 64, heads=4, concat=True, edge_dim=5)
        self.v_conv2 = GATv2Conv(64*4, 64, heads=1, concat=False, edge_dim=5)
        # self.v_conv2 = GATv2Conv(64*4, 128, heads=4, concat=True, edge_dim=5)
        # self.v_conv3 = GATv2Conv(128*4, 64, heads=1, concat=False, edge_dim=5)

        self.a_conv1 = GATv2Conv(self.node_feature_dim, 64, heads=4, concat=True, edge_dim=5)
        self.a_conv2 = GATv2Conv(64*4, 64, heads=1, concat=False, edge_dim=5)
        # self.a_conv2 = GATv2Conv(64*4, 128, heads=4, concat=True, edge_dim=5)
        # self.a_conv3 = GATv2Conv(128*4, 64, heads=1, concat=False, edge_dim=5)

        # Global graph feature extraction
        self.global_pool = global_mean_pool
        
        # Combine node embeddings and global features
        self.node_mlp = nn.Sequential(
            layer_init(nn.Linear(64, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        
        self.global_mlp = nn.Sequential(
            layer_init(nn.Linear(64, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        
        # Value head - uses global features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=1.0),
        )
        
        # Policy heads - use per-node features for source/target selection
        self.source_actor = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=0.01),  # Per-node logit
        )
        
        self.target_actor = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=0.01),  # Per-node logit
        )
        
        # Ship ratio (continuous) - uses global features
        self.ratio_actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=0.01),
        )
        self.ratio_actor_logstd = nn.Parameter(torch.zeros(1))

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
        
        global_features = self.global_mlp(global_features)
        
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
        
        global_features = self.global_mlp(global_features)
        
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
        data, batch = obs, obs.batch

        if batch is None:
            # Single graph case
            batch_size = 1
            num_planets = data.x.size(0)

            # Get masks from node features (owner is first feature)
            planet_owners = data.x[:, 0].unsqueeze(0)  # [1, num_planets]
            transporter_owners_per_edge = data.edge_attr[:, 0].view(num_planets,num_planets-1).unsqueeze(0) # [1, num_planets, num_planets-1]
            transporter_owners = torch.sum(transporter_owners_per_edge, dim=2) > 0  # [1, num_planets]
        else:
            # Batch case
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

        # Get per-node logits for source and target selection
        source_node_logits = self.source_actor(node_features).squeeze(-1)  # [num_nodes]
        target_node_logits = self.target_actor(node_features).squeeze(-1)  # [num_nodes]

        if batch is None:
            # Single graph case
            source_logits = source_node_logits.unsqueeze(0)
            target_logits = target_node_logits.unsqueeze(0)
        else:
            # Batch case
            source_logits = source_node_logits.view(batch_size, -1)
            target_logits = target_node_logits.view(batch_size, -1)
        
        # Get ship ratio distribution
        ratio_mean = torch.sigmoid(self.ratio_actor_mean(global_features))
        # ratio_mean = self.ratio_actor_mean(global_features)
        ratio_std = torch.clamp(self.ratio_actor_logstd.exp(), min=0.01, max=0.5)
        # ratio_std = torch.exp(self.ratio_actor_logstd)  
        ratio_probs = Normal(ratio_mean, ratio_std)
        


        # Create masks
        source_mask = torch.logical_and(planet_owners == 1, transporter_owners == 0) 
        
        # Create masked distributions for source selection
        source_probs = MaskedCategorical(logits=source_logits, mask=source_mask)
        
        if action is None:
            # Sample actions
            source_action = source_probs.sample()
            
            # Create target mask (opponent planets + neutral, but not source planet)
            # target_mask = torch.ones_like(planet_owners, dtype=torch.bool)  # All planets
            # target_mask = (planet_owners != self.player_id).float()  # Not our planets
            # target_mask = (planet_owners != 0).float()  # Not neutrals
            target_mask = (planet_owners == 2).float()  #Currently targetting only opponent planets yields better results
            
            # Prevent sending to self
            if batch is None:
                target_mask[0, source_action[0]] = 0
            else:
                target_mask[torch.arange(batch_size), source_action] = 0
            
            target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)
            target_action = target_probs.sample()
            
            # Sample ship ratio
            ratio_action = torch.clamp(ratio_probs.sample(), 0.0, 0.99)
            
            if batch is None:
                action = torch.stack([
                    source_action.float(),
                    target_action.float(),
                    ratio_action.squeeze(-1)
                ], dim=-1)
            else:
                action = torch.stack([
                    source_action.float(),
                    target_action.float(),
                    ratio_action.squeeze(-1)
                ], dim=-1)
        else:
            # Use provided actions
            source_action = action[:, 0].long()
            target_action = action[:, 1].long()
            ratio_action = action[:, 2]
            
            # Create target mask for log probability calculation
            target_mask = (planet_owners == 3-self.player_id).float()

            # Avoid sending to self
            if batch is None:
                target_mask[0, source_action[0]] = 0
            else:
                target_mask[torch.arange(batch_size), source_action] = 0
            
            target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)
        
        # Calculate log probabilities
        source_logprob = source_probs.log_prob(source_action)
        target_logprob = target_probs.log_prob(target_action)
        ratio_logprob = ratio_probs.log_prob(ratio_action).squeeze(-1)
        
        # Combined log probability
        total_logprob = source_logprob + target_logprob + ratio_logprob
        
        # Combined entropy
        total_entropy = source_probs.entropy() + target_probs.entropy() + ratio_probs.entropy().squeeze(-1)
        
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
            # _, v_global_features = self.forward_value_gnn(data.x, data.edge_index, data.edge_attr)


            # Get per-node logits for source and target selection
            source_node_logits = self.source_actor(node_features).squeeze(-1)  # [num_nodes]
            target_node_logits = self.target_actor(node_features).squeeze(-1)  # [num_nodes]
            
            # Get ship ratio distribution
            ratio_mean = torch.sigmoid(self.ratio_actor_mean(global_features))
            # ratio_mean = self.ratio_actor_mean(global_features)

            source_logits = source_node_logits.unsqueeze(0)  # [1, num_planets]
            target_logits = target_node_logits.unsqueeze(0)  # [1, num_planets]
            
            # Create masks
            source_mask = torch.logical_and(planet_owners == self.player_id, transporter_owners == 0)
            
            # Create masked distributions for source selection
            source_probs = MaskedCategorical(logits=source_logits, mask=source_mask)
            
            # Take highest probability source action
            source_action = source_probs.probs.argmax(dim=-1)  # [1]
            
            # Create target mask
            target_mask = (planet_owners == 3-self.player_id).float()  #Currently targetting only opponent planets yields better results
            
            # Prevent sending to self
            target_mask[0, source_action[0]] = 0
            
            target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)
            target_action = target_probs.probs.argmax(dim=-1)  # [1]
            
            # Take mean for test time
            ratio_action = torch.clamp(ratio_mean, 0.0, 0.99)
            
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