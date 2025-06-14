import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PlanetWarsAgentGNN(nn.Module):
    """Graph Neural Network agent with action masking"""
    
    def __init__(self, args, player_id=1):
        super().__init__()
        self.args = args
        self.player_id = player_id
        
        # Node feature dimension from gym wrapper (13 features per planet)
        self.node_feature_dim = 13
        
        # GNN layers
        self.conv1 = GCNConv(self.node_feature_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        
        # Alternative: Graph Attention Network layers
        # self.conv1 = GATConv(self.node_feature_dim, 64, heads=4, concat=True)
        # self.conv2 = GATConv(64*4, 128, heads=4, concat=True)
        # self.conv3 = GATConv(128*4, 64, heads=1, concat=False)
        
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

    def forward_gnn(self, x, edge_index, batch=None):
        """Forward pass through GNN layers"""
        # GNN forward pass
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = self.conv3(h, edge_index)
        
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

    def process_observation(self, obs):
        """Convert gym observation to PyTorch Geometric format"""
        if isinstance(obs, dict):
            # Single observation
            node_features = torch.FloatTensor(obs['node_features'])
            adj_matrix = torch.FloatTensor(obs['adjacency_matrix'])
            
            # Convert adjacency matrix to edge index
            edge_index = torch.nonzero(adj_matrix).t().contiguous()
            
            return Data(x=node_features, edge_index=edge_index), None
        else:
            # Batch of observations
            data_list = []
            for i, single_obs in enumerate(obs):
                node_features = torch.FloatTensor(single_obs['node_features'])
                adj_matrix = torch.FloatTensor(single_obs['adjacency_matrix'])
                edge_index = torch.nonzero(adj_matrix).t().contiguous()
                data_list.append(Data(x=node_features, edge_index=edge_index))
            
            batch = Batch.from_data_list(data_list)
            return batch, batch.batch

    def get_value(self, obs):
        """Get state value"""
        data, batch = self.process_observation(obs)
        _, global_features = self.forward_gnn(data.x, data.edge_index, batch)
        return self.critic(global_features)

    def get_action_and_value(self, obs, action=None):
        """Get action probabilities and value"""
        data, batch = self.process_observation(obs)
        node_features, global_features = self.forward_gnn(data.x, data.edge_index, batch)
        
        # Get per-node logits for source and target selection
        source_node_logits = self.source_actor(node_features).squeeze(-1)  # [num_nodes]
        target_node_logits = self.target_actor(node_features).squeeze(-1)  # [num_nodes]
        
        # Get ship ratio distribution
        ratio_mean = torch.sigmoid(self.ratio_actor_mean(global_features))
        ratio_std = torch.clamp(self.ratio_actor_logstd.exp(), min=0.01, max=0.5)
        ratio_probs = Normal(ratio_mean, ratio_std)
        
        if batch is None:
            # Single graph case
            num_envs = 1
            batch_size = 1
            num_planets = data.x.size(0)
            
            # Reshape logits for masking
            source_logits = source_node_logits.unsqueeze(0)  # [1, num_planets]
            target_logits = target_node_logits.unsqueeze(0)  # [1, num_planets]
            
            # Get masks from node features (owner is first feature)
            planet_owners = data.x[:, 0].unsqueeze(0)  # [1, num_planets]
        else:
            # Batch case
            batch_size = batch.max().item() + 1
            num_planets = self.args.num_planets
            
            # Reshape node logits to [batch_size, num_planets]
            source_logits = source_node_logits.view(batch_size, num_planets)
            target_logits = target_node_logits.view(batch_size, num_planets)
            
            # Get planet owners
            planet_owners = data.x[:, 0].view(batch_size, num_planets)
        
        # Create masks
        source_mask = (planet_owners == self.player_id).float()  # Own planets
        
        # Create masked distributions for source selection
        source_probs = MaskedCategorical(logits=source_logits, mask=source_mask)
        
        if action is None:
            # Sample actions
            source_action = source_probs.sample()
            
            # Create target mask (opponent planets + neutral, but not source planet)
            target_mask = (planet_owners != self.player_id).float()  # Not our planets
            
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
            target_mask = (planet_owners != self.player_id).float()
            
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
        value = self.critic(global_features)
        
        return action, total_logprob, total_entropy, value


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
    print(f"Observation shape: {obs.node_features.shape}")
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