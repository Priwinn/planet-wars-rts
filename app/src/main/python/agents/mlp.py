import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np
from util.gym_wrapper import owner_one_hot_encoding


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PlanetWarsAgentMLP(nn.Module):
    """Neural network agent with action masking"""
    
    def __init__(self, args, player_id=1):
        super().__init__()
        self.args = args
        self.player_id = player_id

        # Input dimensions based on whether adjacency matrix is used
        node_features_dim = args.num_planets * (args.node_feature_dim + 4) # 4 extra from one-hot encoding of transporter owners and planet owners
        total_input_dim = node_features_dim
        
        if args.use_adjacency_matrix:
            adj_matrix_dim = args.num_planets * args.num_planets
            total_input_dim += adj_matrix_dim
        
        # Adjust network size based on input dimension
        hidden_size = 512 if args.use_adjacency_matrix else 256
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Linear(total_input_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        
        # Value head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Policy heads
        self.source_actor = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, args.num_planets), std=0.01),
        )
        
        self.target_actor = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, args.num_planets), std=0.01),
        )
        
        # Ship ratio (continuous)
        self.ratio_actor_mean = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=0.01),
        )
        self.ratio_actor_logstd = nn.Parameter(torch.zeros(1))  # Log std for ratio action

    def get_value(self, x):
        planet_owners = x[:, :, 0]
        transporter_owners = x[:, :, 5]
        x = torch.cat((owner_one_hot_encoding(planet_owners, self.player_id), 
                       x[:, :, 1:5],
                       owner_one_hot_encoding(transporter_owners, self.player_id), 
                       x[:, :, 6:]
                       ), dim=-1)
        features = self.feature_extractor(x.flatten(start_dim=1))
        return self.critic(features)


    def get_action_and_value(self, x, action=None):
        planet_owners = x[:, :, 0]
        transporter_owners = x[:, :, 5]
        source_mask = planet_owners == 1
        target_mask = planet_owners == 2 #TODO check if sending transporters to owned planets is allowed (pretty sure yes)

        x = torch.cat((owner_one_hot_encoding(planet_owners, self.player_id), 
                       x[:, :, 1:5],
                       owner_one_hot_encoding(transporter_owners, self.player_id), 
                       x[:, :, 6:]
                       ), dim=-1)

        
        features = self.feature_extractor(x.flatten(start_dim=1))
        
        # Get action distributions
        source_logits = self.source_actor(features)
        target_logits = self.target_actor(features)
        ratio_mean = torch.sigmoid(self.ratio_actor_mean(features))  # Ensure 0-1 range
        


        # Create masked distributions
        # TODO: disallow sending transporters to themselves
        source_probs = MaskedCategorical(logits=source_logits, mask=source_mask)
        
        # Ratio distribution
        ratio_std = torch.clamp(self.ratio_actor_logstd.exp(), min=0.01, max=0.5)
        ratio_probs = Normal(ratio_mean, ratio_std)
        
        if action is None:
            # Sample actions
            source_action = source_probs.sample()
            target_mask = planet_owners == 2  # Mask for target actions (only opponent planets)
            target_mask[torch.arange(self.args.num_envs), source_action] = False  # Prevent sending to self
            target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)
            target_action = target_probs.sample()
            ratio_action = torch.clamp(ratio_probs.sample(), 0.0, 0.99)
            action = torch.stack([source_action.float(), target_action.float(), ratio_action.squeeze(-1)], dim=-1)
        else:
            source_action = action[:, 0].long()
            target_mask = planet_owners == 2
            target_mask[torch.arange(self.args.minibatch_size), source_action] = False  # Prevent sending to self
            target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)
            target_action = action[:, 1].long()
            ratio_action = action[:, 2]
        
        # Calculate log probabilities
        source_logprob = source_probs.log_prob(source_action)
        target_logprob = target_probs.log_prob(target_action)
        ratio_logprob = ratio_probs.log_prob(ratio_action).squeeze(-1)
        
        # Combined log probability
        total_logprob = source_logprob + target_logprob + ratio_logprob
        
        # Combined entropy
        total_entropy = source_probs.entropy() + target_probs.entropy() + ratio_probs.entropy().squeeze(-1)
        
        value = self.critic(features)
        
        return action, total_logprob, total_entropy, value
    
class MaskedCategorical(Categorical):
    """Categorical distribution with action masking"""
    
    def __init__(self, logits=None, probs=None, mask=None):
        if mask is not None:
            # Set logits of invalid actions to very negative values
            if logits is not None:
                logits = torch.where(mask, logits, torch.tensor(-1e8, device=logits.device))
            elif probs is not None:
                probs = torch.where(mask, probs, torch.tensor(1e-8, device=probs.device))
        
        super().__init__(logits=logits, probs=probs)
        self.mask = mask
    
    def entropy(self):
        # Only calculate entropy for valid actions
        if self.mask is not None:
            # Get probabilities and zero out invalid actions
            p_log_p = self.logits * self.probs
            p_log_p = torch.where(self.mask, p_log_p, torch.tensor(0.0, device=p_log_p.device))
            return -p_log_p.sum(-1)
        return super().entropy()