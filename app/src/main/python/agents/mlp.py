import math

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np
from gym_utils.distributions import MaskedCategorical, SigmoidTransformedDistribution
from gym_utils.gym_wrapper import owner_one_hot_encoding


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PlanetWarsAgentMLP(nn.Module):
    """Neural network agent with action masking"""
    
    def __init__(self, args, player_id=1, exploit=True):
        super().__init__()
        self.args = args
        self.player_id = player_id
        self.exploit = exploit

        # Input dimensions based on whether adjacency matrix is used
        node_features_dim = 30 * (args.node_feature_dim + 4) # 4 extra from one-hot encoding of transporter owners and planet owners
        total_input_dim = node_features_dim
        
        if args.use_adjacency_matrix:
            adj_matrix_dim = 30 * 30
            total_input_dim += adj_matrix_dim
        
        # Adjust network size based on input dimension
        hidden_size = args.hidden_dim
        self.hidden_dim = hidden_size
        
        # Feature extraction
        self.v_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(total_input_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )
        self.a_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(total_input_dim, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )
        
        # Value head - uses features from v_feature_extractor
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        
        # Policy heads
        self.source_actor = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 30), std=0.01),
        )
        
        self.target_actor = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 30), std=0.01),
        )

        # No-op Policy Head
        self.noop_actor = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01)
        )
        
        # Ship ratio actor
        if args.discretized_ratio_bins == 0:
            self.ratio_actor_mean = nn.Sequential(
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, 1), std=0.01),
            )
            self.ratio_actor_logstd = nn.Parameter(torch.zeros(1))
        else:
            #Discretized ratio actor
            self.ratio_actor = nn.Sequential(
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, args.discretized_ratio_bins-(0 if args.discretize_include_zero else 1)), std=0.01),
            )

    def get_value(self, x):
        planet_owners = x[:, :, 0]
        transporter_owners = x[:, :, 5]
        x = torch.cat((owner_one_hot_encoding(planet_owners, self.player_id), 
                       x[:, :, 1:5],
                       owner_one_hot_encoding(transporter_owners, self.player_id), 
                       x[:, :, 6:]
                       ), dim=-1)
        features = self.v_feature_extractor(x.flatten(start_dim=1))
        return self.critic(features)


    def get_action_and_value(self, x, action=None):
        batch_size = x.size(0)
        num_planets = 30

        # Extract planet owners before preprocessing (needed for target mask)
        planet_owners = x[:, :, 0]
        transporter_owners = x[:, :, 5]
        zero_growth_rate = x[:, :, 2] == 0

        source_mask = torch.logical_and(planet_owners == 1, transporter_owners == 0)  # Mask for source actions (only own planets with transporter not busy)
        #Add a dimension for the no-op action, which is always valid
        source_mask = torch.cat((torch.ones((batch_size, 1), dtype=torch.bool, device=x.device), source_mask), dim=1)
        x = torch.cat((owner_one_hot_encoding(planet_owners, self.player_id), 
                       x[:, :, 1:5],
                       owner_one_hot_encoding(transporter_owners, self.player_id), 
                       x[:, :, 6:]
                       ), dim=-1)

        a_features = self.a_feature_extractor(x.flatten(start_dim=1))
        v_features = self.v_feature_extractor(x.flatten(start_dim=1))
        
        value = self.critic(v_features)

        # Get per-planet logits for source selection with no-op
        source_node_logits = self.source_actor(a_features)  # [batch_size, 30]
        source_logits = torch.full((batch_size, num_planets + 1), fill_value=torch.finfo(torch.float32).min, device=a_features.device)  # +1 for no-op
        source_logits[:, 1:] = source_node_logits  # [batch_size, 30]
        
        # Get no-op action logits
        noop_logits = self.noop_actor(a_features)  # [batch_size, 1]
        source_logits[:, 0] = noop_logits.squeeze(-1)  # [batch_size, num_planets + 1]

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
            source_action = torch.clamp(source_action, max=num_planets)  # Clamp to valid range
            target_action = torch.zeros(batch_size, dtype=torch.long, device=source_action.device)
            ratio_action = torch.zeros(batch_size, 1, dtype=torch.float, device=source_action.device)
        else:
            # Use provided action
            source_action = action[:, 0].long()
            target_action = action[:, 1].long()
            ratio_action = action[:, 2].unsqueeze(-1)

        #Only sample target and ratio actions if source action is non-null
        valid_action_idx = source_action != 0
        if valid_action_idx.any():
            num_valid_actions = valid_action_idx.sum()

            # Get target logits for valid actions
            target_logits = self.target_actor(a_features[valid_action_idx])  # [num_valid, 30]

            # Create target mask
            valid_planet_owners = planet_owners[valid_action_idx]  # [num_valid, 30]
            if self.args.target_mask == "all":
                #We need to mask out padding, which can be identified by a 0 growth rate
                target_mask = torch.ones((num_valid_actions, num_planets), dtype=torch.bool, device=source_logits.device)
                target_mask = torch.logical_and(target_mask, ~zero_growth_rate[valid_action_idx])
            elif self.args.target_mask == "enemy":
                target_mask = (valid_planet_owners == 2)
            elif self.args.target_mask == "not_self":
                target_mask = (valid_planet_owners != 1)
            elif self.args.target_mask == "not_neutral":
                target_mask = (valid_planet_owners != 0)
            
            # Prevent sending to self (source_action - 1 because source_action includes no-op offset)
            target_mask[torch.arange(num_valid_actions), source_action[valid_action_idx] - 1] = False
            
            target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)

            if action is None:
                valid_target_action = target_probs.sample()
                valid_target_action = torch.clamp(valid_target_action, max=num_planets - 1)
                target_action[valid_action_idx] = valid_target_action
            else:
                valid_target_action = target_action[valid_action_idx]

            # Get ship ratio distribution
            ratio_input = a_features[valid_action_idx]

            if self.args.discretized_ratio_bins == 0:
                ratio_mean = self.ratio_actor_mean(ratio_input)
                ratio_std = self.ratio_actor_logstd.exp()
                ratio_std = torch.clamp(ratio_std, max=10.0)
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
                    ratio_action[valid_action_idx] = (ratio_action_bins + (0 if self.args.discretize_include_zero else 1)) / (self.args.discretized_ratio_bins-1)
                else:
                    discrete_ratio_action = (ratio_action[valid_action_idx] * (self.args.discretized_ratio_bins-1)).round()
                    discrete_ratio_action = discrete_ratio_action.long() - (0 if self.args.discretize_include_zero else 1)
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
        total_entropy = source_probs.entropy() + target_entropy + ratio_entropy
        
        return action, total_logprob, total_entropy, value

    def get_action(self, x):
        """Get action for test time."""
        with torch.no_grad():
            num_planets = 30

            if self.player_id == 1:
                planet_owners = x[:, :, 0]
                transporter_owners = x[:, :, 5]
            else:
                # This maps 0 to 0, 1 to 2, and 2 to 1
                planet_owners = (x[:, :, 0] * 2) % 3
                transporter_owners = (x[:, :, 5] * 2) % 3 
            zero_growth_rate = x[:, :, 2] == 0
            source_mask = torch.logical_and(planet_owners == 1, transporter_owners == 0)  # Mask for source actions (only own planets with transporter not busy)

            # One-hot encode planet owners and transporter owners, we already swapped the owners so we assume player_id is 1 (for this method)
            x = torch.cat((owner_one_hot_encoding(planet_owners, 1), 
                        x[:, :, 1:5],
                        owner_one_hot_encoding(transporter_owners, 1), 
                        x[:, :, 6:]
                        ), dim=-1)

            a_features = self.a_feature_extractor(x.flatten(start_dim=1))
            
            # Get source logits with no-op
            source_node_logits = self.source_actor(a_features)  # [1, 30]
            source_logits = torch.full((1, num_planets + 1), fill_value=torch.finfo(torch.float32).min, device=a_features.device)
            source_logits[0, 1:] = source_node_logits  # [1, 30]
            
            # Get no-op action logits
            noop_logits = self.noop_actor(a_features)  # [1, 1]
            source_logits[0, 0] = noop_logits  # [1, num_planets + 1]

            # Create masked distributions for source selection
            source_probs = MaskedCategorical(logits=source_logits, mask=source_mask)

            # Exploit or explore
            if self.exploit:
                source_action = source_probs.probs.argmax(dim=-1)  # [1]
            else:
                source_action = source_probs.sample()  # [1]

            # Initialize target and ratio actions
            target_action = torch.zeros(1, dtype=torch.long, device=source_action.device)
            ratio_action = torch.zeros(1, dtype=torch.float, device=source_action.device)
            
            # Only sample target and ratio actions if source action is non-null (not no-op)
            valid_action_idx = source_action != 0
            if valid_action_idx.any():
                # Get target logits
                target_logits = self.target_actor(a_features)  # [1, 30]

                # Create target mask
                if self.args.target_mask == "all":
                    target_mask = torch.ones((1, num_planets), dtype=torch.bool, device=source_logits.device)
                    target_mask = torch.logical_and(target_mask, ~zero_growth_rate)
                elif self.args.target_mask == "enemy":
                    target_mask = (planet_owners == 2)
                elif self.args.target_mask == "not_self":
                    target_mask = (planet_owners != 1)
                elif self.args.target_mask == "not_neutral":
                    target_mask = (planet_owners != 0)
                
                # Prevent sending to self (source_action - 1 because source_action includes no-op offset)
                target_mask[0, source_action[0] - 1] = 0
                
                target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)
                if self.exploit:
                    target_action = target_probs.probs.argmax(dim=-1)  # [1]
                else:
                    target_action = target_probs.sample()  # [1]

                # Get ship ratio distribution
                ratio_input = a_features

                if self.args.discretized_ratio_bins == 0:
                    ratio_logits = self.ratio_actor_mean(ratio_input)
                    if self.exploit:
                        ratio_action = torch.sigmoid(ratio_logits).squeeze(-1)
                    else:
                        ratio_std = self.ratio_actor_logstd.exp()
                        ratio_std = torch.clamp(ratio_std, max=10.0)
                        ratio_action = SigmoidTransformedDistribution(ratio_logits, ratio_std).sample().squeeze(-1)
                else:
                    if self.exploit:
                        ratio_action = torch.argmax(self.ratio_actor(ratio_input), dim=-1) + (0 if self.args.discretize_include_zero else 1)
                    else:
                        ratio_action = Categorical(logits=self.ratio_actor(ratio_input)).sample() + (0 if self.args.discretize_include_zero else 1)
                    ratio_action = ratio_action.float() / (self.args.discretized_ratio_bins-1)
            
            action = torch.cat([
                source_action.float(),
                target_action.float(),
                ratio_action
            ], dim=-1)

            return action

    def get_action_samples(self, x, source_mask, num_samples=10, temperatures={'source': 1.0, 'target': 1.0, 'ratio': 1.0}):
        """Sample multiple actions from the same observation without grad and return the batch of actions."""
        with torch.no_grad():
            num_planets = 30
            if self.exploit:
                num_samples = min(num_samples, num_planets)

            if self.player_id == 1:
                planet_owners = x[:, :, 0]
                transporter_owners = x[:, :, 5]
            else:
                planet_owners = (x[:, :, 0] * 2) % 3
                transporter_owners = (x[:, :, 5] * 2) % 3
            zero_growth_rate = x[:, :, 2] == 0

            x = torch.cat((owner_one_hot_encoding(planet_owners, 1), 
                        x[:, :, 1:5],
                        owner_one_hot_encoding(transporter_owners, 1), 
                        x[:, :, 6:]
                        ), dim=-1)

            a_features = self.a_feature_extractor(x.flatten(start_dim=1))

            # Get source logits with no-op
            source_node_logits = self.source_actor(a_features)  # [1, 30]
            source_logits = torch.zeros((1, num_planets + 1), device=a_features.device)
            source_logits[0, 1:] = source_node_logits
            
            # Get no-op action logits
            noop_logits = self.noop_actor(a_features)  # [1, 1]
            source_logits[0, 0] = noop_logits
            source_logits = source_logits / temperatures['source']

            # Create masked distributions for source selection
            source_probs = MaskedCategorical(logits=source_logits, mask=source_mask)

            # Sample
            if self.exploit:
                _, source_action = torch.topk(source_probs.probs, num_samples, dim=-1)
                source_action = source_action.squeeze(0)
            else:
                source_action = source_probs.sample((num_samples,)).squeeze(-1)

            # Initialize target and ratio actions
            target_action = torch.zeros(num_samples, dtype=torch.long, device=source_action.device)
            ratio_action = torch.zeros(num_samples, dtype=torch.float, device=source_action.device)

            # Only sample target and ratio actions if source action is non-null (not no-op)
            valid_action_idx = source_action != 0
            if valid_action_idx.any():
                num_valid_actions = valid_action_idx.sum()

                # Get target logits - expand features for valid actions
                valid_a_features = a_features.expand(num_valid_actions, -1)
                target_logits = self.target_actor(valid_a_features) / temperatures['target']  # [num_valid, 30]

                # Create target mask
                if self.args.target_mask == "all":
                    target_mask = torch.ones((num_valid_actions, num_planets), dtype=torch.bool, device=source_logits.device)
                    target_mask = torch.logical_and(target_mask, ~zero_growth_rate[valid_action_idx])
                elif self.args.target_mask == "enemy":
                    target_mask = (planet_owners == 2).expand(num_valid_actions, -1).clone()
                elif self.args.target_mask == "not_self":
                    target_mask = (planet_owners != 1).expand(num_valid_actions, -1).clone()
                elif self.args.target_mask == "not_neutral":
                    target_mask = (planet_owners != 0).expand(num_valid_actions, -1).clone()
                
                source_idx = source_action[valid_action_idx] - 1
                # Prevent sending to self
                target_mask[torch.arange(num_valid_actions), source_idx] = False
                
                target_probs = MaskedCategorical(logits=target_logits, mask=target_mask)
                if self.exploit:
                    target_action[valid_action_idx] = target_probs.probs.argmax(dim=-1)
                else:
                    target_action[valid_action_idx] = target_probs.sample()

                # Get ship ratio distribution
                ratio_input = a_features.expand(num_valid_actions, -1)

                if self.args.discretized_ratio_bins == 0:
                    ratio_logits = self.ratio_actor_mean(ratio_input)
                    if self.exploit:
                        ratio_action[valid_action_idx] = torch.sigmoid(ratio_logits).squeeze(-1)
                    else:
                        ratio_std = self.ratio_actor_logstd.exp()
                        ratio_std = torch.clamp(ratio_std, max=10.0) * math.sqrt(temperatures['ratio'])
                        ratio_action[valid_action_idx] = SigmoidTransformedDistribution(ratio_logits, ratio_std).sample().squeeze(-1)
                else:
                    if self.exploit:
                        ratio_action[valid_action_idx] = torch.argmax(self.ratio_actor(ratio_input), dim=-1) + (0 if self.args.discretize_include_zero else 1)
                    else:
                        ratio_action[valid_action_idx] = Categorical(logits=self.ratio_actor(ratio_input)).sample() + (0 if self.args.discretize_include_zero else 1)
                    ratio_action[valid_action_idx] = ratio_action[valid_action_idx].float() / (self.args.discretized_ratio_bins-1)
            
            action = torch.stack([
                source_action.float(),
                target_action.float(),
                ratio_action
            ], dim=-1)

            return action

    def copy(self):
        """Create a copy of the agent"""
        new_agent = PlanetWarsAgentMLP(self.args, self.player_id)
        new_agent.load_state_dict(self.state_dict())
        return new_agent
    
    def copy_as_opponent(self):
        """Create a copy of the agent as an opponent"""
        new_agent = PlanetWarsAgentMLP(self.args, self.player_id)
        new_agent.load_state_dict(self.state_dict())
        new_agent.player_id = 3 - self.player_id
        return new_agent