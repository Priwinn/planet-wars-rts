import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from util.gym_wrapper import PlanetWarsForwardModelEnv
from core.game_state import Player


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "PlanetWarsForwardModel"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.99
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.4
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Planet Wars specific
    num_planets: int = 6
    """number of planets in the game"""
    node_feature_dim: int = 13
    """dimension of node features (owner, ship_count, growth_rate, x, y)"""
    max_ticks: int = 2000
    """maximum game ticks"""
    use_adjacency_matrix: bool = False
    """whether to include adjacency matrix in observations"""
    
    # Opponent configuration
    opponent_type: str = "random"  # "random", "greedy", or "do_nothing"
    """type of opponent to train against"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, args):
    def thunk():
        if env_id == "PlanetWarsForwardModel":
            # Configure opponent policy
            if args.opponent_type == "random":
                opponent_policy = None  # Will use default random policy
            elif args.opponent_type == "greedy":
                opponent_policy = "greedy"  # Will be set after env creation
            else:  # do_nothing
                opponent_policy = lambda state, player: Action.do_nothing()
            
            env = PlanetWarsForwardModelEnv(
                controlled_player=Player.Player1,
                opponent_player=Player.Player2,
                max_ticks=args.max_ticks,
                game_params={
                    'numPlanets': args.num_planets,
                    'maxTicks': args.max_ticks,
                    'transporterSpeed': 3.0,
                    'width': 640,
                    'height': 480
                },
                opponent_policy=opponent_policy
            )
            
            # Set greedy opponent if specified
            if args.opponent_type == "greedy":
                env.set_opponent_policy(env._greedy_opponent_policy)
                
        else:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
        
        
        env = PlanetWarsActionWrapper(env, args.num_planets, args.use_adjacency_matrix)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


class PlanetWarsActionWrapper(gym.Wrapper):
    """Wrapper to flatten the tuple action space for Planet Wars"""
    
    def __init__(self, env, num_planets, use_adjacency_matrix=False):
        super().__init__(env)
        self.num_planets = num_planets
        self.use_adjacency_matrix = use_adjacency_matrix
        
        # Flatten action space: source_planet (discrete) + target_planet (discrete) + ship_ratio (continuous)
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0.0]),
            high=np.array([num_planets-1, num_planets-1, 1.0]),
            dtype=np.float32
        )
        
        # Calculate observation space size based on whether adjacency matrix is included
        obs_size = num_planets * 13  # node_features
        if use_adjacency_matrix:
            obs_size += num_planets * num_planets  # adjacency_matrix
        
        # Flatten observation space from Dict to Box
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def step(self, action):
        # Convert flattened action back to tuple format
        source_planet = int(np.clip(action[0], 0, self.num_planets-1))
        target_planet = int(np.clip(action[1], 0, self.num_planets-1))
        ship_ratio = np.clip(action[2], 0.0, 1.0)
        
        tuple_action = (source_planet, target_planet, np.array([ship_ratio]))
        obs, reward, done, truncated, info = self.env.step(tuple_action)
        
        # Flatten observation
        flat_obs = self._flatten_observation(obs)
        return flat_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_observation(obs), info
    
    def _flatten_observation(self, obs):
        """Flatten the graph observation to a 1D array"""
        if hasattr(obs, 'node_features') and hasattr(obs, 'adjacency_matrix'):
            # GraphObservation object
            node_features_flat = obs.node_features.flatten()
            
            if self.use_adjacency_matrix:
                adj_matrix_flat = obs.adjacency_matrix.flatten()
                return np.concatenate([node_features_flat, adj_matrix_flat]).astype(np.float32)
            else:
                return node_features_flat.astype(np.float32)
                
        elif isinstance(obs, dict):
            # Dict observation
            node_features_flat = obs['node_features'].flatten()
            
            if self.use_adjacency_matrix:
                adj_matrix_flat = obs['adjacency_matrix'].flatten()
                return np.concatenate([node_features_flat, adj_matrix_flat]).astype(np.float32)
            else:
                return node_features_flat.astype(np.float32)
        else:
            # Assume already flattened
            return np.array(obs).astype(np.float32)

    


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PlanetWarsAgent(nn.Module):
    """Neural network agent with action masking"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Input dimensions based on whether adjacency matrix is used
        node_features_dim = args.num_planets * args.node_feature_dim
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
        features = self.feature_extractor(x)
        return self.critic(features)


    def get_action_and_value(self, x, action=None):
        features = self.feature_extractor(x)
        
        # Get action distributions
        source_logits = self.source_actor(features)
        target_logits = self.target_actor(features)
        ratio_mean = torch.sigmoid(self.ratio_actor_mean(features))  # Ensure 0-1 range
        

        #Get source mask, target mask depends on source mask
        planet_owners = x[:, ::self.args.node_feature_dim]  # owner info is on dimensions positions pos s.t. pos % num_features_dim==0
        source_mask = planet_owners == 1  # Mask for source actions (only own planets)

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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.opponent_type}__adj_{args.use_adjacency_matrix}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args) for i in range(args.num_envs)],
    )

    agent = PlanetWarsAgent(args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, 3)).to(device)  # 3D action space
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Track performance metrics
    episode_rewards = []
    episode_lengths = []
    win_rate = []
    os.makedirs("models", exist_ok=True)
    
    print(f"Training with adjacency matrix: {args.use_adjacency_matrix}")
    print(f"Observation shape: {envs.single_observation_space.shape}")
    print(f"Network input size: {args.num_planets * args.node_feature_dim + (args.num_planets * args.num_planets if args.use_adjacency_matrix else 0)}")
    
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Log episode statistics
            if next_done.any():
                for i in range(args.num_envs):
                    if next_done[i]:
                        episode_length = infos.get("episode", {}).get("l", 0)[i]
                        episode_reward = infos.get("episode", {}).get("r", 0.0)[i]
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(episode_length)
                        win = reward[i] == 1.0 
                        win_rate.append(1.0 if win else 0.0)

                        print(f"global_step={global_step}, episodic_return={episode_reward:.3f}, length={episode_length}, win={win}")
                        writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)
                        
                        # Calculate recent win rate (last 10 episodes)
                        if len(win_rate) >= 10:
                            recent_win_rate = np.mean(win_rate[-10:])
                            writer.add_scalar("charts/win_rate", recent_win_rate, global_step)

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 3))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef


                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        # Performance metrics
        steps_per_second = int(global_step / (time.time() - start_time))
        mean_reward = rewards[-1000:].mean().item() if len(rewards) > 0 else 0.0
        
        writer.add_scalar("charts/SPS", steps_per_second, global_step)
        writer.add_scalar("charts/mean_reward", mean_reward, global_step)
        
        # Print progress
        if iteration % 10 == 0:
            wr = np.mean(win_rate[-10:]) if len(win_rate) >= 10 else np.mean(win_rate) if win_rate else 0.0
            print(f"Iteration {iteration}/{args.num_iterations}")
            print(f"  SPS: {steps_per_second}")
            print(f"  Mean reward: {mean_reward:.3f}")
            print(f"  Recent win rate: {wr:.3f}")
            print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save model checkpoint
        if iteration % 100 == 0:
            torch.save({
                'iteration': iteration,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
            }, f"models/{run_name}_iter_{iteration}.pt")

    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'iteration': iteration,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
    }, f"models/{run_name}_final.pt")

    envs.close()
    writer.close()
    
    print(f"\nTraining completed!")
    print(f"Final win rate: {np.mean(win_rate[-100:]) if len(win_rate) >= 100 else 0.0:.3f}")
    print(f"Model saved as: models/{run_name}_final.pt")