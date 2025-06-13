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

from util.gym_wrapper_server import PlanetWarsKotlinEnv
from core.game_state import Player


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
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
    env_id: str = "PlanetWars"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1  # Planet Wars is single env for now
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
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
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Planet Wars specific
    num_planets: int = 20
    """number of planets in the game"""
    node_feature_dim: int = 5
    """dimension of node features (owner, ship_count, growth_rate, x, y)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, args):
    def thunk():
        if env_id == "PlanetWars":
            env = PlanetWarsKotlinEnv(
                use_gym_agent_server=True,
                gym_agent_server_port=8080 + idx,  # Different ports for parallel envs
                controlled_player=Player.Player2,  # Gym controls player 2
            )
        else:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = PlanetWarsActionWrapper(env, args.num_planets)
        return env

    return thunk


class PlanetWarsActionWrapper(gym.Wrapper):
    """Wrapper to flatten the tuple action space for Planet Wars"""
    
    def __init__(self, env, num_planets):
        super().__init__(env)
        self.num_planets = num_planets
        
        # Flatten action space: source_planet (discrete) + target_planet (discrete) + ship_ratio (continuous)
        # We'll use a mixed approach: discrete for planets, continuous for ratio
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0.0]),
            high=np.array([num_planets-1, num_planets-1, 1.0]),
            dtype=np.float32
        )
        
        # Flatten observation space from Dict to Box
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_planets * 5 
                   + num_planets * num_planets
                   ,),  
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
            adj_matrix_flat = obs.adjacency_matrix.flatten()
            # return node_features_flat.astype(np.float32)
            return np.concatenate([node_features_flat, adj_matrix_flat]).astype(np.float32)
        elif isinstance(obs, dict):
            # Dict observation
            node_features_flat = obs['node_features'].flatten()
            adj_matrix_flat = obs['adjacency_matrix'].flatten()
            # return node_features_flat.astype(np.float32)
            return np.concatenate([node_features_flat, adj_matrix_flat]).astype(np.float32)
        else:
            # Assume already flattened
            return np.array(obs).astype(np.float32)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PlanetWarsAgent(nn.Module):
    """Neural network agent for Planet Wars with graph-aware architecture"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Input dimensions
        node_features_dim = args.num_planets * args.node_feature_dim
        adj_matrix_dim = args.num_planets * args.num_planets
        total_input_dim = node_features_dim + adj_matrix_dim
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Linear(total_input_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        
        # Value head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Policy heads for mixed action space
        # Source planet (discrete)
        self.source_actor = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, args.num_planets), std=0.01),
        )
        
        # Target planet (discrete) 
        self.target_actor = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, args.num_planets), std=0.01),
        )
        
        # Ship ratio (continuous)
        self.ratio_actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=0.01),
        )
        self.ratio_actor_logstd = nn.Parameter(torch.zeros(1))

    def get_value(self, x):
        features = self.feature_extractor(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.feature_extractor(x)
        
        # Get action distributions
        source_logits = self.source_actor(features)
        target_logits = self.target_actor(features)
        ratio_mean = self.ratio_actor_mean(features)
        
        source_probs = Categorical(logits=source_logits)
        target_probs = Categorical(logits=target_logits)
        ratio_probs = Normal(ratio_mean, self.ratio_actor_logstd.exp())
        
        if action is None:
            source_action = source_probs.sample()
            target_action = target_probs.sample()
            ratio_action = torch.clamp(ratio_probs.sample(), 0.0, 1.0)
            action = torch.stack([source_action.float(), target_action.float(), ratio_action.squeeze(-1)], dim=-1)
        else:
            source_action = action[:, 0].long()
            target_action = action[:, 1].long()
            ratio_action = action[:, 2]
        
        # Calculate log probabilities
        source_logprob = source_probs.log_prob(source_action)
        target_logprob = target_probs.log_prob(target_action)
        ratio_logprob = ratio_probs.log_prob(ratio_action.unsqueeze(-1)).squeeze(-1)
        
        # Combined log probability
        total_logprob = source_logprob + target_logprob + ratio_logprob
        
        # Combined entropy
        total_entropy = source_probs.entropy() + target_probs.entropy() + ratio_probs.entropy().squeeze(-1)
        
        value = self.critic(features)
        
        return action, total_logprob, total_entropy, value


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
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

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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
        print("SPS:", int(global_step / (time.time() - start_time)))
        print(f"Rewards: {rewards[-100:].mean().item():.3f}, ")

        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
