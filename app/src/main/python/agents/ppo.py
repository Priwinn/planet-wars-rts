import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import shutil
import time
from dataclasses import dataclass
import sys
from itertools import chain
import glob

import gymnasium as gym
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from typing import List
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from gym_utils.gym_wrapper import PlanetWarsForwardModelEnv, PlanetWarsForwardModelGNNEnv
from core.game_state import Player, GameParams
from agents.mlp import PlanetWarsAgentMLP
from agents.gnn import PlanetWarsAgentGNN, GraphInstanceToPyG
from agents.passive_agent import PassiveAgent
from agents.baseline_policies import GreedyPolicy,RandomPolicy, FocusPolicy, DefensivePolicy
from agents.GalacticArmada import GalacticArmada
from agents.random_agents import CarefulRandomAgent, PureRandomAgent
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.better_greedy_heuristic_agent import BetterGreedyHeuristicAgent
from agents.aggressive_greedy_heuristic_agent import AggressiveGreedyHeuristicAgent
from agents.torch_agent_gnn import TorchAgentGNN
from gym_utils.self_play import get_self_play_class
from gym_utils.gnn_utils import preprocess_graph_data, owner_one_hot_encoding
from config_files.ppo_config import Args

def make_env(env_id, idx, capture_video, run_name, device, args, self_play=None):
    def thunk():        


        if env_id == "PlanetWarsForwardModel":
            env = PlanetWarsForwardModelEnv(
                args,
                controlled_player=Player.Player1,
                opponent_player=Player.Player2,
                max_ticks=args.max_ticks,
                game_params={
                    'numPlanets': args.num_planets,
                    'maxTicks': args.max_ticks,
                    'transporterSpeed': 3.0,
                    'width': 640,
                    'height': 480,
                    'newMapEachRun': args.new_map_each_run
                }
            )
        elif env_id == "PlanetWarsForwardModelGNN":
            if args.num_planets is None:
                num_planets = np.random.randint(args.num_planets_min, args.num_planets_max + 1)
            else:
                num_planets = args.num_planets

            env = PlanetWarsForwardModelGNNEnv(
                args,
                controlled_player=Player.Player1,
                opponent_player=Player.Player2,
                max_ticks=args.max_ticks,
                game_params={
                    'numPlanets': num_planets,
                    'maxTicks': args.max_ticks,
                    'transporterSpeed': np.random.uniform(2.0, 5.0),
                    'width': 640,
                    'height': 480,
                    'newMapEachRun': args.new_map_each_run,
                    'minGrowthRate': 0.05,
                    'maxGrowthRate': 0.2,
                    'initialNeutralRatio': np.random.uniform(0.25, 0.35)
                },
                self_play= self_play if args.self_play else None
            )
        if args.opponent_type == "greedy":
            env.set_opponent_policy(GreedyPolicy(game_params=env.game_params, player=Player.Player2))
        elif args.opponent_type == "passive":
            opponent = PassiveAgent()
            opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
            env.set_opponent_policy(opponent)
        elif args.opponent_type == "random":
            env.set_opponent_policy(RandomPolicy(game_params=env.game_params, player=Player.Player2))
        elif args.opponent_type == "focus":
            env.set_opponent_policy(FocusPolicy(game_params=env.game_params, player=Player.Player2))
        elif args.opponent_type == "defensive":
            env.set_opponent_policy(DefensivePolicy(game_params=env.game_params, player=Player.Player2))
        elif args.opponent_type == "careful_random":
            opponent = CarefulRandomAgent()
            opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
            env.set_opponent_policy(opponent)
        elif args.opponent_type == "better_greedy":
            opponent = BetterGreedyHeuristicAgent()
            opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
            env.set_opponent_policy(opponent)
        elif args.opponent_type == "aggressive_greedy":
            opponent = AggressiveGreedyHeuristicAgent()
            opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
            env.set_opponent_policy(opponent)
        elif args.opponent_type == "galactic":
            opponent = GalacticArmada()
            opponent.prepare_to_play_as(params=GameParams(**env.game_params), player=Player.Player2)
            env.set_opponent_policy(opponent)

        env = PlanetWarsActionWrapper(env, num_planets, args.use_adjacency_matrix, args.flatten_observation, device, node_feature_dim=args.node_feature_dim)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeReward(env)
        return env

    return thunk

def make_vector_env(env_id, capture_video, run_name, device, args, self_play=None):
    if args.use_async:
        return gym.vector.AsyncVectorEnv([make_env(env_id, i, capture_video, run_name, device, args, self_play=self_play) for i in range(args.num_envs)], shared_memory=False)
    else:
        return gym.vector.SyncVectorEnv([make_env(env_id, i, capture_video, run_name, device, args, self_play=self_play) for i in range(args.num_envs)])

class PlanetWarsActionWrapper(gym.Wrapper):
    """Wrapper to flatten the tuple action space for Planet Wars"""

    def __init__(self, env, num_planets, use_adjacency_matrix=False, flatten_observation=False, device='cpu', node_feature_dim=0):
        super().__init__(env)
        self.num_planets = num_planets
        self.use_adjacency_matrix = use_adjacency_matrix
        self.flatten_observation = flatten_observation
        self.device = device
        self.node_feature_dim = node_feature_dim

        # Action space: source_planet (discrete) + target_planet (discrete) + ship_ratio (continuous)
        self.action_space = gym.spaces.Box(
                low=np.array([0, 0, 0.0]),
                high=np.array([args.num_planets_max-1, args.num_planets_max-1, 1.0]),
                dtype=np.float32
            )
        if flatten_observation:
            # Calculate observation space size based on whether adjacency matrix is included
            # obs_size = num_planets * self.node_feature_dim  # node_features
            # if use_adjacency_matrix:
            #     obs_size += num_planets * num_planets  # adjacency_matrix
        
            # Flatten observation space from Dict to Box
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_planets,self.node_feature_dim),
                dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Graph(node_space=gym.spaces.Box(low=0, high=1000, shape=(14,), dtype=np.float32),
                                                  edge_space=gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32))
    
    def step(self, action):
        # Convert flattened action back to tuple format
        source_planet = int(action[0])
        target_planet = int(action[1])
        ship_ratio = np.clip(action[2], 0.0, 1.0)
        
        tuple_action = (source_planet, target_planet, np.array([ship_ratio]))
        obs, reward, done, truncated, info = self.env.step(tuple_action)
        
        # Flatten observation
        if self.flatten_observation:
            obs = obs.x
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.flatten_observation:
            obs = obs.x
        return obs, info

if __name__ == "__main__":
    global_step = 0
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.flatten_observation = args.agent_type != "gnn"
    args.env_id = "PlanetWarsForwardModelGNN" if args.agent_type == "gnn" else "PlanetWarsForwardModel"
    args.node_feature_dim = 5 if args.agent_type == "gnn" else 10
    if args.use_tick:
        args.node_feature_dim += 1  # Add tick feature if use_tick is enabled
    if args.exp_name is None:
        args.run_name = f"{os.path.basename(__file__)[: -len('.py')]}__{int(time.time())}"
    else:
        args.run_name = f"{args.exp_name}__{int(time.time())}"

    if args.use_async:
        import multiprocessing as mp
        mp.set_start_method("spawn")
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )
        print(f"Logging code for {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
        wandb.run.log_code(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if args.profile_path is not None:
        os.makedirs(args.profile_path, exist_ok=True)
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                args.profile_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.agent_type == "gnn":
        agent = PlanetWarsAgentGNN(args).to(device)
        agent = torch.compile(agent, dynamic=True)
    else:
        agent = PlanetWarsAgentMLP(args).to(device)
    if args.optimizer == "muon":
        from heavyball import ForeachMuon
        optimizer = ForeachMuon(agent.parameters(), lr=args.learning_rate, eps=1e-5, betas=(0.9, 0.99))
    else:
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)

    if args.model_weights is not None:
        state_dict = torch.load(args.model_weights, map_location=torch.device('cpu'), weights_only=False)
        agent.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        global_step = state_dict['iteration'] * args.num_envs * args.num_steps

    if args.self_play:
        self_play = get_self_play_class(args.self_play)(player_id=2)
        for opponent in args.buffer_opponents:
            self_play.add_opponent(TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path=opponent, device=args.opponent_device))
        self_play.add_opponent(TorchAgentGNN(model=agent.copy_as_opponent(), device=args.opponent_device)) 
        args.use_async = False # Override async setting for self-play
    else:
        self_play = None

    # Environment setup
    envs = make_vector_env(
        env_id=args.env_id,
        capture_video=args.capture_video,
        run_name=args.run_name,
        device=device,
        args=args,
        self_play=self_play
    )


    # Storage setup
    if args.flatten_observation:
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    else:
        obs = [[PyGData(x=torch.zeros((0, args.node_feature_dim), dtype=torch.float32),
                       edge_index=torch.zeros(2, 0), dtype=torch.int64)
                for _ in range(args.num_envs)]
                for _ in range(args.num_steps)]

    actions = torch.zeros((args.num_steps, args.num_envs, 3)).to(device) 
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    if args.flatten_observation:
        next_obs = torch.Tensor(next_obs).to(device)
    # else:
    #     next_obs = [(o).to(device) for o in next_obs]
    next_done = torch.zeros(args.num_envs).to(device)
    curriculum_step = 0
    lesson_number = 0
    lesson_episode_count = 0

    # Track performance metrics
    episode_rewards = []
    episode_lengths = []
    win_rate = []
    player_conquers = []
    neutral_conquers = []
    planets_lost = []
    
    os.makedirs("models", exist_ok=True)

    ent_coef = args.ent_coef
    start_iteration = 1
    if args.anneal_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iterations)
        if args.model_weights is not None:
            scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            start_iteration = state_dict['iteration'] + 1

    for iteration in range(start_iteration, args.num_iterations + 1):

        if args.anneal_ent_coef:
            frac = 0.5+np.cos(np.pi * (iteration - 1) / args.num_iterations) / 2.0
            ent_coef = frac * ent_coef
            
        penalized_noops = 0

        for step in range(0, args.num_steps):
            
            global_step += args.num_envs
            curriculum_step += args.num_envs
            if args.flatten_observation or step == 0:
                obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                if args.flatten_observation:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                else:
                    input, source_mask = preprocess_graph_data(next_obs, agent.player_id, args.use_tick)
                    action, logprob, _, value = agent.get_action_and_value(input.to(device), source_mask=source_mask.to(device))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            if args.flatten_observation:
                next_obs = torch.Tensor(next_obs).to(device)
            else:
                if step < args.num_steps - 1:
                    obs[step+1] = [o for o in next_obs]  # Keep on cpu for dataloader
                
            next_done = torch.Tensor(next_done).to(device)

            # Track penalized no-ops
            penalized_noops += sum(infos.get("penalized_noop", [0]*args.num_envs))

            # if args.profile_path is not None:
            #     profiler.step()  # Step the profiler if profiling is enabled

            # Log episode statistics
            if next_done.any():
                for i in range(args.num_envs):
                    if next_done[i]:
                        episode_length = infos.get("episode", {}).get("l", 0)[i]
                        episode_reward = infos.get("episode", {}).get("r", 0.0)[i]
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(episode_length)
                        win = infos.get("leader")[i] == infos.get("controlled_player")[i]
                        win_rate.append(1.0 if win else 0.0)
                        player_conquers.append(infos.get("controlled_conquers", [0]*args.num_envs)[i])
                        neutral_conquers.append(infos.get("neutral_conquers", [0]*args.num_envs)[i])
                        planets_lost.append(infos.get("opponent_conquers", [0]*args.num_envs)[i])
                        lesson_episode_count += 1

                        print(f"global_step={global_step}, episodic_return={episode_reward:.3f}, length={episode_length}, win={win}")
                        writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)

                        #Log conquer stats
                        writer.add_scalar("conquer_stats/player_conquers", player_conquers[-1], global_step)
                        writer.add_scalar("conquer_stats/neutral_conquers", neutral_conquers[-1], global_step)
                        writer.add_scalar("conquer_stats/planets_lost", planets_lost[-1], global_step)
                        #Conquer_rate relative to number of planets
                        num_planets = infos.get("num_planets", [args.num_planets_max]*args.num_envs)[i]
                        neutral_planets = infos.get("neutral_planets", [int(0.3*args.num_planets_max)]*args.num_envs)[i]
                        writer.add_scalar("conquer_stats/player_conquer_rate", player_conquers[-1]/num_planets, global_step)
                        writer.add_scalar("conquer_stats/neutral_conquer_rate", neutral_conquers[-1]/neutral_planets, global_step)
                        writer.add_scalar("conquer_stats/planets_lost_rate", planets_lost[-1]/num_planets, global_step)
                        writer.add_scalar("conquer_stats/conquer_difference", player_conquers[-1]-planets_lost[-1], global_step)

                        # Calculate recent win rate (last 50 episodes)
                        recent_win_rate = np.mean(win_rate[-50:]) if len(win_rate) >= 50 else np.mean(win_rate) if win_rate else 0.0
                        writer.add_scalar("charts/win_rate", recent_win_rate, global_step)
                        writer.add_scalar("charts/lesson_number", lesson_number, global_step)
                         # Reset curriculum step if win rate is good and move to next curriculum step
                        if recent_win_rate >= 0.8 and lesson_episode_count >= 50 and not args.self_play and lesson_number < args.curriculum_opponents.__len__()-1:
                            args.opponent_type = args.curriculum_opponents[lesson_number + 1]  # Switch to next baseline opponent
                            print(f"Lesson completed in {curriculum_step} steps, switching to lesson {lesson_number} with opponent type '{args.opponent_type}'")
                            curriculum_step = 0
                            lesson_episode_count = 0
                            lesson_number += 1
                            envs.close()
                            envs = make_vector_env(env_id=args.env_id, capture_video=args.capture_video, run_name=args.run_name, device=device, args=args, self_play=self_play)
                        if (recent_win_rate >= 0.8 and lesson_episode_count >= 50) and not args.self_play and lesson_number == args.curriculum_opponents.__len__()-1:
                            args.self_play = "baseline_buffer"  # Switch to self-play
                            self_play = get_self_play_class(args.self_play)(player_id=2)
                            print(f"Lesson completed in {curriculum_step} steps, switching to self-play with self-play type '{args.self_play}'.")
                            curriculum_step = 0
                            lesson_episode_count = 0
                            args.opponent_type = None
                            lesson_number += 1

                            #Save the model after switching to self-play
                            torch.save({
                                'iteration': iteration,
                                'model_state_dict': agent.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'args': args,
                                'scheduler_state_dict': scheduler.state_dict() if args.anneal_lr else None
                            }, f"models/{args.run_name}_galactic.pt")
                            if args.track:
                                wandb.save(f"models/{args.run_name}_galactic.pt")
                            

                            envs.close()
                            self_play.add_opponent(TorchAgentGNN(model=agent.copy_as_opponent(), device=args.opponent_device))
                            args.use_async=False
                            envs = make_vector_env(env_id=args.env_id, capture_video=args.capture_video, run_name=args.run_name, device=device, args=args, self_play=self_play)
                            envs.reset()

                        if (recent_win_rate >= 0.7 and lesson_episode_count >= 50 and args.self_play):
                            print(f"Lesson completed in {curriculum_step} steps, updating opponent policy in self-play.")
                            curriculum_step = 0
                            lesson_episode_count = 0
                            lesson_number += 1
                            self_play.add_opponent(TorchAgentGNN(model=agent.copy_as_opponent(), device=args.opponent_device))
                            envs = make_vector_env(env_id=args.env_id, capture_video=args.capture_video, run_name=args.run_name, device=device, args=args, self_play=self_play)
                            envs.reset()

        # Bootstrap value if not done
        with torch.no_grad():
            if args.flatten_observation:
                next_value = agent.get_value(next_obs).reshape(1, -1)
            else:
                input = preprocess_graph_data(next_obs, agent.player_id, args.use_tick, return_mask=False)
                next_value = agent.get_value(input.to(device)).reshape(1, -1)
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
        if args.flatten_observation:
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        else:
            b_obs = list(chain.from_iterable(obs))  # Flatten list of lists to a single list
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 3))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # with torch.autograd.detect_anomaly():
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                if args.flatten_observation:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                else:
                    input, source_mask = preprocess_graph_data([b_obs[idx] for idx in mb_inds], agent.player_id, args.use_tick)
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(input.to(device), b_actions[mb_inds], source_mask=source_mask.to(device))
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
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef


                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.profile_path is not None:
                    profiler.step()  # Step the profiler if profiling is enabled

            if args.target_kl is not None and approx_kl > args.target_kl:
                print("Early stopping at epoch {}: reached target KL divergence".format(epoch))
                break
        if args.anneal_lr:
            scheduler.step()
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
        steps_per_second = int(args.batch_size / (time.time() - start_time))
        start_time = time.time()
        mean_reward = rewards[-1000:].mean().item() if len(rewards) > 0 else 0.0
        
        writer.add_scalar("charts/SPS", steps_per_second, global_step)
        writer.add_scalar("charts/mean_reward", mean_reward, global_step)

        # Action statistics
        is_op = b_actions[:, 0] != 0
        ratio_mean = (b_actions[is_op, 2].mean() if b_actions.shape[1] > 2 else 0.0)
        ratio_std = (b_actions[is_op, 2].std() if b_actions.shape[1] > 2 else 0.0)
        source_counts = torch.bincount(b_actions[:, 0].long(), minlength=args.num_planets_max+1)
        _obs= [o for i,o in zip(is_op, b_obs) if i]
        target_owners = torch.Tensor([o.x[b_action[1].long().item()][0] for (o, b_action) in zip(_obs, b_actions[is_op])])
        target_own_freq = (target_owners == 1).sum().item() / len(target_owners) if len(target_owners) > 0 else 0.0
        target_neutral_freq = (target_owners == 0).sum().item() / len(target_owners) if len(target_owners) > 0 else 0.0
        target_enemy_freq = (target_owners == 2).sum().item() / len(target_owners) if len(target_owners) > 0 else 0.0
        penalized_noop_freq = penalized_noops / args.batch_size

        writer.add_scalar("action_stats/mean_action_ratio", ratio_mean, global_step)
        writer.add_scalar("action_stats/std_action_ratio", ratio_std, global_step)
        writer.add_scalar("action_stats/source_noop_freq", source_counts[0].float()/args.batch_size, global_step)
        writer.add_scalar("action_stats/target_own_freq", target_own_freq, global_step)
        writer.add_scalar("action_stats/target_neutral_freq", target_neutral_freq, global_step)
        writer.add_scalar("action_stats/target_enemy_freq", target_enemy_freq, global_step)
        writer.add_scalar("action_stats/penalized_noop_freq", penalized_noop_freq, global_step)

        sys.stdout.flush()

        # Save model checkpoint
        if iteration % 250 == 0:
            torch.save({
                'iteration': iteration,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'scheduler_state_dict': scheduler.state_dict() if args.anneal_lr else None
            }, f"models/{args.run_name}_iter_{iteration}.pt")
            if args.track:
                wandb.save(f"models/{args.run_name}_iter_{iteration}.pt")


    if args.profile_path is not None:
        profiler.stop()
        profile_files = glob.glob(f"{args.profile_path}/*")
        for file in profile_files:
            wandb.save(file, base_path=args.profile_path)

    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'iteration': iteration,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'scheduler_state_dict': scheduler.state_dict(),
    }, f"models/{args.run_name}_final.pt")
    if args.track:
        wandb.save(f"models/{args.run_name}_final.pt")


    envs.close()
    writer.close()
    # Delete profile files and its contents (recursively)
    if args.track:
        wandb.finish()
    if args.profile_path is not None:
        shutil.rmtree(args.profile_path)


    print(f"\nTraining completed!")
    print(f"Final win rate: {np.mean(win_rate[-100:]) if len(win_rate) >= 100 else 0.0:.3f}")
    print(f"Model saved as: models/{args.run_name}_final.pt")