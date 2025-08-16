from dataclasses import dataclass, field
import os

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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "planet-wars-ppo"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "PlanetWarsForwardModel"
    """the id of the environment. Filled in runtime, either `PlanetWarsForwardModel` or `PlanetWarsForwardModelGNN` according to agent type"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    num_envs: int = 24
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    anneal_ent_coef: bool = False
    """Toggle entropy coefficient annealing"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 96
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0005
    """coefficient of the entropy"""
    vf_coef: float = 1.3
    """coefficient of the value function"""
    max_grad_norm: float = 3.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Planet Wars specific
    agent_type: str = "gnn"  # "mlp" or "gnn"
    num_planets: int = None
    """number of planets in the game. If None, will be set to a random value between num_planets_min and num_planets_max (new_map_each_run needs to be set to true)"""
    num_planets_min: int = 10
    """minimum number of planets in the game"""
    num_planets_max: int = 30
    """maximum number of planets in the game"""
    node_feature_dim: int = 0 #Filled in runtime 5 for gnn, 14 for mlp
    """dimension of node features (owner, ship_count, growth_rate, x, y)"""
    max_ticks: int = 2000
    """maximum game ticks"""
    use_adjacency_matrix: bool = False
    """whether to include adjacency matrix in observations"""
    flatten_observation: bool = True
    """Filled on run time, mlp uses flattened observation, gnn uses graph observation"""
    discretized_ratio_bins: int = 0
    """number of bins for the discretized ratio actor. Set to 0 to disable discretization"""
    new_map_each_run: bool = True
    """whether to create a new map for each run or use the same map"""
    hidden_dim: int = 256
    """hidden dimension for the layers"""
    profile_path: str = None
    """Path to save profiling data, if None profiling is disabled"""
    use_async: bool = True
    """if toggled, AsyncVectorEnv will be used"""
    use_tick: bool = False
    """if toggled, the game tick will be passed as an observation"""
    model_weights: str = "models/PlanetWarsForwardModelGNN__ppo_config__random__1755299838_final.pt"
    """If specified, the initial model weights will be loaded from this path"""
    resume_iteration: int = None
    """The iteration to resume training from, for annealing purposes"""

    # Opponent configuration
    opponent_type: str = "random"  # "random", "greedy", "focus", "defensive"
    """type of opponent to train against"""
    self_play: str = "baseline_buffer"  # "naive", "buffer", "baseline_buffer"
    """self-play strategy to use, if applicable"""
    buffer_opponents: list = field(default_factory=lambda: ["models/PlanetWarsForwardModelGNN__ppo_config__random__1755299838_iter_600.pt",
                              "models/PlanetWarsForwardModelGNN__ppo_config__random__1755299838_iter_700.pt",
                              "models/PlanetWarsForwardModelGNN__ppo_config__random__1755299838_iter_800.pt"])
    """list of opponents to use for buffer"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    run_name: str = 'temp'
    """the name of the run (computed in runtime)"""