import gymnasium as gym
import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import time
import torch
from agents.baseline_policies import RandomPolicy, GreedyPolicy
from agents.planet_wars_agent import PlanetWarsPlayer

from gym_utils.self_play import SelfPlayBase
from gym_utils.KotlinForwardModelBridge import KotlinForwardModelBridge
from gym_utils.PythonForwardModelBridge import PythonForwardModelBridge
from util.gnn_utils import preprocess_graph_data, owner_one_hot_encoding
from torch_geometric.data import Data
from core.game_state import GameParams, Player, Action


def tensor_to_action(tensor: torch.Tensor, player_id: Player) -> Action:
    """Convert a tensor to an Action object"""
    if tensor.shape[0] != 3:
        raise ValueError("Tensor must have shape (3,) for Action conversion")
    
    source_planet = int(tensor[0]) 
    target_planet = int(tensor[1])
    num_ships = float(tensor[2])

    if source_planet == 0:
        # No-op action
        return Action.do_nothing()
    else:
        return Action(
            player_id=player_id,
            source_planet_id=source_planet-1,  # -1 to account for no-op action
            destination_planet_id=target_planet,
        num_ships=num_ships
        )

def are_there_valid_actions(game_state: Dict[str, Any], player_id: int) -> bool:
    """Check if there are any valid actions for the given player in the current game state."""
    for planet in game_state['planets']:
        if planet['owner'] == player_id and planet['transporter'] is None:
            return True
    return False

class PlanetWarsForwardModelEnv(gym.Env):
    """
    Gym environment that uses the Kotlin ForwardModel for local game simulation
    """
    
    def __init__(
        self,
        args,
        jar_path: str = None,
        max_distance_threshold: float = None,
        distance_power: float = 1.0,
        normalize_weights: bool = True,
        controlled_player: Player = Player.Player1,  # Which player the gym controls
        opponent_player: Player = Player.Player2,   # Opponent player
        max_ticks: int = 500,
        game_params: Optional[Dict[str, Any]] = None,
        opponent_policy: Optional[callable] = None,  # Function that takes game_state and returns Action
        self_play: Optional[SelfPlayBase] = None  # Function for self-play, if needed

    ):
        super().__init__()
        self.args = args
        if jar_path is None:
            self.bridge = PythonForwardModelBridge()
        else:
            self.bridge = KotlinForwardModelBridge(jar_path=jar_path)
        self.max_distance_threshold = max_distance_threshold
        self.distance_power = distance_power
        self.normalize_weights = normalize_weights
        self.controlled_player = controlled_player
        self.opponent_player = opponent_player
        self.max_ticks = max_ticks
        self.opponent_policy = opponent_policy or RandomPolicy(game_params, opponent_player)
        self.self_play = self_play
        self.previous_score = 0.0
        self.game_params = game_params or {
            'maxTicks': 500,
            'numPlanets': 10,
            'transporterSpeed': 3.0,
            'width': 640,
            'height': 480
        }
        
        self.player_int = 1 if controlled_player == Player.Player1 else 2
        self.opponent_int = 1 if opponent_player == Player.Player1 else 2
        
        # Get initial state to number of planets
        self.bridge.create_new_game(self.game_params)
        self.current_game_state = self.bridge.get_game_state()
        self.num_planets = len(self.current_game_state['planets'])
        self.edge_index = torch.Tensor([[i, j] for i in range(self.num_planets) for j in range(self.num_planets) if i != j]).long().permute(1, 0)
        
        
        # Define action and observation spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()
        
    def _create_action_space(self) -> gym.Space:
        """Create action space for fleet movements"""
        # Action: [source_planet, target_planet, ship_ratio (0-1)]
        return gym.spaces.Tuple(
            (
                gym.spaces.Discrete(self.num_planets),  # Source planet
                gym.spaces.Discrete(self.num_planets),  # Target planet
                gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)  # Ship ratio
            )
        )
    
    def _create_observation_space(self) -> gym.Space:
        """Create observation space for graph features"""
        # Node features: [owner, ship_count, growth_rate, x, y, transporter_info...]
        node_feature_dim = 16
        
        return gym.spaces.Dict({
            'node_features': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_planets, node_feature_dim),
                dtype=np.float32
            ),
            'adjacency_matrix': gym.spaces.Box(
                low=0, high=1,
                shape=(self.num_planets, self.num_planets),
                dtype=np.float32
            )
        })

    def reset(self, **kwargs) -> Tuple[Data, Dict[str, Any]]:
        """Reset the environment and return initial observation"""
        self.game_params['initialNeutralRatio'] = np.random.uniform(0.25, 0.35)
        self.game_params['transporterSpeed'] = np.random.uniform(2.0, 5.0)
        self.bridge.create_new_game(self.game_params)
        initial_state = self.bridge.get_game_state()
        obs = self._get_observation()
        if self.self_play:
            self.opponent_policy = self.self_play.get_opponent()
        if isinstance(self.opponent_policy, PlanetWarsPlayer):    
            self.opponent_policy.prepare_to_play_as(params=self.game_params, player=self.opponent_player)

        return obs, {
            'tick': initial_state['tick'],
            'leader': initial_state['leader'],
            'status': 'Game started',
            'player1Ships': initial_state['player1Ships'],
            'player2Ships': initial_state['player2Ships']
        }
    
    def step(self, action: np.ndarray) -> Tuple[Data, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        
        # Convert gym action to game action for controlled player
        controlled_action = self._convert_gym_action_to_game_action(action)
        
        # Get opponent action
        
        if isinstance(self.opponent_policy, PlanetWarsPlayer):
            opponent_action = self.opponent_policy.get_action(self.bridge.game_state)
        elif callable(self.opponent_policy):
            opponent_action = self.opponent_policy(self.current_game_state)
            
        # Create actions dict
        actions = {}
        actions[self.controlled_player] = controlled_action
        actions[self.opponent_player] = opponent_action
        
        # Step the forward model
        valid_actions_bool = are_there_valid_actions(self.current_game_state, self.player_int)
        self.current_game_state = self.bridge.step(actions)
        
        # Calculate reward based on current state
        reward = self._calculate_reward(self.current_game_state)

        # Check if done
        done = self.current_game_state['isTerminal'] or self.current_game_state['tick'] >= self.max_ticks

        # Check truncation
        truncated = self.current_game_state['tick'] >= self.max_ticks and not self.current_game_state['isTerminal']

        # Penalize for no-op actions if there is a planet to send ships from
        if controlled_action.source_planet_id == -1 and valid_actions_bool:
            reward -= 0.01

        # Additional info
        info = {
            'tick': self.current_game_state['tick'],
            'leader': self.current_game_state['leader'],
            'status': self.current_game_state.get('statusString', ''),
            'player1Ships': self.current_game_state['player1Ships'],
            'player2Ships': self.current_game_state['player2Ships'],
            'controlled_player': self.player_int,
            'opponent_player': self.opponent_int
        }
        if done: 
            print(f"Game over at tick {self.current_game_state['tick']}, leader: {self.current_game_state['leader']}")



        return self._get_observation(), reward, done, truncated, info
    
    def _convert_gym_action_to_game_action(self, gym_action: np.ndarray) -> Action:
        """Convert gym action to game engine action format"""
        source_planet = int(gym_action[0])
        target_planet = int(gym_action[1])
        ship_ratio = float(gym_action[2])
        if source_planet == 0:
            # No-op action
            return Action.do_nothing()
        else:
            source_planet -= 1  # -1 to account for no-op action
        
        planets = self.current_game_state['planets']

        # # Validate source planet
        if source_planet >= len(planets):
            print(f"Invalid source planet: {source_planet} for number of planets {len(planets)}")
            return Action.do_nothing()
        
        source_planet_data = planets[source_planet]
        
        # # Check if we own the source planet and it's not busy
        # if (source_planet_data['owner'] != self.player_int or 
        #     source_planet_data.get('transporter') is not None):
        #     return Action.do_nothing()
        
        # Calculate number of ships to send
        num_ships = source_planet_data['numShips'] * ship_ratio
        
        # Ships sent has to be positive and less than available ships
        # if num_ships <= 0 or num_ships >= source_planet_data['numShips']:
        #     return Action.do_nothing()
        
        # # Validate target planet (can't send to self)
        if target_planet == source_planet or target_planet >= len(planets):
            print(f"Invalid target planet: {target_planet} for source planet: {source_planet}, and number of planets {len(planets)}")
            return Action.do_nothing()
        
        return Action(
            player_id=self.controlled_player,
            source_planet_id=source_planet,
            destination_planet_id=target_planet,
            num_ships=num_ships
        )
    

    
    def _get_observation(self) -> Data:
        """Create graph observation from current game state"""
        planets = self.current_game_state['planets']
        node_features = torch.Tensor(np.stack([self._get_planet_features(p) for p in planets], axis=0))

        return Data(
            x = node_features,
            edge_index=self.edge_index,
            tick=torch.tensor(self.current_game_state['tick'] / self.game_params['maxTicks'], dtype=torch.float32)
        )
    
    def _owner_one_hot_encoding(self, owner: torch.Tensor) -> torch.Tensor:
        """Convert owner integer to one-hot encoding. Assume Neutral=0, Controlled=1, Opponent=2 (swaps controlled and opponent if needed)"""
        one_hot = torch.nn.functional.one_hot(
            owner.long(), num_classes=3
        )
        # Swap controlled and opponent if needed
        if self.controlled_player == Player.Player2:
            one_hot = one_hot[[0, 2, 1], :]
        return one_hot
    
    def _planet_index_one_hot_encoding(self, planet_index: torch.Tensor) -> torch.Tensor:
        """Convert planet index to one-hot encoding"""
        return torch.nn.functional.one_hot(
            planet_index.long(), num_classes=self.num_planets
        )
    
    def _get_planet_features(self, planet: Dict[str, Any]) -> np.ndarray:
        """Extract features from a single planet"""
        features = [
            planet['owner'],  # Owner ID
            planet['numShips'],  # Number of ships
            planet['growthRate'],  # Growth rate
            planet['x'] / self.game_params['width'],  # X coordinate
            planet['y'] / self.game_params['height']  # Y coordinate
        ]
        
        # Add transporter info if available
        if planet.get('transporter'):
            transporter = planet['transporter']
            features.extend([
                transporter['owner'],
                # transporter['sourceIndex'],
                transporter['destinationIndex'], 
                transporter['numShips'],
                # Normalized transporter position
                transporter['x']*transporter['vx']/(self.game_params['width'] * self.game_params['transporterSpeed']),
                transporter['y']*transporter['vy']/(self.game_params['height'] * self.game_params['transporterSpeed']),
                # transporter['vx'],
                # transporter['vy']
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        return np.array(features, dtype=np.float32)
    
    def _calculate_normalized_score_delta(self, game_state: Dict[str, Any]) -> float:
        """Calculate reward based on game state for the controlled player"""
        planets = game_state['planets']
        
        controlled_player_score = 0
        opponent_score = 0
        
        # Score based on planets owned and ships
        for planet in planets:
            owner = planet['owner']
            
            # Base score: ship (+ growth rate)
            planet_value = planet['numShips'] + planet['growthRate'] * 100 #np.sqrt(self.game_params['maxTicks']- game_state['tick'])

            if owner == self.player_int:
                controlled_player_score += planet_value
            elif owner == self.opponent_int:
                opponent_score += planet_value
            # Neutral planets don't count toward either player
        
        # Add transporter ships in transit
        transporters = game_state.get('transporters', [])
        for transporter in transporters:
            owner = transporter['owner']
            ships = transporter['numShips']
            
            if owner == self.player_int:
                controlled_player_score += ships
            elif owner == self.opponent_int:
                opponent_score += ships
        
        
        # Ongoing game normalized score delta
        total_score = controlled_player_score + opponent_score
        if total_score == 0:
            return 0.0
        
        return (controlled_player_score - opponent_score)

    def _calculate_change_in_score_delta(self, game_state: Dict[str, Any]) -> float:
        """Calculate change in score delta based on game state for the controlled player"""
        current_score = self._calculate_normalized_score_delta(game_state)
        previous_score = self.previous_score if self.previous_score is not None else 0
        self.previous_score = current_score
        return (current_score - previous_score)

    def _calculate_change_in_ship_delta(self, game_state: Dict[str, Any]) -> float:
        """Calculate change in ship delta based on game state for the controlled player"""
        current_score = self._calculate_ship_delta(game_state)
        previous_score = self.previous_score if self.previous_score is not None else 0
        self.previous_score = current_score

        return (current_score - previous_score)

    def _calculate_growth_rate(self, game_state: Dict[str, Any]) -> float:
        """Calculate growth rate based on game state for the controlled player"""
        planets = game_state['planets']
        
        controlled_growth = 0
        
        # Growth rate based on owned planets
        for planet in planets:
            if planet['owner'] == self.player_int:
                controlled_growth += planet['growthRate']
        return controlled_growth
    
    def _calculate_ship_delta(self, game_state: Dict[str, Any]) -> float:
        """Calculate delta based on ship ownership changes"""
        planets = game_state['planets']
        
        controlled_delta = 0
        opponent_delta = 0
        
        for planet in planets:
            if planet['owner'] == self.player_int:
                controlled_delta += planet['numShips']
            elif planet['owner'] == self.opponent_int:
                opponent_delta += planet['numShips']

        transporters = game_state.get('transporters', [])
        for transporter in transporters:
            if transporter['owner'] == self.player_int:
                controlled_delta += transporter['numShips']
            elif transporter['owner'] == self.opponent_int:
                opponent_delta += transporter['numShips']

        return (controlled_delta - opponent_delta)/(self.game_params['maxTicks'])
    
    def _calculate_growth_delta(self, game_state: Dict[str, Any]) -> float:
        """Calculate delta based on growth rate changes"""
        planets = game_state['planets']
        
        controlled_growth = 0
        opponent_growth = 0
        
        for planet in planets:
            if planet['owner'] == self.player_int:
                controlled_growth += planet['growthRate']
            elif planet['owner'] == self.opponent_int:
                opponent_growth += planet['growthRate']
        
        return (controlled_growth - opponent_growth)


    def _calculate_reward(self, game_state: Dict[str, Any]) -> float:
        """Calculate reward based on game state for the controlled player"""
        # reward = self._calculate_normalized_score_delta(game_state)*0.1
        # reward = self._calculate_growth_rate(game_state)/ self.game_params['maxTicks']*10
        # reward = self._calculate_growth_delta(game_state)*0.1
        # reward = self._calculate_ship_delta(game_state)*0.1
        reward = self._calculate_change_in_score_delta(game_state)/20
        
        # If game is terminal, give a final reward based on outcome
        if game_state['isTerminal'] or game_state['tick'] >= self.max_ticks:
            if game_state['leader'] == self.player_int:
                return 10.0
            elif game_state['leader'] == self.opponent_int:
                return -10.0
            else:
                return 0.0
        return reward
    
    def render(self, mode='human') -> Optional[np.ndarray]:
        """Render the environment"""
        if mode == 'human':
            game_state = self.bridge.get_game_state()
            print(f"Tick: {game_state['tick']}")
            print(f"Terminal: {game_state['isTerminal']}")
            print(f"Controlled Player: {self.player_int}")
            print(f"Player 1 Ships: {game_state['player1Ships']:.1f}")
            print(f"Player 2 Ships: {game_state['player2Ships']:.1f}")
            print(f"Leader: {game_state['leader']}")
            
            for i, planet in enumerate(game_state['planets'][:10]):  # Show first 10 planets
                owner_str = {0: "Neutral", 1: "Player 1", 2: "Player 2"}.get(
                    planet['owner'], f"Unknown({planet['owner']})"
                )
                controlled_marker = " (*)" if planet['owner'] == self.player_int else ""
                transporter_info = ""
                if planet.get('transporter'):
                    trans = planet['transporter']
                    transporter_info = f" [Transport: {trans['numShips']:.1f} ships]"
                
                print(f"Planet {i}: {owner_str}{controlled_marker}, "
                      f"Ships: {planet['numShips']:.1f}, "
                      f"Growth: {planet['growthRate']:.2f}{transporter_info}")
        
        return None
    
    def close(self):
        """Clean up resources"""
        self.bridge.cleanup()

    def set_opponent_policy(self, policy_func):
        """Set a custom opponent policy function"""
        self.opponent_policy = policy_func

class PlanetWarsForwardModelGNNEnv(PlanetWarsForwardModelEnv):
    """Forward model environment for Planet Wars with graph-based state representation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_attr = None

    def reset(self, **kwargs) -> Tuple[Data, Dict[str, Any]]:
        """Reset the environment and return initial observation"""

        # If num_planets is None, generate a random number of planets
        if self.args.num_planets is None:
            self.game_params['numPlanets'] = np.random.randint(self.args.num_planets_min, self.args.num_planets_max + 1)
        self.game_params['initialNeutralRatio'] = np.random.uniform(0.25, 0.35)
        self.game_params['transporterSpeed'] = np.random.uniform(2.0, 5.0)

        self.bridge.create_new_game(self.game_params)
        self.current_game_state = self.bridge.get_game_state()
        self.num_planets = len(self.current_game_state['planets'])
        if self.args.num_planets is None:
            self.edge_index = torch.Tensor([[i, j] for i in range(self.num_planets) for j in range(self.num_planets) if i != j]).long().permute(1, 0)
        print(f"Number of planets: {self.num_planets}")
        # Reset edge attributes based on new game state if newMapEachRun is True
        if self.game_params.get('newMapEachRun', True) or self.edge_attr is None:
            self.edge_attr = torch.Tensor(np.stack(
                [self._get_default_edge_features(edge[0], edge[1]) for edge in self.edge_index.permute(1, 0).numpy()]
            ))

        obs = self._get_observation()
        if self.self_play:
            self.opponent_policy = self.self_play.get_opponent()
        if isinstance(self.opponent_policy, PlanetWarsPlayer):    
            params = GameParams(**self.game_params)
            self.opponent_policy.prepare_to_play_as(params=params, player=self.opponent_player)
        return obs, {
            'tick': self.current_game_state['tick'],
            'leader': self.current_game_state['leader'],
            'status': 'Game started',
            'player1Ships': self.current_game_state['player1Ships'],
            'player2Ships': self.current_game_state['player2Ships']
        }

    def _get_observation(self) -> Data:
        if self.edge_attr is None:
            self.edge_attr = torch.Tensor(np.stack(
                [self._get_default_edge_features(edge[0], edge[1]) for edge in self.edge_index.permute(1, 0).numpy()]
            ))
        planets = self.current_game_state['planets']
        node_features = torch.Tensor(np.stack([self._get_planet_features(p) for p in planets], axis=0))

        edge_features = self.edge_attr.detach().clone()
        planets_with_transporters = [p for p in planets if p.get('transporter') is not None]
        for p in planets_with_transporters:
            edge_features[self._get_edge_index(p['id'], p['transporter']['destinationIndex'])] = self._get_transporter_features(p)

        return Data(
            x=node_features,
            edge_index=self.edge_index,
            edge_attr=edge_features,
            tick=self.current_game_state['tick'] / self.game_params['maxTicks']
        )




    def _get_planet_features(self, planet: Dict[str, Any]) -> np.ndarray:
        """Extract features from a single planet for GNN"""
        features = np.asarray([
            planet['owner'],  # Owner ID
            planet['numShips']/10,  # Number of ships
            planet['growthRate']*10,  # Growth rate
            1.0 if planet['transporter'] is not None else 0.0  # Has transporter
        ])
        return features
    def _get_transporter_features(self, planet) -> torch.Tensor:
        """Calculate edge features between two planets. Weight is normalized by game width/height and transporter speed."""
        if planet['transporter'] is not None:
            target_planet = self._get_planet_by_id(planet['transporter']['destinationIndex'])
            distance = np.sqrt((target_planet['x'] - planet['transporter']['x'])**2 + (target_planet['y'] - planet['transporter']['y'])**2) - target_planet['radius']
            weight = 10*self.game_params['transporterSpeed'] / (distance ** self.distance_power + 1e-8)
            return torch.FloatTensor([planet['transporter']['owner'], planet['transporter']['numShips']/10, weight])
        else:
            raise ValueError("Planet does not have a transporter")
    def _get_default_edge_features(self,i,j) -> np.ndarray:
        """Get default edge features for planets without transporters in use"""
        planet_i = self._get_planet_by_id(i)
        planet_j = self._get_planet_by_id(j)
        distance = np.sqrt((planet_i['x'] - planet_j['x']) ** 2 + (planet_i['y'] - planet_j['y']) ** 2) - planet_j['radius']
        weight = 10 * self.game_params['transporterSpeed'] / (distance ** self.distance_power + 1e-8)
        return np.array([0.0,0.0, weight], dtype=np.float32)
    def _get_planet_by_id(self, planet_id: int) -> Dict[str, Any]:
        """Get planet data by ID"""
        for planet in self.current_game_state['planets']:
            if planet['id'] == planet_id:
                return planet
        raise ValueError(f"Planet with ID {planet_id} not found in game state")
    def _get_edge_index(self,i,j) -> int:
        """Get edge index for graph representation. Considers no self-loops are present."""
        if j>i:
            return i * (self.num_planets-1) + j-1
        elif j<i:
            return i * (self.num_planets-1) + j
        else:
            raise ValueError("No self-loops allowed")

# Example usage
if __name__ == "__main__":
    print("=== Forward Model Gym Environment ===")
    
    # Create environment with greedy opponent
    env = PlanetWarsForwardModelGNNEnv(
        controlled_player=Player.Player1,
        opponent_player=Player.Player2,
        max_ticks=200,
        game_params={
            'numPlanets': 14,
            'maxTicks': 200,
            'transporterSpeed': 2.0,
            'width': 800,
            'height': 600,
        }
    )
    
    # Set greedy opponent
    env.set_opponent_policy(GreedyPolicy(
        game_params=env.game_params,
        player=env.opponent_player
    ))
    
    obs, info = env.reset()
    print(f"Graph has {len(obs.x)} nodes and {len(obs.edge_index)} edges")
    print(f"Initial state: {info}")
    
    # Simulate training loop
    total_reward = 0
    start_time = time.time()
    
    for step in range(50):
        # Random action for demonstration
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: Reward = {reward:.3f}, "
                  f"Total = {total_reward:.3f}, "
                  f"Done = {done}, "
                  f"Tick = {info['tick']}")
            print(f"  Player 1: {info['player1Ships']:.1f}, "
                  f"Player 2: {info['player2Ships']:.1f}, "
                  f"Leader: {info['leader']}")
        
        if done:
            print(f"Game finished at step {step}!")
            print(f"Final reward: {total_reward:.3f}")
            print(f"Winner: Player {info['leader']}")
            break
    
    end_time = time.time()
    steps_per_second = step / (end_time - start_time)
    print(f"\nPerformance: {steps_per_second:.1f} steps/second")
    
    env.close()
    print("Done!")

