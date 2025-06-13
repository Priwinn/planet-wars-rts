import gymnasium as gym
import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import time

from util.KotlinForwardModelBridge import KotlinForwardModelBridge
from torch_geometric.data import Data
from core.game_state import Player, Action

@dataclass
class GraphObservation:
    """Observation containing graph representation of the game state"""
    graph: nx.Graph
    node_features: np.ndarray  # Shape: (num_planets, feature_dim)
    edge_features: np.ndarray  # Shape: (num_edges, edge_feature_dim)
    adjacency_matrix: np.ndarray
    torch_geometric_data: Optional[Data] = None

class PlanetWarsForwardModelEnv(gym.Env):
    """
    Gym environment that uses the Kotlin ForwardModel for local game simulation
    """
    
    def __init__(
        self,
        jar_path: str = None,
        max_distance_threshold: float = None,
        distance_power: float = 1.0,
        normalize_weights: bool = True,
        controlled_player: Player = Player.Player1,  # Which player the gym controls
        opponent_player: Player = Player.Player2,   # Opponent player
        max_ticks: int = 500,
        game_params: Optional[Dict[str, Any]] = None,
        opponent_policy: Optional[callable] = None  # Function that takes game_state and returns Action
    ):
        super().__init__()
        
        self.kotlin_bridge = KotlinForwardModelBridge(jar_path=jar_path)
        self.max_distance_threshold = max_distance_threshold
        self.distance_power = distance_power
        self.normalize_weights = normalize_weights
        self.controlled_player = controlled_player
        self.opponent_player = opponent_player
        self.max_ticks = max_ticks
        self.opponent_policy = opponent_policy or self._random_opponent_policy
        self.previous_score = 0.0
        
        # Game parameters #TODO: Params is not exposed to the Kotlin side, so we need to set them here
        self.game_params = game_params or {
            'maxTicks': max_ticks,
            'numPlanets': 10,
            'transporterSpeed': 3.0,
            'width': 640,
            'height': 480
        }
        
        self.player_int = 1 if controlled_player == Player.Player1 else 2
        self.opponent_int = 1 if opponent_player == Player.Player1 else 2
        
        # Get initial state to determine action/observation spaces
        self.kotlin_bridge.create_new_game(self.game_params)
        initial_state = self.kotlin_bridge.get_game_state()
        self.num_planets = len(initial_state['planets'])
        
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
        node_feature_dim = 13
        
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

    def reset(self, **kwargs) -> Tuple[GraphObservation, Dict[str, Any]]:
        """Reset the environment and return initial observation"""
        self.kotlin_bridge.create_new_game(self.game_params)
        initial_state = self.kotlin_bridge.get_game_state()
        obs = self._get_observation()
        self.previous_score = self._calculate_normalized_score_delta(initial_state)
        return obs, {
            'tick': initial_state['tick'],
            'leader': initial_state['leader'],
            'status': 'Game started',
            'player1Ships': initial_state['player1Ships'],
            'player2Ships': initial_state['player2Ships']
        }
    
    def step(self, action: np.ndarray) -> Tuple[GraphObservation, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        
        # Convert gym action to game action for controlled player
        controlled_action = self._convert_gym_action_to_game_action(action)
        
        # Get opponent action
        current_state = self.kotlin_bridge.get_game_state()
        opponent_action = self.opponent_policy(current_state, self.opponent_player)
        
        # Create actions dict
        actions = {}
        actions[self.controlled_player] = controlled_action
        actions[self.opponent_player] = opponent_action
        
        # Step the forward model
        game_state = self.kotlin_bridge.step(actions)
        
        # Calculate reward based on current state
        reward = self._calculate_reward(game_state)
        
        # Check if done
        done = game_state['isTerminal'] or game_state['tick'] >= self.max_ticks
        
        # Check truncation
        truncated = game_state['tick'] >= self.max_ticks and not game_state['isTerminal']
                # Additional info
        info = {
            'tick': game_state['tick'],
            'leader': game_state['leader'],
            'status': game_state.get('statusString', ''),
            'player1Ships': game_state['player1Ships'],
            'player2Ships': game_state['player2Ships'],
            'controlled_player': self.player_int,
            'opponent_player': self.opponent_int
        }
        if done: 
            print(f"Game over at tick {game_state['tick']}, leader: {game_state['leader']}")
            # info['final_info'] = {'episode': {'r': reward, 'l': game_state['tick']}}



        return self._get_observation(), reward, done, truncated, info
    
    def _convert_gym_action_to_game_action(self, gym_action: np.ndarray) -> Action:
        """Convert gym action to game engine action format"""
        source_planet = int(gym_action[0])
        target_planet = int(gym_action[1])
        ship_ratio = float(gym_action[2][0])
        
        # Get current game state to check planet ownership and ships
        current_state = self.kotlin_bridge.get_game_state()
        planets = current_state['planets']
        
        # Validate source planet
        if source_planet >= len(planets):
            return Action.do_nothing()
        
        source_planet_data = planets[source_planet]
        
        # Check if we own the source planet and it's not busy
        if (source_planet_data['owner'] != self.player_int or 
            source_planet_data.get('transporter') is not None):
            return Action.do_nothing()
        
        # Calculate number of ships to send
        num_ships = source_planet_data['numShips'] * ship_ratio
        
        # Must have at least 1 ship and leave at least 1 ship on planet
        if num_ships < 1 or source_planet_data['numShips'] - num_ships < 1:
            return Action.do_nothing()
        
        # Validate target planet (can't attack self)
        if target_planet == source_planet or target_planet >= len(planets):
            return Action.do_nothing()
        
        return Action(
            player_id=self.controlled_player,
            source_planet_id=source_planet,
            destination_planet_id=target_planet,
            num_ships=num_ships
        )
    
    def _random_opponent_policy(self, game_state: Dict[str, Any], player: Player) -> Action:
        """Default random opponent policy"""
        planets = game_state['planets']
        player_int = 1 if player == Player.Player1 else 2
        
        # Find planets owned by the opponent that can send ships
        owned_planets = [
            p for p in planets 
            if (p['owner'] == player_int and 
                p['numShips'] > 5000 and 
                p.get('transporter') is None)
        ]
        
        if not owned_planets:
            return Action.do_nothing()
        
        # Random source planet
        source = np.random.choice(owned_planets)
        
        # Find target planets (not owned by this player)
        target_candidates = [p for p in planets if p['owner'] != player_int]
        
        if not target_candidates:
            return Action.do_nothing()
        
        # Random target
        target = np.random.choice(target_candidates)
        
        # Send random portion of ships (10-80%)
        ship_ratio = np.random.uniform(0.1, 0.8)
        num_ships = source['numShips'] * ship_ratio
        
        return Action(
            player_id=player,
            source_planet_id=source['id'],
            destination_planet_id=target['id'],
            num_ships=num_ships
        )
    
    def _greedy_opponent_policy(self, game_state: Dict[str, Any], player: Player) -> Action:
        """Greedy opponent policy - attacks weakest nearby targets"""
        planets = game_state['planets']
        player_int = 1 if player == Player.Player1 else 2
        
        # Find planets owned by the opponent that can send ships
        owned_planets = [
            p for p in planets 
            if (p['owner'] == player_int and 
                p['numShips'] > 10 and 
                p.get('transporter') is None)
        ]
        
        if not owned_planets:
            return Action.do_nothing()
        
        # Choose source planet with most ships
        source = max(owned_planets, key=lambda p: p['numShips'])
        
        # Find target planets (not owned by this player)
        target_candidates = [p for p in planets if p['owner'] != player_int]
        
        if not target_candidates:
            return Action.do_nothing()
        
        # Choose closest weak target
        def target_score(target):
            distance = np.sqrt((source['x'] - target['x'])**2 + (source['y'] - target['y'])**2)
            strength = target['numShips'] if target['owner'] == 0 else target['numShips'] * 1.5
            return distance + strength * 0.1
        
        target = min(target_candidates, key=target_score)
        
        # Send appropriate number of ships
        distance = np.sqrt((source['x'] - target['x'])**2 + (source['y'] - target['y'])**2)
        eta = distance / self.game_params.get('transporterSpeed', 3.0)
        estimated_defense = target['numShips'] + target['growthRate'] * eta
        
        num_ships = max(estimated_defense * 1.2, source['numShips'] * 0.5)
        num_ships = min(num_ships, source['numShips'] * 0.8)
        
        if num_ships < 1:
            return Action.do_nothing()
        
        return Action(
            player_id=player,
            source_planet_id=source['id'],
            destination_planet_id=target['id'],
            num_ships=num_ships
        )
    
    def _get_observation(self) -> GraphObservation:
        """Create graph observation from current game state"""
        game_state = self.kotlin_bridge.get_game_state()
        planets = game_state['planets']
        
        # Create graph
        graph = nx.Graph()
        
        # Add nodes (planets)
        for i, planet in enumerate(planets):
            graph.add_node(i, 
                          owner=planet['owner'],
                          ship_count=planet['numShips'],
                          growth_rate=planet['growthRate'],
                          x=planet['x'],
                          y=planet['y'],
                          transporter=planet.get('transporter', None))
        
        # Add edges with distance-based weights
        for i in range(len(planets)):
            for j in range(i + 1, len(planets)):
                p1, p2 = planets[i], planets[j]
                distance = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                
                # Skip if distance exceeds threshold
                if self.max_distance_threshold and distance > self.max_distance_threshold:
                    continue
                
                # Calculate inverse distance weight
                weight = 1.0 / (distance ** self.distance_power + 1e-8)
                graph.add_edge(i, j, distance=distance, weight=weight)
        
        # Normalize weights if requested
        if self.normalize_weights:
            weights = [data['weight'] for _, _, data in graph.edges(data=True)]
            if weights:
                max_weight = max(weights)
                for _, _, data in graph.edges(data=True):
                    data['weight'] /= max_weight
        
        # Extract features
        node_features = self._extract_node_features(graph)
        adjacency_matrix = nx.adjacency_matrix(graph, weight='weight').todense()
        
        return GraphObservation(
            graph=graph,
            node_features=node_features,
            edge_features=np.array([]),  # Not needed for this version
            adjacency_matrix=np.array(adjacency_matrix),
        )
    
    def _extract_node_features(self, graph: nx.Graph) -> np.ndarray:
        """Extract node features from graph"""
        features = []
        for node in graph.nodes():
            data = graph.nodes[node]
            planet_features = [
                data['owner'],
                data['ship_count'],
                data['growth_rate'],
                data['x'],
                data['y']
            ]
            if data['transporter']:
                transporter = data['transporter']
                planet_features.extend([
                    transporter['owner'],
                    transporter['sourceIndex'],
                    transporter['destinationIndex'],
                    transporter['numShips'],
                    transporter['x'],
                    transporter['y'],
                    transporter['vx'],
                    transporter['vy']
                ])
            else:
                planet_features.extend([0, 0, 0, 0, 0, 0, 0, 0])
            features.append(planet_features)
        return np.array(features, dtype=np.float32)
    
    def _calculate_normalized_score_delta(self, game_state: Dict[str, Any]) -> float:
        """Calculate reward based on game state for the controlled player"""
        planets = game_state['planets']
        
        controlled_player_score = 0
        opponent_score = 0
        
        # Score based on planets owned and ships
        for planet in planets:
            owner = planet['owner']
            
            # Base score: ships + growth potential
            planet_value = 1*planet['numShips'] + planet['growthRate'] * 1000
            
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
        
        return (controlled_player_score - opponent_score) / total_score
    
    def _calculate_reward(self, game_state: Dict[str, Any]) -> float:
        """Calculate reward based on game state for the controlled player"""
        current_score = self._calculate_normalized_score_delta(game_state)
        
        # Reward is the change in score since last step
        reward = current_score - self.previous_score
        self.previous_score = current_score
        
        # If game is terminal, give a final reward based on outcome
        if game_state['isTerminal']:
            if game_state['leader'] == self.player_int:
                return 1.0
            elif game_state['leader'] == self.opponent_int:
                return -1.0
            else:
                return 0.0
        return reward
    
    def render(self, mode='human') -> Optional[np.ndarray]:
        """Render the environment"""
        if mode == 'human':
            game_state = self.kotlin_bridge.get_game_state()
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
        self.kotlin_bridge.cleanup()

    def set_opponent_policy(self, policy_func):
        """Set a custom opponent policy function"""
        self.opponent_policy = policy_func


# Example usage
if __name__ == "__main__":
    print("=== Forward Model Gym Environment ===")
    
    # Create environment with greedy opponent
    env = PlanetWarsForwardModelEnv(
        controlled_player=Player.Player1,
        opponent_player=Player.Player2,
        max_ticks=200,
        game_params={
            'numPlanets': 15,
            'maxTicks': 200,
            'transporterSpeed': 2.0
        }
    )
    
    # Set greedy opponent
    env.set_opponent_policy(env._greedy_opponent_policy)
    
    obs, info = env.reset()
    print(f"Graph has {obs.graph.number_of_nodes()} nodes and {obs.graph.number_of_edges()} edges")
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