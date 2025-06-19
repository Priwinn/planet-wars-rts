import gymnasium as gym
import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import json
import asyncio
import websockets
import uuid
import threading
from queue import Queue, Empty
from websockets import serve
import time

from gym_utils.KotlinGameRunnerBridge import KotlinGameRunnerBridge
from torch_geometric.data import Data
from client_server.util import RemoteInvocationRequest, RemoteInvocationResponse, deserialize_args, serialize_result
from core.game_state import Player, camel_to_snake

@dataclass
class GraphObservation:
    """Observation containing graph representation of the game state"""
    graph: nx.Graph
    node_features: np.ndarray  # Shape: (num_planets, feature_dim)
    edge_features: np.ndarray  # Shape: (num_edges, edge_feature_dim)
    adjacency_matrix: np.ndarray
    torch_geometric_data: Optional[Data] = None

class GymAgentServer:
    """Server that acts as a PlanetWarsAgent for the gym environment"""

    def __init__(self, gym_env, host: str = "localhost", port: int = 8081, controlled_player: Player = Player.Player2):
        self.gym_env = gym_env
        self.host = host
        self.port = port
        self.agent_map: Dict[str, 'GymAgentWrapper'] = {}
        self.action_queue = Queue()
        self.server_task = None
        self.controlled_player = controlled_player

    async def handler(self, websocket):
        async for message in websocket:
            try:
                request = RemoteInvocationRequest.model_validate_json(message)
                
                if request.requestType == "init":
                    agent_id = str(uuid.uuid4())
                    agent = GymAgentWrapper(self.gym_env, self.action_queue, self.controlled_player)
                    self.agent_map[agent_id] = agent
                    result = {"objectId": agent_id}

                elif request.requestType == "invoke":
                    agent = self.agent_map.get(request.objectId)
                    if not agent:
                        raise ValueError(f"No such agent: {request.objectId}")

                    method_name = camel_to_snake(request.method)
                    method = getattr(agent, method_name, None)
                    if method is None:
                        raise ValueError(f"Unknown method: {request.method}")

                    args = deserialize_args(method_name, request.args)
                    result = method(*args)
                    result = serialize_result(result)

                elif request.requestType == "end":
                    removed = self.agent_map.pop(request.objectId, None)
                    msg = "Agent removed" if removed else "No such agent"
                    result = {"message": msg}

                else:
                    raise ValueError(f"Unknown request type: {request.requestType}")

                response = RemoteInvocationResponse(status="ok", result=result)

            except Exception as e:
                print(f"Error handling message: {e}")
                response = RemoteInvocationResponse(status="error", error=str(e))

            await websocket.send(response.model_dump_json())

    async def start_server(self):
        """Start the websocket server"""
        async with serve(self.handler, self.host, self.port):
            print(f"GymAgentServer running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever
            
    def start_server_thread(self):
        """Start server in a separate thread"""
        def run_server():
            asyncio.run(self.start_server())
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
    def get_action_from_queue(self, timeout=1.0):
        """Get action from the queue (called by gym environment)"""
        try:
            return self.action_queue.get(timeout=timeout)
        except Empty:
            return None

class GymAgentWrapper:
    """Wrapper that implements the PlanetWarsAgent interface for gym"""
    
    def __init__(self, gym_env, action_queue: Queue, player: Player):
        self.gym_env = gym_env
        self.action_queue = action_queue
        self.player = player
        self.params = None
        
    def get_action(self, game_state):
        """Called by the game engine - gets action from gym environment"""
        # Convert game state to gym observation
        start_time = time.time()
        obs = self.gym_env._game_state_to_observation(game_state)
        
        # Get action from gym (this would typically be from an RL agent)
        action = self.gym_env._get_gym_action(obs)
        
        # Convert gym action to game action
        game_action = self.gym_env._convert_gym_action_to_game_action(action, game_state)
        print(f"GymAgentWrapper.get_action took {time.time() - start_time:.2f} seconds")
        return game_action
        
    def get_agent_type(self) -> str:
        return "Gym Environment Agent"
        
    def prepare_to_play_as(self, player: Player, params, opponent: Optional[str] = None) -> str:
        self.player = player
        self.params = params
        return self.get_agent_type()
        
    def process_game_over(self, final_state) -> None:
        pass

class PlanetWarsKotlinEnv(gym.Env):
    """
    Gym environment that uses the Kotlin game engine via Py4J bridge
    with gym agent server integration
    """
    
    def __init__(
        self,
        jar_path: str = None,
        max_distance_threshold: float = None,
        distance_power: float = 1.0,
        normalize_weights: bool = True,
        use_gym_agent_server: bool = False,
        gym_agent_server_port: int = 8081,
        controlled_player: Player = Player.Player2,  # Which player the gym controls (1 or 2)
        max_ticks: int = 500
    ):
        super().__init__()
        
        self.kotlin_bridge = KotlinGameRunnerBridge(jar_path=jar_path)
        self.max_distance_threshold = max_distance_threshold
        self.distance_power = distance_power
        self.normalize_weights = normalize_weights
        self.use_gym_agent_server = use_gym_agent_server
        self.controlled_player = controlled_player
        self.max_ticks = max_ticks
        self.player_int : int = 1 if controlled_player == Player.Player1 else 2
        
        # Gym agent server
        if use_gym_agent_server:
            self.gym_agent_server = GymAgentServer(self, port=gym_agent_server_port, controlled_player=controlled_player)
            self.gym_agent_server.start_server_thread()
        else:
            self.gym_agent_server = None
            
        # Get initial state to determine action/observation spaces
        self.kotlin_bridge.new_game()
        initial_state = self.kotlin_bridge.get_game_state()
        self.num_planets = len(initial_state['planets'])
        
        # Define action and observation spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()
        
        # Current action from RL agent
        self.current_action = None
        
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
        # Node features: [owner, ship_count, growth_rate, x, y]
        node_feature_dim = 5
        
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
        self.kotlin_bridge.new_game()
        return self._get_observation(), {
            'tick': 0,
            'leader': self.controlled_player,
            'status': 'Game started',
            # 'episode': {
            #     'l': 0,  # Length of episode
            #     'r': 0,  # Initial reward
            # }
        }
    
    def step(self, action: np.ndarray) -> Tuple[GraphObservation, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        
        # Store the action for when the agent server requests it
        self.current_action = action
        
        # Step the game (this will call get_action on our gym agent server)
        game_state = self.kotlin_bridge.step_game()
        
        # Calculate reward based on current state
        reward = self._calculate_reward(game_state)
        
        # Check if done
        done = game_state['isTerminal'] or game_state['tick'] >= self.max_ticks
        if done: 
            print(f"Game over at tick {game_state['tick']}, winner: {game_state['leader']}")

        #Check truncation
        truncated = True if game_state['tick'] >= self.max_ticks else False #TODO: Implement truncation logic if needed

        # Additional info
        info = {
            'tick': game_state['tick'],
            'leader': game_state['leader'],
            'status': game_state.get('statusString', ''),
            # "episode": {
            #     'l': game_state['tick'],
            #     'r': 1 if reward > 0 else 0,
            # }
        }

        return self._get_observation(), reward, done, truncated, info

    def _get_gym_action(self, obs: GraphObservation) -> np.ndarray:
        """Get action from the current RL policy (to be overridden or set externally)"""
        if self.current_action is not None:
            return self.current_action
        else:
            # Default to random action if no action set
            return self.action_space.sample()
    
    def _game_state_to_observation(self, game_state) -> GraphObservation:
        """Convert game state to gym observation format"""
        # This would convert the Kotlin game state to the GraphObservation format
        # For now, just return current observation
        return self._get_observation()
    
    def _convert_gym_action_to_game_action(self, gym_action: np.ndarray, game_state):
        """Convert gym action to game engine action format"""
        from core.game_state import Action
        
        source_planet = int(gym_action[0])
        target_planet = int(gym_action[1])
        ship_ratio = float(gym_action[2][0])
        
        # Get current ships at source planet
        planets = game_state.planets if hasattr(game_state, 'planets') else game_state['planets']
        
        if source_planet < len(planets):
            source_planet_data = planets[source_planet]
            if hasattr(source_planet_data, 'n_ships'):
                num_ships = source_planet_data.n_ships * ship_ratio
            else:
                num_ships = source_planet_data['numShips'] * ship_ratio
        else:
            num_ships = 0
        
        # Create action
        if num_ships > 0:
            return Action(
                player_id=self.controlled_player,
                source_planet_id=source_planet,
                destination_planet_id=target_planet,
                num_ships=num_ships
            )
        else:
            return Action.do_nothing()
    
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
                          transporter=planet.get('transporter', 0))
        
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
            features.append([
                data['owner'],
                data['ship_count'],
                data['growth_rate'],
                data['x'],
                data['y']
            ])
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, game_state: Dict[str, Any]) -> float:
        """Calculate reward based on game state for the controlled player"""
        planets = game_state['planets']
        
        controlled_player_score = 0
        opponent_score = 0
        
        # Score based on planets owned and ships
        for planet in planets:
            owner = planet['owner']
            score = planet['numShips'] + planet['growthRate']
            
            if owner == self.player_int:
                controlled_player_score += score
            elif owner != -1:  # Not neutral
                opponent_score += score
            #Add transporter score if applicable
            if planet.get('transporter'):
                if planet['transporter'].get('owner', -1) == self.player_int:
                    controlled_player_score += planet['transporter'].get('numShips', 0)
                elif planet['transporter'].get('owner', -1) != 0:
                    opponent_score += planet['transporter'].get('numShips', 0)

        # Return normalized score difference
        total_score = controlled_player_score + opponent_score
        if total_score == 0:
            return 0.0
        
        return (controlled_player_score - opponent_score) / total_score
    
    def render(self, mode='human') -> Optional[np.ndarray]:
        """Render the environment"""
        if mode == 'human':
            game_state = self.kotlin_bridge.get_game_state()
            print(f"Tick: {game_state['tick']}")
            print(f"Terminal: {game_state['isTerminal']}")
            print(f"Controlled Player: {self.controlled_player}")
            
            for i, planet in enumerate(game_state['planets']):
                owner_str = {-1: "Neutral", 0: "Player 1", 1: "Player 2"}.get(
                    planet['owner'], f"Unknown({planet['owner']})"
                )
                controlled_marker = " (*)" if planet['owner'] == self.controlled_player else ""
                print(f"Planet {i}: {owner_str}{controlled_marker}, Ships: {planet['numShips']}, Growth: {planet['growthRate']}")
        
        return None
    
    def close(self):
        """Clean up resources"""
        self.kotlin_bridge.cleanup()


# Example usage
if __name__ == "__main__":
    # Example with gym agent server
    print("=== Gym Agent Server Environment ===")
    env = PlanetWarsKotlinEnv(
        use_gym_agent_server=True,
        controlled_player=Player.Player2,  # Gym controls player 2
        gym_agent_server_port=8080
    )
    
    obs, info = env.reset()
    print(f"Graph has {obs.graph.number_of_nodes()} nodes and {obs.graph.number_of_edges()} edges")
    print("Gym agent server started on port 8080")
    print("Kotlin game can now connect to ws://localhost:8080")

    # Simulate training loop
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: Reward = {reward:.3f}, Done = {done}, Truncated = {truncated}, Tick = {info['tick']}")

        if done:
            print("Game finished!")
            break
    
    env.close()