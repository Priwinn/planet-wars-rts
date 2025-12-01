import copy
import time
from typing import Dict, Any, List, Tuple, Optional, Union

from core.game_state import Player, Action, GameState, GameParams
from core.forward_model_extended_metrics import ForwardModelWithMetrics as ForwardModel
from core.game_state_factory import GameStateFactory


class PythonForwardModelBridge:
    """Bridge to interact with Python ForwardModel for local game simulation"""
    
    def __init__(self):
        self.forward_model: Optional[ForwardModel] = None
        self.game_state: Optional[GameState] = None
        self.game_params: Optional[GameParams] = None
        self.initial_game_state: Optional[GameState] = None
    
    def create_new_game(self, game_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new game with ForwardModel"""
        
        # Create game parameters
        if game_params is None:
            self.game_params = GameParams()
        else:
            self.game_params = self._create_game_params_from_dict(game_params)
        
        # Create game state 
        if self.initial_game_state is not None and game_params.get('newMapEachRun', True) == False:
            # If we have an initial game state, use it
            self.game_state = self.initial_game_state.model_copy(deep=True)
        else:
            game_state_factory = GameStateFactory(self.game_params)
            self.game_state = game_state_factory.create_game()
            self.initial_game_state = self.game_state.model_copy(deep=True)  # Store initial state for cloning
        
        # Create forward model
        self.forward_model = ForwardModel(self.game_state, self.game_params)
        self.forward_model.reset_metrics()
        
        return self.get_game_state()
    
    def _create_game_params_from_dict(self, params_dict: Dict[str, Any]) -> GameParams:
        """Create GameParams from Python dict"""
        return GameParams(**params_dict)

    def step(self, actions: Dict[Player, Action]) -> Dict[str, Any]:
        """Step the forward model with given actions"""
        if self.forward_model is None:
            raise ValueError("No game created. Call create_new_game() first.")
        
        # Step the forward model
        self.forward_model.step(actions)
        
        return self.get_game_state()
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state as Python dict"""
        if self.game_state is None:
            raise ValueError("No game created. Call create_new_game() first.")
        
        # Get planets
        planets = []
        for planet in self.game_state.planets:
            planet_dict = {
                'id': planet.id,
                'owner': self._convert_player_to_int(planet.owner),
                'numShips': float(planet.n_ships),
                'growthRate': float(planet.growth_rate),
                'x': float(planet.position.x),
                'y': float(planet.position.y),
                'radius': float(planet.radius),
                'transporter': self._convert_transporter_to_dict(planet.transporter) if planet.transporter else None
            }
            planets.append(planet_dict)
        
        # Get transporters in transit (currently planets store their own transporters)
        transporters = []
        for planet in self.game_state.planets:
            if planet.transporter:
                transporter_dict = self._convert_transporter_to_dict(planet.transporter)
                transporters.append(transporter_dict)
        
        # Get game status
        return {
            'planets': planets,
            'transporters': transporters,
            'tick': int(self.game_state.game_tick),
            'isTerminal': bool(self.forward_model.is_terminal()),
            'statusString': str(self.forward_model.status_string()),
            'leader': self._convert_player_to_int(self.forward_model.get_leader()),
            'player1Ships': float(self.forward_model.get_ships(Player.Player1)),
            'player2Ships': float(self.forward_model.get_ships(Player.Player2))
        }

    def _convert_player_to_int(self, player: Player) -> int:
        """Convert Player to integer ID"""
        if player == Player.Player1:
            return 1
        elif player == Player.Player2:
            return 2
        else:
            return 0  # Neutral
    
    def _convert_transporter_to_dict(self, transporter) -> Optional[Dict[str, Any]]:
        """Convert Transporter to Python dict"""
        if transporter is None:
            return None
        
        return {
            'owner': self._convert_player_to_int(transporter.owner),
            'sourceIndex': int(transporter.source_index),
            'destinationIndex': int(transporter.destination_index),
            'numShips': float(transporter.n_ships),
            'x': float(transporter.s.x),
            'y': float(transporter.s.y),
            'vx': float(transporter.v.x),
            'vy': float(transporter.v.y)
        }
    
    def is_terminal(self) -> bool:
        """Check if game is terminal"""
        if self.forward_model is None:
            return True
        return bool(self.forward_model.is_terminal())
    
    def get_leader(self) -> Player:
        """Get current leader"""
        if self.forward_model is None:
            return Player.Neutral
        
        return self.forward_model.get_leader()
    
    def get_ships(self, player: Player) -> float:
        """Get number of ships for a player"""
        if self.forward_model is None:
            return 0.0
        
        return float(self.forward_model.get_ships(player))
    
    def get_player_conquers(self) -> Dict[Player, Dict[str, int]]:
        """Get the number of planets conquered by each player"""
        if self.forward_model is None:
            return {}
        
        return self.forward_model.player_conquers
    
    def status_string(self) -> str:
        """Get status string"""
        if self.forward_model is None:
            return "No game"
        return str(self.forward_model.status_string())
    
    def reset(self) -> Dict[str, Any]:
        """Reset the game to initial state"""
        return self.create_new_game()
    
    def clone_state(self) -> Dict[str, Any]:
        """Create a copy of the current game state for rollouts"""
        if self.game_state is None:
            return {}
        
        return self.get_game_state()
    
    def step_copy(self, actions: Dict[Player, Action]) -> Tuple[Dict[str, Any], 'PythonForwardModelBridge']:
        """Step a copy of the current state (useful for tree search)"""
        # Create a new bridge
        new_bridge = PythonForwardModelBridge()
        
        # Copy the current state
        if self.game_state is not None and self.game_params is not None:
            new_bridge.game_state = self.game_state.model_copy(deep=True)
            new_bridge.game_params = self.game_params.model_copy(deep=True)
            new_bridge.forward_model = ForwardModel(new_bridge.game_state, new_bridge.game_params)
            
            # Step the new bridge
            result = new_bridge.step(actions)
            return result, new_bridge
        else:
            raise ValueError("No game state to copy")
    
    def cleanup(self):
        """Clean up resources (no-op for Python implementation)"""
        pass


# Example usage and testing
if __name__ == "__main__":
    import time
    from agents.random_agents import CarefulRandomAgent, PureRandomAgent
    
    # Test the bridge
    print("Creating Python ForwardModel bridge...")
    bridge = PythonForwardModelBridge()
    
    # Create new game
    print("Creating new game...")
    game_params = {
        'maxTicks': 1000,
        'transporterSpeed': 1.0,
        'numPlanets': 10
    }
    initial_state = bridge.create_new_game(game_params=game_params)
    
    print(f"Initial game state: {len(initial_state['planets'])} planets")
    print(f"Game tick: {initial_state['tick']}")
    print(f"Terminal: {initial_state['isTerminal']}")
    print(f"Status: {initial_state['statusString']}")
    
    # Print planet info
    for i, planet in enumerate(initial_state['planets'][:5]):  # Show first 5 planets
        owner_name = {0: "Neutral", 1: "Player1", 2: "Player2"}[planet['owner']]
        print(f"Planet {i}: {owner_name}, Ships: {planet['numShips']:.1f}, Growth: {planet['growthRate']:.3f}")
    
    # Create some test agents
    agent1 = CarefulRandomAgent()
    agent2 = PureRandomAgent()
    agent1.prepare_to_play_as(Player.Player1, params=bridge._create_game_params_from_dict(game_params))
    agent2.prepare_to_play_as(Player.Player2, params=bridge._create_game_params_from_dict(game_params))

    # Simulate some steps
    print("\nSimulating game steps...")
    current_state = initial_state
    start_time = time.time()
    
    for step in range(1000):
        # Create some actions based on current state
        actions = {}
        
        # Convert state back to GameState for agents
        game_state = bridge.game_state
        
        # Get actions from agents
        action1 = agent1.get_action(game_state)
        action2 = agent2.get_action(game_state)
        
        actions = {
            Player.Player1: action1,
            Player.Player2: action2
        }
        
        # Step the game
        current_state = bridge.step(actions)
        
        if step % 10 == 0:  # Print every 10 steps
            print(f"Step {step + 1}: Tick {current_state['tick']}")
            print(f"  Player 1 ships: {current_state['player1Ships']:.1f}")
            print(f"  Player 2 ships: {current_state['player2Ships']:.1f}")
            print(f"  Leader: {current_state['leader']}")
            print(f"  Terminal: {current_state['isTerminal']}")
        
        if current_state['isTerminal']:
            print(f"\nGame finished at step {step + 1}!")
            break
    
    # Test performance
    end_time = time.time()
    steps_per_second = 1000 / (end_time - start_time)
    print(f"Performance: {steps_per_second:.1f} steps/second")
    
    # Test step_copy functionality
    print("\nTesting step_copy functionality...")
    original_state = bridge.get_game_state()
    
    # Create a test action
    test_actions = {
        Player.Player1: Action.do_nothing(),
        Player.Player2: Action.do_nothing()
    }
    
    # Step a copy
    copied_state, copied_bridge = bridge.step_copy(test_actions)
    
    # Original should be unchanged
    print(f"Original tick: {bridge.get_game_state()['tick']}")
    print(f"Copied tick: {copied_state['tick']}")
    
    bridge.cleanup()
    copied_bridge.cleanup()
    print("Done!")