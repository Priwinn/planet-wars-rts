import jpype
import jpype.imports
from jpype.types import *
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import atexit

from torch import Tensor
from core.game_state import Player, Action




class KotlinForwardModelBridge:
    """Bridge to interact with Kotlin ForwardModel for local game simulation using JPype"""
    
    def __init__(self, jar_path: str = None, auto_start_jvm: bool = True):
        self.jar_path = jar_path or self._find_jar_path()
        self.forward_model = None
        self.game_state = None
        self.game_params = None
        self._jvm_started = False
        self.initial_game_state = None
        
        if auto_start_jvm:
            self.start_jvm()
        
        atexit.register(self.cleanup)
    
    def _find_jar_path(self) -> str:
        """Find the built JAR file"""
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
        print(f"Workspace root: {workspace_root}")
        jar_path = os.path.join(workspace_root, "app", "build", "libs", "app.jar")
        if not os.path.exists(jar_path):
            # Try alternative path
            jar_path = os.path.join(workspace_root, "app", "build", "libs", "planet-wars-rts.jar")
        return jar_path
    
    def start_jvm(self):
        """Start the JVM with the required JAR"""
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[self.jar_path])
            self._jvm_started = True
        
        # Import required Java classes
        GameState = jpype.JClass("games.planetwars.core.GameState")
        GameParams = jpype.JClass("games.planetwars.core.GameParams")
        ForwardModel = jpype.JClass("games.planetwars.core.ForwardModel")
        KotlinPlayer = jpype.JClass("games.planetwars.core.Player")
        GameStateFactory = jpype.JClass("games.planetwars.core.GameStateFactory")
        KotlinAction = jpype.JClass("games.planetwars.agents.Action")
        HashMap = jpype.JClass("java.util.HashMap")

        # Store references to Java classes
        self.GameState = GameState
        self.GameParams = GameParams
        self.ForwardModel = ForwardModel
        self.KotlinPlayer = KotlinPlayer
        self.GameStateFactory = GameStateFactory
        self.KotlinAction = KotlinAction
        self.HashMap = HashMap
    
    def create_new_game(self, game_params: Optional[Dict[str, Any]] = None):
        """Create a new game with ForwardModel"""
        if not self._jvm_started:
            self.start_jvm()
        
        # Create game parameters
        if game_params is None:
            self.game_params = self.GameParams()
        else:
            self.game_params = self._create_game_params_from_dict(game_params)
        
        # Create game state 
        if self.initial_game_state is not None and game_params.get('newMapEachRun', True) == False:
            # If we have an initial game state, use it
            self.game_state = self.initial_game_state.deepCopy()
        else:
            game_state_factory = self.GameStateFactory(self.game_params)
            self.game_state = game_state_factory.createGame()
            self.initial_game_state = self.game_state.deepCopy()  # Store initial state for cloning
        
        # Create forward model
        self.forward_model = self.ForwardModel(self.game_state, self.game_params)
        
        return self.get_game_state()
    
    def _create_game_params_from_dict(self, params_dict: Dict[str, Any]):
        """Create Kotlin GameParams from Python dict using positional arguments"""
        
        # Extract values with defaults matching Kotlin GameParams
        width = params_dict.get('width', 640)
        height = params_dict.get('height', 480)
        edge_separation = params_dict.get('edgeSeparation', 25.0)
        radial_separation = params_dict.get('radialSeparation', 1.5)
        growth_to_radius_factor = params_dict.get('growthToRadiusFactor', 200.0)
        num_planets = params_dict.get('numPlanets', 10)
        initial_neutral_ratio = params_dict.get('initialNeutralRatio', 0.5)
        max_ticks = params_dict.get('maxTicks', 2000)
        min_initial_ships_per_planet = params_dict.get('minInitialShipsPerPlanet', 2)
        max_initial_ships_per_planet = params_dict.get('maxInitialShipsPerPlanet', 20)
        min_growth_rate = params_dict.get('minGrowthRate', 0.02)
        max_growth_rate = params_dict.get('maxGrowthRate', 0.1)
        transporter_speed = params_dict.get('transporterSpeed', 3.0)
        new_map_each_run = params_dict.get('newMapEachRun', True)
        
        # Call constructor with all positional arguments in the correct order
        return self.GameParams(
            width,
            height,
            edge_separation,
            radial_separation,
            growth_to_radius_factor,
            num_planets,
            initial_neutral_ratio,
            max_ticks,
            min_initial_ships_per_planet,
            max_initial_ships_per_planet,
            min_growth_rate,
            max_growth_rate,
            transporter_speed,
            new_map_each_run
        )
    
    def step(self, actions: Dict[Player, Action]) -> Dict[str, Any]:
        """Step the forward model with given actions"""
        if self.forward_model is None:
            raise ValueError("No game created. Call create_new_game() first.")
        
        # Convert Python actions to Kotlin actions
        kotlin_actions = self._convert_actions_to_kotlin(actions)
        
        # Step the forward model
        self.forward_model.step(kotlin_actions)
        
        return self.get_game_state()
    
    def _convert_actions_to_kotlin(self, actions: Dict[Player, Action]):
        """Convert Python actions dict to Kotlin Map<Player, Action>"""
        kotlin_actions = self.HashMap()
        
        for player, action in actions.items():
            kotlin_player = self._convert_player_to_kotlin(player)
            kotlin_action = self._convert_action_to_kotlin(action)
            kotlin_actions.put(kotlin_player, kotlin_action)
        
        return kotlin_actions
    
    def _convert_player_to_kotlin(self, player: Player):
        """Convert Python Player to Kotlin Player"""
        if player == Player.Player1:
            return self.KotlinPlayer.Player1
        elif player == Player.Player2:
            return self.KotlinPlayer.Player2
        else:
            return self.KotlinPlayer.Neutral
    
    def _convert_action_to_kotlin(self, action: Action):
        """Convert Python Action to Kotlin Action"""
        if action == Action.DO_NOTHING:
            return self.kotlin_do_nothing()
        else:
            return self.KotlinAction(
                self._convert_player_to_kotlin(action.player_id),
                action.source_planet_id,
                action.destination_planet_id,
                float(action.num_ships)
            )
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state as Python dict"""
        if self.game_state is None:
            raise ValueError("No game created. Call create_new_game() first.")
        
        # Get planets
        planets = []
        planets_list = self.game_state.getPlanets()
        
        for i in range(len(planets_list)):
            planet = planets_list[i]
            planet_dict = {
                'id': planet.getId(),
                'owner': self._convert_kotlin_player_to_python(planet.getOwner()),
                'numShips': float(planet.getNShips()),
                'growthRate': float(planet.getGrowthRate()),
                'x': float(planet.getPosition().getX()),
                'y': float(planet.getPosition().getY()),
                'radius': float(planet.getRadius()),
                'transporter': self._convert_transporter_to_dict(planet.getTransporter()) if planet.getTransporter() else None
            }
            planets.append(planet_dict)
        
        # Get transporters in transit
        transporters = []
        if hasattr(self.game_state, 'transporters'):
            for transporter in self.game_state.transporters:
                transporters.append(self._convert_transporter_to_dict(transporter))
        
        # Get game status
        return {
            'planets': planets,
            'transporters': transporters,
            'tick': int(self.game_state.getGameTick()),
            'isTerminal': bool(self.forward_model.isTerminal()),
            'statusString': str(self.forward_model.statusString()),
            'leader': self._convert_kotlin_player_to_python(self.forward_model.getLeader()),
            'player1Ships': float(self.forward_model.getShips(self.KotlinPlayer.Player1)),
            'player2Ships': float(self.forward_model.getShips(self.KotlinPlayer.Player2))
        }
    
    def _convert_kotlin_player_to_python(self, kotlin_player) -> int:
        """Convert Kotlin Player to Python player ID"""
        if kotlin_player == self.KotlinPlayer.Player1:
            return 1  # Player1 -> 1
        elif kotlin_player == self.KotlinPlayer.Player2:
            return 2  # Player2 -> 2
        else:
            return 0  # Neutral -> 0

    def _convert_transporter_to_dict(self, transporter) -> Optional[Dict[str, Any]]:
        """Convert Kotlin Transporter to Python dict"""
        if transporter is None:
            return None
        
        return {
            'owner': self._convert_kotlin_player_to_python(transporter.getOwner()),
            'sourceIndex': int(transporter.getSourceIndex()),
            'destinationIndex': int(transporter.getDestinationIndex()),
            'numShips': float(transporter.getNShips()),
            'x': float(transporter.getS().getX()),
            'y': float(transporter.getS().getY()),
            'vx': float(transporter.getV().getX()),
            'vy': float(transporter.getV().getY())
        }
    
    def is_terminal(self) -> bool:
        """Check if game is terminal"""
        if self.forward_model is None:
            return True
        return bool(self.forward_model.isTerminal())
    
    def get_leader(self) -> Player:
        """Get current leader"""
        if self.forward_model is None:
            return Player.Neutral
        
        kotlin_leader = self.forward_model.getLeader()
        player_id = self._convert_kotlin_player_to_python(kotlin_leader)
        
        if player_id == 0:
            return Player.Player1
        elif player_id == 1:
            return Player.Player2
        else:
            return Player.Neutral
    
    def get_ships(self, player: Player) -> float:
        """Get number of ships for a player"""
        if self.forward_model is None:
            return 0.0
        
        kotlin_player = self._convert_player_to_kotlin(player)
        return float(self.forward_model.getShips(kotlin_player))
    
    def kotlin_do_nothing(self) -> Action:
        """Create a do-nothing action for Kotlin"""
        return self.KotlinAction(
            self.KotlinPlayer.Neutral,  # Neutral player
            -1,  # No source planet
            -1,  # No destination planet
            0.0  # No ships
        )
    
    def status_string(self) -> str:
        """Get status string"""
        if self.forward_model is None:
            return "No game"
        return str(self.forward_model.statusString())
    
    def reset(self) -> Dict[str, Any]:
        """Reset the game to initial state"""
        return self.create_new_game()
    
    def clone_state(self) -> Dict[str, Any]:
        """Create a copy of the current game state for rollouts"""
        if self.game_state is None:
            return {}
        
        # Note: This creates a shallow reference. For true cloning,
        # you'd need to implement a deep copy method in Kotlin
        return self.get_game_state()
    
    def step_copy(self, actions: Dict[Player, Action]) -> Tuple[Dict[str, Any], 'KotlinForwardModelBridge']:
        """Step a copy of the current state (useful for tree search)"""
        # Create a new bridge with the same parameters
        new_bridge = KotlinForwardModelBridge(self.jar_path, auto_start_jvm=False)
        new_bridge.start_jvm()
        
        # Clone the current state (simplified - you might need proper cloning in Kotlin)
        new_bridge.create_new_game()
        # Note: Proper state cloning would require additional Kotlin methods
        
        # Step the new bridge
        result = new_bridge.step(actions)
        return result, new_bridge
    
    def cleanup(self):
        """Clean up resources"""
        # JPype cleanup is handled automatically
        pass


# Example usage and testing
if __name__ == "__main__":
    import time
    # Test the bridge
    print("Creating Kotlin ForwardModel bridge...")
    bridge = KotlinForwardModelBridge()
    
    # Create new game
    print("Creating new game...")
    initial_state = bridge.create_new_game({
        'maxTicks': 1000,
        'transporterSpeed': 1.0
    })
    
    print(f"Initial game state: {len(initial_state['planets'])} planets")
    print(f"Game tick: {initial_state['tick']}")
    print(f"Terminal: {initial_state['isTerminal']}")
    print(f"Status: {initial_state['statusString']}")
    
    # Print planet info
    for i, planet in enumerate(initial_state['planets'][:5]):  # Show first 5 planets
        owner_name = {-1: "Neutral", 0: "Player1", 1: "Player2"}[planet['owner']]
        print(f"Planet {i}: {owner_name}, Ships: {planet['numShips']:.1f}, Growth: {planet['growthRate']:.1f}")
    
    # Simulate some steps
    print("\nSimulating game steps...")
    current_state = initial_state
    start_time = time.time()
    for step in range(20):
        # Create some actions based on current state
        actions = {}
        
        # Player 1 action
        player1_planets = [p for p in current_state['planets'] if p['owner'] == 1 and p['numShips'] > 1 and not p['transporter']]
        if player1_planets:
            source = max(player1_planets, key=lambda x: x['numShips'])
            target_candidates = [p for p in current_state['planets'] if p['owner'] != 1]
            if target_candidates:
                target = min(target_candidates, key=lambda x: (x['x'] - source['x'])**2 + (x['y'] - source['y'])**2)
                actions[Player.Player1] = Action(
                    player_id=Player.Player1,
                    source_planet_id=source['id'],
                    destination_planet_id=target['id'],
                    num_ships=source['numShips'] * 0.7
                )
        
        # Player 2 action
        player2_planets = [p for p in current_state['planets'] if p['owner'] == 2 and p['numShips'] > 1 and not p['transporter']]
        if player2_planets:
            source = max(player2_planets, key=lambda x: x['numShips'])
            target_candidates = [p for p in current_state['planets'] if p['owner'] != 2]
            if target_candidates:
                target = min(target_candidates, key=lambda x: (x['x'] - source['x'])**2 + (x['y'] - source['y'])**2)
                actions[Player.Player2] = Action(
                    player_id=Player.Player2,
                    source_planet_id=source['id'],
                    destination_planet_id=target['id'],
                    num_ships=source['numShips'] * 0.9
                )
        
        # Step the game
        current_state = bridge.step(actions)
        
        if step % 5 == 0:  # Print every 5 steps
            print(f"Step {step + 1}: Tick {current_state['tick']}")
            print(f"  Player 1 ships: {current_state['player1Ships']:.1f}")
            print(f"  Player 2 ships: {current_state['player2Ships']:.1f}")
            print(f"  Leader: {current_state['leader']}")
            print(f"  Terminal: {current_state['isTerminal']}")
        
        if current_state['isTerminal']:
            print(f"\nGame finished at step {step + 1}!")
            print(f"Final status: {current_state['statusString']}")
            break
    
    # Test performance
    end_time = time.time()
    steps_per_second = 20 / (end_time - start_time)
    print(f"Performance: {steps_per_second:.1f} steps/second")

    bridge.cleanup()
    print("Done!")