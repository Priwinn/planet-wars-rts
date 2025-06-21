import random
from typing import Optional, Dict, Any
from agents.ppo import Args 
import numpy as np

import torch

from agents.planet_wars_agent import DEFAULT_OPPONENT, PlanetWarsPlayer
from agents.mlp import PlanetWarsAgentMLP
from core.game_state import GameState, Action, Player, GameParams
from core.game_state_factory import GameStateFactory


class TorchAgent(PlanetWarsPlayer):
    def __init__(self, model_class=None, weights_path: Optional[str] = None):
        super().__init__()
        self.model_class = model_class
        self.weights_path = weights_path
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        self.model = model_class(state_dict['args'])
        self.model.load_state_dict(state_dict['model_state_dict']) if model_class and weights_path else None
        self.game_params = {
            'maxTicks': 2000,
            'numPlanets': 20,
            'transporterSpeed': 3.0,
            'width': 640,
            'height': 480
        }
    def get_action(self, game_state: GameState) -> Action:
        x = self.game_state_to_dict(game_state)
        x = self.dict_to_tensor(x).unsqueeze(0)  # Add batch dimension

        action = self.model.get_action(x)
        if self.player_id == 2:
            half = self.params.num_planets // 2
            action[0] = (action[0] + half) % self.params.num_planets  # Adjust source planet index
            action[1] = (action[1] + half) % self.params.num_planets  # Adjust destination planet index

        return Action(
            player_id=self.player,
            source_planet_id=action[0],
            destination_planet_id=action[1],
            num_ships=action[2] * game_state.planets[action[0].int()].n_ships,
        )

    def get_agent_type(self) -> str:
        return "TorchAgent {model_class} {weights_path}".format(
            model_class=self.model_class.__name__ if self.model_class else "None",
            weights_path=self.weights_path if self.weights_path else "None"
        )
    
    def dict_to_tensor(self, planets: Dict[str, Any]) -> torch.Tensor:
        """Convert a dictionary of planets to a tensor suitable for the model"""
        features = []
        for planet in planets:
            planet_features = self._get_planet_features(planet)
            features.append(planet_features)
        if self.player_id == 2:
            # Swap first half of planets with second half 
            half = len(features) // 2
            features = features[half:] + features[:half]
        
        # Stack all planet features into a single tensor
        return torch.tensor(np.stack(features), dtype=torch.float32)
    
    def game_state_to_dict(self, game_state: GameState):
        # Convert the game state to a tensor suitable for the model
        planets = []
        planets_list = game_state.planets
        for planet in planets_list:
            planet_dict = {
                'id': planet.id,
                'owner': 0 if planet.owner == Player.Neutral else 1 if planet.owner == Player.Player1 else 2,  # Neutral, Player1, Player2
                'numShips': float(planet.n_ships),
                'growthRate': float(planet.growth_rate),
                'x': float(planet.position.x),
                'y': float(planet.position.y),
                'radius': float(planet.radius),
                'transporter': self._convert_transporter_to_dict(planet.transporter) if planet.transporter else None
            }
            planets.append(planet_dict)


        return planets
    
    def _convert_transporter_to_dict(self, transporter) -> Optional[Dict[str, Any]]:
        if transporter is None:
            return None
        
        return {
            'owner': 0 if transporter.owner == Player.Neutral else 1 if transporter.owner == Player.Player1 else 2,  # Neutral, Player1, Player2
            'sourceIndex': int(transporter.source_index),
            'destinationIndex': int(transporter.destination_index),
            'numShips': float(transporter.n_ships),
            'x': float(transporter.s.x),
            'y': float(transporter.s.y),
            'vx': float(transporter.v.x),
            'vy': float(transporter.v.y)
        }
    def _get_planet_features(self, planet: Dict[str, Any]) -> np.ndarray:
        """Extract features from a single planet. Model is only trained as playerr 1 so we mirror the features for player 2."""
        features = [
            planet['owner'],  # Owner ID
            planet['numShips'],  # Number of ships
            planet['growthRate'],  # Growth rate
            planet['x'] / self.game_params['width'] if self.player_id == 1 else 1 - (planet['x'] / self.game_params['width']),  # X coordinate
            planet['y'] / self.game_params['height'] if self.player_id == 1 else 1 - (planet['y'] / self.game_params['height'])  # Y coordinate
        ]

        
        # Add transporter info if available
        if planet.get('transporter'):
            transporter = planet['transporter']
            features.extend([
                transporter['owner'],
                # transporter['sourceIndex'],
                transporter['destinationIndex'], 
                # TODO: Consider symmetry when controlled player changes either:
                # 1.- Swap planet labels (if using onehot, obs vector is different size depending on num planets)
                # 2.- Use edge features for transporters
                transporter['numShips'],
                # Normalized transporter position
                transporter['x']*transporter['vx']/(self.game_params['width'] * self.game_params['transporterSpeed']) if self.player_id == 1 else
                (self.game_params['width'] - transporter['x'])*transporter['vx']/(self.game_params['width'] * self.game_params['transporterSpeed']),
                transporter['y']*transporter['vy']/(self.game_params['height'] * self.game_params['transporterSpeed']) if self.player_id == 1 else
                (self.game_params['height'] - transporter['y'])*transporter['vy']/(self.game_params['height'] * self.game_params['transporterSpeed']),
                # transporter['vx'],
                # transporter['vy']
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        return np.array(features, dtype=np.float32)
    
    def prepare_to_play_as(self, player: Player, params: GameParams, opponent: str | None = ...) -> str:
        self.model.player_id = 1 if player == Player.Player1 else 2
        self.player_id = self.model.player_id
        return super().prepare_to_play_as(player, params, opponent)
    


if __name__ == "__main__":
    agent = TorchAgent(model_class=PlanetWarsAgentMLP, weights_path="models/PlanetWarsForwardModel__ppo__greedy__adj_False__1__1750343129_final.pt")  
    agent.prepare_to_play_as(Player.Player1, GameParams())
    game_state = GameStateFactory(GameParams(num_planets=20)).create_game()
    action = agent.get_action(game_state)
    print(action)
