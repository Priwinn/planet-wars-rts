import random
from typing import Optional, Dict, Any
from agents.ppo import Args 
import numpy as np

import torch
from torch_geometric.data import Data

from agents.planet_wars_agent import DEFAULT_OPPONENT, PlanetWarsPlayer
from agents.mlp import PlanetWarsAgentMLP
from agents.gnn import PlanetWarsAgentGNN
from core.game_state import GameState, Action, Player, GameParams, Planet, Transporter
from core.game_state_factory import GameStateFactory
import time


class TorchAgentGNN(PlanetWarsPlayer):
    def __init__(self, model_class=None, weights_path: Optional[str] = None):
        super().__init__()
        self.model_class = model_class
        self.weights_path = weights_path
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        self.model = model_class(state_dict['args'])
        self.model.load_state_dict(state_dict['model_state_dict']) if model_class and weights_path else None
        self.game_params = {
            'maxTicks': 20000,
            'numPlanets': 20,
            'transporterSpeed': 3.0,
            'width': 640,
            'height': 480
        }
        self.edge_index = torch.Tensor([[i, j] for i in range(self.game_params['numPlanets']) for j in range(self.game_params['numPlanets']) if i != j]).long().permute(1, 0)
        self.initial_game_state = None
        self.edge_attr = None
        
    
    def get_action(self, game_state: GameState) -> Action:

        x = self._get_observation(game_state)# Add batch dimension

        action = self.model.get_action(x)

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

    def _get_observation(self, game_state: GameState) -> Data:
        if self.initial_game_state is None:
            self.initial_game_state = game_state
            self.edge_attr = torch.Tensor(np.stack(
                [self._get_default_edge_features(edge[0], edge[1], self.initial_game_state) for edge in self.edge_index.permute(1, 0).numpy()]
            ))
        planets = game_state.planets
        node_features = torch.Tensor(np.stack([self._get_planet_features(p) for p in planets], axis=0))

        edge_features = self.edge_attr.detach().clone()
        planets_with_transporters = [p for p in planets if p.transporter is not None]
        for p in planets_with_transporters:
            edge_features[self._get_edge_index(p.id, p.transporter.destination_index)] = self._get_transporter_features(p, game_state)
        #One-hot encode transporter owner
        # transporter_owners = self._owner_one_hot_encoding(edge_features[:, 0].long())
        # edge_features = torch.cat((transporter_owners, edge_features[:, 1:]), dim=1)
        return Data(
            x=node_features,
            edge_index=self.edge_index,
            edge_attr=edge_features
        )




    def _get_planet_features(self, planet: Planet) -> np.ndarray:
        """Extract features from a single planet for GNN"""
        features = np.asarray([
            0 if planet.owner == Player.Neutral else 1 if planet.owner == Player.Player1 else 2,  # Owner ID
            planet.n_ships,  # Number of ships
            planet.growth_rate,  # Growth rate
        ])
        return features

    def _get_transporter_features(self, planet: Planet, game_state: GameState) -> torch.Tensor:
        """Calculate edge features between two planets. Weight is normalized by game width/height and transporter speed."""
        if planet.transporter is not None:
            target_planet = self._get_planet_by_id(planet.transporter.destination_index, game_state=game_state)
            distance = np.sqrt((target_planet.position.x - planet.transporter.s.x)**2 + (target_planet.position.y - planet.transporter.s.y)**2)
            weight = self.game_params['transporterSpeed'] / (distance + 1e-8)
            return torch.FloatTensor([self.player_to_int(planet.transporter.owner),
                                       planet.transporter.n_ships,
                                       weight])
        else:
            raise ValueError("Planet does not have a transporter")

    def _get_default_edge_features(self,i,j,game_state: GameState) -> np.ndarray:
        """Get default edge features for planets without transporters in use"""
        weight = self.game_params['transporterSpeed'] / (
            np.sqrt((self._get_planet_by_id(i, game_state=game_state).position.x - self._get_planet_by_id(j, game_state=game_state).position.x) ** 2 + (self._get_planet_by_id(i, game_state=game_state).position.y - self._get_planet_by_id(j, game_state=game_state).position.y) ** 2) + 1e-8)
        return np.array([0.0,0.0, weight], dtype=np.float32)
    def _get_planet_by_id(self, planet_id: int, game_state: GameState) -> Dict[str, Any]:
        """Get planet data by ID"""
        for planet in game_state.planets:
            if planet.id == planet_id:
                return planet
        return None
    def _get_edge_index(self,i,j) -> int:
        """Get edge index for graph representation. Considers no self-loops are present."""
        if j>i:
            return i * (self.game_params['numPlanets']-1) + j-1
        elif j<i:
            return i * (self.game_params['numPlanets']-1) + j
        else:
            raise ValueError("No self-loops allowed")
        
    def player_to_int(self, player: Player) -> int:
        """Convert Player enum to integer"""
        if player == Player.Neutral:
            return 0
        elif player == Player.Player1:
            return 1
        elif player == Player.Player2:
            return 2
        else:
            raise ValueError(f"Unknown player: {player}")
        
    def prepare_to_play_as(self, player: Player, params: GameParams, opponent: str | None = ...) -> str:
        self.model.player_id = 1 if player == Player.Player1 else 2
        self.player_id = self.model.player_id
        return super().prepare_to_play_as(player, params, opponent)
        
    


if __name__ == "__main__":
    agent = TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models\\PlanetWarsForwardModelGNN__ppo__greedy__adj_False__1__1750389797_final.pt")  
    agent.prepare_to_play_as(Player.Player1, GameParams())
    game_state = GameStateFactory(GameParams(num_planets=20)).create_game()
    action = agent.get_action(game_state)
    print(action)
