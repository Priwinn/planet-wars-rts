import random
from typing import Optional, Dict, Any
from agents.ppo import Args 
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

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

        self.initial_game_state = None
        self.edge_attr = None
        
    
    def get_action(self, game_state: GameState) -> Action:

        x = self._get_observation(game_state)
        x.edge_index, x.edge_attr = add_self_loops(x.edge_index, x.edge_attr, fill_value='mean')


        action = self.model.get_action(x)
        if action[0] == 0:
            # No-op action, return None
            return Action.do_nothing()
        else:
            return Action(
                player_id=self.player,
                source_planet_id=action[0]-1,
                destination_planet_id=action[1],
                num_ships=action[2] * game_state.planets[action[0].int()-1].n_ships,
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
            planet.n_ships/10,  # Number of ships
            10*planet.growth_rate,  # Growth rate
            1.0 if planet.transporter is not None else 0.0  # Has transporter
        ])
        return features

    def _get_transporter_features(self, planet: Planet, game_state: GameState) -> torch.Tensor:
        """Calculate edge features between two planets. Weight is normalized by game width/height and transporter speed."""
        if planet.transporter is not None:
            target_planet = self._get_planet_by_id(planet.transporter.destination_index, game_state=game_state)
            distance = np.sqrt((target_planet.position.x - planet.transporter.s.x)**2 + (target_planet.position.y - planet.transporter.s.y)**2)- target_planet.radius
            weight = 10*self.params.transporter_speed / (distance + 1e-8)
            return torch.FloatTensor([self.player_to_int(planet.transporter.owner),
                                       planet.transporter.n_ships/10,
                                       weight])
        else:
            raise ValueError("Planet does not have a transporter")

    def _get_default_edge_features(self,i,j,game_state: GameState) -> np.ndarray:
        """Get default edge features for planets without transporters in use"""
        planet_i = self._get_planet_by_id(i, game_state=game_state)
        planet_j = self._get_planet_by_id(j, game_state=game_state)
        distance = np.sqrt((planet_i.position.x - planet_j.position.x) ** 2 + (planet_i.position.y - planet_j.position.y) ** 2) - planet_j.radius
        weight = 10*self.params.transporter_speed / (distance + 1e-8)
        return np.array([0.0,0.0, weight], dtype=np.float32)
    def _get_planet_by_id(self, planet_id: int, game_state: GameState) -> Dict[str, Any]:
        """Get planet data by ID"""
        for planet in game_state.planets:
            if planet.id == planet_id:
                return planet
        raise ValueError(f"Planet with ID {planet_id} not found")
    def _get_edge_index(self,i,j) -> int:
        """Get edge index for graph representation. Considers no self-loops are present."""
        if j>i:
            return i * (self.params.num_planets-1) + j-1
        elif j<i:
            return i * (self.params.num_planets-1) + j
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
        self.edge_index = torch.Tensor([[i, j] for i in range(params.num_planets) for j in range(params.num_planets) if i != j]).long().permute(1, 0)
        return super().prepare_to_play_as(player, params, opponent)
        
    


if __name__ == "__main__":
    agent = TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models\\PlanetWarsForwardModelGNN__ppo__greedy__adj_False__1__1750389797_final.pt")  
    agent.prepare_to_play_as(Player.Player1, GameParams())
    game_state = GameStateFactory(GameParams(num_planets=20)).create_game()
    action = agent.get_action(game_state)
    print(action)
