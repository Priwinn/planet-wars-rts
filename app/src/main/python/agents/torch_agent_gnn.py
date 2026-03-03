import random
from typing import Optional, Dict, Any
from config_files.ppo_config import Args 
import numpy as np
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from agents.planet_wars_agent import DEFAULT_OPPONENT, PlanetWarsPlayer
from agents.gnn import PlanetWarsAgentGNN
from core.game_state import GameState, Action, Player, GameParams, Planet, Transporter
from core.game_state_factory import GameStateFactory
from core.forward_model_extended_metrics import ForwardModelWithMetrics as ForwardModel
from gym_utils.PythonForwardModelBridge import PythonForwardModelBridge
from core.forward_model import ForwardModelDict
import time

from gym_utils.gnn_utils import preprocess_graph_data


class TorchAgentGNN(PlanetWarsPlayer):
    def __init__(self, model_class=None, weights_path: Optional[str] = None, model = None, device: Optional[torch.device] = None, use_topk_q: bool = False, topk_k: int = 4, opponent_policy: Optional[callable] = None, temperatures: dict = {'source': 1.0, 'target': 1.0, 'ratio': 1.0}, exploit = True):
        super().__init__()
        self.model_class = model_class
        self.weights_path = weights_path
        self.temperatures = temperatures
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')
        if model is not None:
            self.model = model.to(self.device)
            self.model.exploit = exploit
        else:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.model = model_class(state_dict['args']).to(self.device)
            is_compiled_weights = any('_orig_mod.' in key for key in state_dict['model_state_dict'].keys())
            if is_compiled_weights:
                self.model = torch.compile(self.model, dynamic=True)
            self.model.load_state_dict(state_dict['model_state_dict']) if model_class and weights_path else None
            self.model.exploit = exploit

        self.initial_game_state = None
        self.edge_attr = None
        self.use_topk_q = use_topk_q
        self.topk_k = topk_k
        self.opponent_policy = opponent_policy
        self.previous_score = None
        self.planet_distance_cache = None
        
    
    def get_action(self, game_state: GameState) -> Action:

        # start = time.time()
        x = self._get_observation(game_state)
        # time1 = time.time()
        # print(f"Time to get observation: {time1 - start:.4f} seconds", flush=True)
        x, source_mask = preprocess_graph_data([x], self.player_id, use_tick=self.model.args.use_tick, return_mask=True)
        # time2 = time.time()
        # print(f"Time to preprocess graph data: {time2 - time1:.4f} seconds", flush=True)

        if self.use_topk_q and self.topk_k > 1:
            action = self._get_action_topk_q(x.to(self.device), source_mask.to(self.device), game_state, self.temperatures)
        else:
            action = self.model.get_action(x.to(self.device), source_mask=source_mask.to(self.device))
        # time3 = time.time()
        # print(f"Time for model to get action: {time3 - time2:.4f} seconds", flush=True)
        if action[0] == 0:
            # No-op action, return None
            return Action.do_nothing()
        else:
            proposed_action = Action(
                player_id=self.player,
                source_planet_id=action[0]-1,
                destination_planet_id=action[1],
                num_ships=action[2] * game_state.planets[action[0].int()-1].n_ships)
            final_action = self._override_non_reaching_action(game_state, proposed_action)
            return final_action

    def get_agent_type(self) -> str:
        return "TorchAgent {weights_path} topq{use_topk_q} {topk_k} ex{exploit} T{temperatures} opp{opponent_policy}".format(
            model_class=self.model_class.__name__ if self.model_class else "None",
            weights_path=self.weights_path if self.weights_path else "None",
            use_topk_q=self.use_topk_q,
            topk_k=self.topk_k,
            exploit=self.model.exploit if hasattr(self.model, 'exploit') else False,
            temperatures=self.temperatures,
            opponent_policy=self.opponent_policy.__class__.__name__ if self.opponent_policy else "None"
        )

    def set_opponent_policy(self, opponent_policy: Optional[callable]) -> None:
        self.opponent_policy = opponent_policy
        if isinstance(opponent_policy, PlanetWarsPlayer) and self.player != Player.Neutral:
            opponent_policy.prepare_to_play_as(player=self.player.opponent(), params=self.params)

    def _get_observation(self, game_state: GameState) -> Data:
        if self.initial_game_state is None:
            self.initial_game_state = game_state
            self.edge_attr = torch.Tensor(self._get_default_edge_features_efficient(game_state))
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

    def _get_observation_dict(self, obs: Dict[str, Any]) -> Data:
        """Build graph `Data` from a dict-format observation (same fields as GameState).
        This mirrors _get_observation but accesses elements via dict indexing.
        """
        # Initialize edge attributes if needed
        if self.initial_game_state is None:
            self.initial_game_state = obs
            self.edge_attr = torch.Tensor(self._get_default_edge_features_efficient_dict(obs))

        planets = obs['planets']
        node_features = torch.Tensor(np.stack([self._planet_features_from_dict(p) for p in planets], axis=0))

        edge_features = self.edge_attr.detach().clone()
        planets_with_transporters = [p for p in planets if 'transporter' in p and p['transporter'] is not None]
        for p in planets_with_transporters:
            src_id = p['id']
            dst_id = p['transporter']['destination_index']
            edge_features[self._get_edge_index(src_id, dst_id)] = self._transporter_features_from_dict(p, obs)

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

    def _planet_features_from_dict(self, planet: Dict[str, Any]) -> np.ndarray:
        """Extract planet features from planet dict."""
        owner = planet['owner']
        owner_val = 0 if owner == Player.Neutral else 1 if owner == Player.Player1 else 2
        n_ships = planet['n_ships'] / 10
        growth = 10 * planet['growth_rate']
        has_transporter = 1.0 if planet['transporter'] is not None else 0.0
        return np.asarray([owner_val, n_ships, growth, has_transporter])
    
    def _override_non_reaching_action(self, game_state: GameState, proposed_action: Action) -> Action:
        """Override actions that cannot reach the target in time."""
        distance = self.planet_distance_cache[proposed_action.source_planet_id, proposed_action.destination_planet_id]
        if distance / self.params.transporter_speed > (self.params.max_ticks - game_state.game_tick):
            return Action.do_nothing()
        return proposed_action

    def _get_action_topk_q(self, graph_batch: Data, source_mask: torch.Tensor, game_state: GameState, temperatures: dict) -> torch.Tensor:
        actions = self.model.get_action_samples(graph_batch, source_mask=source_mask, num_samples=self.topk_k, temperatures=temperatures)
        if actions.dim() == 1:
            return actions
        q_values = self._evaluate_actions_q(actions, game_state)
        best_idx = torch.argmax(q_values).item()
        return actions[best_idx]

    def _evaluate_actions_q(self, actions: torch.Tensor, game_state: GameState) -> torch.Tensor:
        base_bridge = self._build_bridge_from_state(game_state)
        opponent_action = self._get_opponent_action(game_state)
        q_values = []
        gamma = getattr(self.model.args, "gamma", 1.0)
        states = []

        for action_tensor in actions:
            action = self._tensor_to_action(action_tensor, game_state)
            actions_dict = {
                self.player: action,
                self.player.opponent(): opponent_action
            }
            # next_state, copied_bridge = base_bridge.step_copy(actions_dict)
            # states.append(copied_bridge.game_state)
            gs = base_bridge.step_copy_dict(actions_dict)
            states.append(gs)
            #We probably don't need this, current actions do not change the reward until many staes later.
            # reward = self._calculate_change_in_score_delta(next_state)
            #We just add the no op penalty since is the only action with instant effects
            reward = -self.model.args.noop_penalty if action.source_planet_id == -1 else 0.0
            q_values.append(reward * gamma)
        # x = [self._get_observation(state) for state in states] 
        x = [self._get_observation_dict(state) for state in states]

        x = preprocess_graph_data(x, self.player_id, use_tick=self.model.args.use_tick, return_mask=False)
        with torch.no_grad():
            v_values = self.model.get_value(x.to(self.device)).cpu().numpy()
        q_values = np.array(q_values) + v_values.squeeze(-1)

        return torch.tensor(q_values, device=actions.device, dtype=torch.float32)

    def _build_bridge_from_state(self, game_state: GameState) -> PythonForwardModelBridge:
        bridge = PythonForwardModelBridge()
        bridge.game_state = game_state.model_dump()
        bridge.game_params = self.params
        bridge.forward_model = ForwardModelDict(bridge.game_state, bridge.game_params)
        return bridge

    def _get_opponent_action(self, game_state: GameState) -> Action:
        if self.opponent_policy is None:
            return Action.do_nothing()
        if isinstance(self.opponent_policy, PlanetWarsPlayer):
            return self.opponent_policy.get_action(game_state)
        raise TypeError("opponent_policy must be a PlanetWarsPlayer ")

    def _tensor_to_action(self, action_tensor: torch.Tensor, game_state: GameState) -> Action:
        source_planet = int(action_tensor[0].item())
        target_planet = int(action_tensor[1].item())
        ship_ratio = float(action_tensor[2].item())
        if source_planet == 0:
            return Action.do_nothing()
        source_planet_id = source_planet - 1
        if source_planet_id < 0 or source_planet_id >= len(game_state.planets):
            return Action.do_nothing()
        ship_ratio = max(0.0, min(1.0, ship_ratio))
        num_ships = ship_ratio * game_state.planets[source_planet_id].n_ships
        return Action(
            player_id=self.player,
            source_planet_id=source_planet_id,
            destination_planet_id=target_planet,
            num_ships=num_ships
        )


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

    def _transporter_features_from_dict(self, planet: Dict[str, Any], obs: Dict[str, Any]) -> torch.Tensor:
        """Calculate transporter edge features from planet dict and full observation dict."""
        if 'transporter' in planet and planet['transporter'] is not None:
            dest_idx = planet['transporter']['destination_index']
            target_planet = self._get_planet_by_id_from_dict(dest_idx, obs)
            sx = planet['transporter']['s']['x']
            sy = planet['transporter']['s']['y']
            distance = np.sqrt((target_planet['position']['x'] - sx) ** 2 + (target_planet['position']['y'] - sy) ** 2) - target_planet['radius']
            weight = 10 * self.params.transporter_speed / (distance + 1e-8)
            owner_val = self.player_to_int(planet['transporter']['owner'])
            return torch.FloatTensor([owner_val, planet['transporter']['n_ships'] / 10, weight])
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

    def _get_planet_by_id_from_dict(self, planet_id: int, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Get planet dict by ID from dict-format observation."""
        for p in obs['planets']:
            if p['id'] == planet_id:
                return p
        raise ValueError(f"Planet with ID {planet_id} not found")
    
    def _get_default_edge_features_efficient(self,game_state: GameState) -> np.ndarray:
        """Get default edge features for planets without transporters in use. More efficient version that precomputes all distances."""
        planets = game_state.planets
        num_planets = len(planets)
        edge_features = np.zeros((num_planets * (num_planets - 1), 3), dtype=np.float32)
        self.planet_distance_cache = np.zeros((num_planets, num_planets), dtype=np.float32)
        for planet_i in planets:
            for planet_j in planets:
                if planet_i.id != planet_j.id:
                    self.planet_distance_cache[planet_i.id, planet_j.id] = np.sqrt((planet_i.position.x - planet_j.position.x) ** 2 + (planet_i.position.y - planet_j.position.y) ** 2) - planet_j.radius
                    distance = self.planet_distance_cache[planet_i.id, planet_j.id]
                    weight = 10*self.params.transporter_speed / (distance + 1e-8)
                    edge_features[self._get_edge_index(planet_i.id, planet_j.id)] = np.array([0.0, 0.0, weight], dtype=np.float32)
        return edge_features

    def _get_default_edge_features_efficient_dict(self, obs: Dict[str, Any]) -> np.ndarray:
        """Efficient default edge features calculation for dict-format observation."""
        planets = obs['planets']
        num_planets = len(planets)
        edge_features = np.zeros((num_planets * (num_planets - 1), 3), dtype=np.float32)
        self.planet_distance_cache = np.zeros((num_planets, num_planets), dtype=np.float32)
        for pi in planets:
            for pj in planets:
                if pi['id'] != pj['id']:
                    dist = np.sqrt((pi['position']['x'] - pj['position']['x']) ** 2 + (pi['position']['y'] - pj['position']['y']) ** 2) - pj['radius']
                    self.planet_distance_cache[pi['id'], pj['id']] = dist
                    weight = 10 * self.params.transporter_speed / (dist + 1e-8)
                    edge_features[self._get_edge_index(pi['id'], pj['id'])] = np.array([0.0, 0.0, weight], dtype=np.float32)
        return edge_features

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
        params.num_planets = params.num_planets - params.num_planets % 2  # Ensure even number of planets
        self.edge_index = torch.Tensor([[i, j] for i in range(params.num_planets) for j in range(params.num_planets) if i != j]).long().permute(1, 0)
        self.initial_game_state = None
        self.planet_distance_cache = None
        if isinstance(self.opponent_policy, PlanetWarsPlayer):
            self.opponent_policy.prepare_to_play_as(player=player.opponent(), params=params)
        return super().prepare_to_play_as(player=player, params=params, opponent=opponent)
        
    

if __name__ == "__main__":
    import cProfile
    import pstats
    agent_opp = TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_gamma_999_v0.pt", use_topk_q=False)
    agent = TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_gamma_999_v0.pt", use_topk_q=True, topk_k=16, opponent_policy=agent_opp, exploit=False)  
    agent.prepare_to_play_as(Player.Player1, GameParams(num_planets=20))
    game_state = GameStateFactory(GameParams(num_planets=20)).create_game()
    for i in range(10):
        action = agent.get_action(game_state)


    pr = cProfile.Profile()
    pr.enable()

    for i in range(1000):
        action = agent.get_action(game_state)


    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(20)

