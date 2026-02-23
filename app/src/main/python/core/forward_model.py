from typing import Dict, Optional
from core.game_state import GameState, GameParams, Player, Action, Planet, Transporter, Vec2d


class ForwardModel:
    n_updates = 0
    n_failed_actions = 0
    n_actions = 0

    def __init__(self, state: GameState, params: GameParams):
        self.state = state
        self.params = params

    def step(self, actions: Dict[Player, Action]):
        self.apply_actions(actions)
        pending: Dict[int, Dict[Player, float]] = {}
        self.update_transporters(pending)
        self.update_planets(pending)
        ForwardModel.n_updates += 1
        self.state.game_tick += 1

    def apply_actions(self, actions: Dict[Player, Action]):
        for player, action in actions.items():
            if action == Action.DO_NOTHING:
                continue
            source = self.state.planets[action.source_planet_id]
            target = self.state.planets[action.destination_planet_id]
            if source.transporter is None and source.owner == player and source.n_ships >= action.num_ships:
                source.n_ships -= action.num_ships
                direction = (target.position - source.position).normalize()
                velocity = direction * self.params.transporter_speed
                transporter = Transporter(
                    s=source.position,
                    v=velocity,
                    owner=player,
                    source_index=action.source_planet_id,
                    destination_index=action.destination_planet_id,
                    n_ships=action.num_ships
                )
                source.transporter = transporter
                ForwardModel.n_actions += 1
            else:
                ForwardModel.n_failed_actions += 1

    def is_terminal(self) -> bool:
        if self.state.game_tick > self.params.max_ticks:
            return True
        return not any(p.owner == Player.Player1 for p in self.state.planets) or \
               not any(p.owner == Player.Player2 for p in self.state.planets)

    def status_string(self) -> str:
        return (
            f"Game tick: {self.state.game_tick}; "
            f"Player 1: {int(self.get_ships(Player.Player1))}; "
            f"Player 2: {int(self.get_ships(Player.Player2))}; "
            f"Leader: {self.get_leader().value}"
        )

    def get_ships(self, player: Player) -> float:
        return sum(p.n_ships for p in self.state.planets if p.owner == player)

    def get_leader(self) -> Player:
        s1 = self.get_ships(Player.Player1)
        s2 = self.get_ships(Player.Player2)
        if s1 == s2:
            return Player.Neutral
        return Player.Player1 if s1 > s2 else Player.Player2

    def transporter_arrival(self, destination: Planet, transporter: Transporter,
                            pending: Dict[int, Dict[Player, float]]):
        if destination.id not in pending:
            pending[destination.id] = {Player.Player1: 0.0, Player.Player2: 0.0}
        pending[destination.id][transporter.owner] += transporter.n_ships

    def update_transporters(self, pending: Dict[int, Dict[Player, float]]):
        for planet in self.state.planets:
            transporter = planet.transporter
            if transporter:
                destination = self.state.planets[transporter.destination_index]
                if transporter.s.distance(destination.position) < destination.radius:
                    self.transporter_arrival(destination, transporter, pending)
                    planet.transporter = None
                else:
                    transporter.s = transporter.s + transporter.v

    def update_neutral_planet(self, planet: Planet, pending: Optional[Dict[Player, float]]):
        if not pending:
            return
        p1 = pending.get(Player.Player1, 0.0)
        p2 = pending.get(Player.Player2, 0.0)
        net = p1 - p2
        planet.n_ships -= abs(net)
        if planet.n_ships < 0:
            planet.owner = Player.Player1 if net > 0 else Player.Player2
            planet.n_ships = -planet.n_ships

    def update_player_planet(self, planet: Planet, pending: Optional[Dict[Player, float]]):
        planet.n_ships += planet.growth_rate
        if not pending:
            return
        own_incoming = pending.get(planet.owner, 0.0)
        opp_incoming = pending.get(planet.owner.opponent(), 0.0)
        planet.n_ships += own_incoming - opp_incoming
        if planet.n_ships < 0:
            planet.owner = planet.owner.opponent()
            planet.n_ships = -planet.n_ships

    def update_planets(self, pending: Dict[int, Dict[Player, float]]):
        for planet in self.state.planets:
            p_pending = pending.get(planet.id)
            if planet.owner == Player.Neutral:
                self.update_neutral_planet(planet, p_pending)
            else:
                self.update_player_planet(planet, p_pending)


class ForwardModelDict:
    """Forward model that operates on dict-format game state (mirrors ForwardModel)."""
    n_updates = 0
    n_failed_actions = 0
    n_actions = 0

    def __init__(self, state: Dict, params: GameParams):
        # state is expected to be a dict with keys like 'planets' (list of planet dicts) and 'gameTick' or 'game_tick'
        self.state = state
        self.params = params

    # --- vector helpers for dicts ---
    @staticmethod
    def _vec_add(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
        return {"x": a['x'] + b['x'], "y": a['y'] + b['y']}

    @staticmethod
    def _vec_sub(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
        return {"x": a['x'] - b['x'], "y": a['y'] - b['y']}

    @staticmethod
    def _vec_mul(a: Dict[str, float], scalar: float) -> Dict[str, float]:
        return {"x": a['x'] * scalar, "y": a['y'] * scalar}

    @staticmethod
    def _vec_mag(a: Dict[str, float]) -> float:
        return (a['x'] ** 2 + a['y'] ** 2) ** 0.5

    @classmethod
    def _vec_distance(cls, a: Dict[str, float], b: Dict[str, float]) -> float:
        return cls._vec_mag(cls._vec_sub(a, b))

    @classmethod
    def _normalize(cls, a: Dict[str, float]) -> Dict[str, float]:
        mag = cls._vec_mag(a)
        return cls._vec_mul(a, 1.0 / mag) if mag > 0 else {"x": 0.0, "y": 0.0}

    # --- core methods mirroring ForwardModel ---
    def step(self, actions: Dict[Player, Action]):
        self.apply_actions(actions)
        pending: Dict[int, Dict[Player, float]] = {}
        self.update_transporters(pending)
        self.update_planets(pending)
        ForwardModelDict.n_updates += 1
        # support both snake and camel aliases
        if 'game_tick' in self.state:
            self.state['game_tick'] += 1
        else:
            self.state['gameTick'] = self.state.get('gameTick', 0) + 1

    def apply_actions(self, actions: Dict[Player, Action]):
        for player, action in actions.items():
            if action == Action.DO_NOTHING:
                continue
            source = self.state['planets'][action.source_planet_id]
            target = self.state['planets'][action.destination_planet_id]
            if source.get('transporter') is None and source['owner'] == player and source['n_ships'] >= action.num_ships:
                source['n_ships'] -= action.num_ships
                # compute direction and velocity using dict vectors
                direction = self._vec_sub(target['position'], source['position'])
                direction = self._normalize(direction)
                velocity = self._vec_mul(direction, self.params.transporter_speed)
                transporter = {
                    's': {'x': source['position']['x'], 'y': source['position']['y']},
                    'v': velocity,
                    'owner': player,
                    'source_index': action.source_planet_id,
                    'destination_index': action.destination_planet_id,
                    'n_ships': action.num_ships
                }
                source['transporter'] = transporter
                ForwardModelDict.n_actions += 1
            else:
                ForwardModelDict.n_failed_actions += 1

    def is_terminal(self) -> bool:
        gt = self.state.get('game_tick', self.state.get('gameTick', 0))
        if gt > self.params.max_ticks:
            return True
        planets = self.state['planets']
        if not any(p['owner'] == Player.Player1 for p in planets) or not any(p['owner'] == Player.Player2 for p in planets):
            return True
        return False

    def transporter_arrival(self, destination: Dict, transporter: Dict, pending: Dict[int, Dict[Player, float]]):
        dst_id = destination.get('id')
        if dst_id not in pending:
            pending[dst_id] = {Player.Player1: 0.0, Player.Player2: 0.0}
        pending[dst_id][transporter['owner']] += transporter['n_ships']

    def update_transporters(self, pending: Dict[int, Dict[Player, float]]):
        for planet in self.state['planets']:
            transporter = planet.get('transporter')
            if transporter:
                destination = self.state['planets'][transporter['destination_index']]
                if self._vec_distance(transporter['s'], destination['position']) < destination['radius']:
                    self.transporter_arrival(destination, transporter, pending)
                    planet['transporter'] = None
                else:
                    planet['transporter']['s'] = self._vec_add(transporter['s'], transporter['v'])

    def update_neutral_planet(self, planet: Dict, pending: Optional[Dict[Player, float]]):
        if not pending:
            return
        p1 = pending.get(Player.Player1, 0.0)
        p2 = pending.get(Player.Player2, 0.0)
        net = p1 - p2
        planet['n_ships'] -= abs(net)
        if planet['n_ships'] < 0:
            planet['owner'] = Player.Player1 if net > 0 else Player.Player2
            planet['n_ships'] = -planet['n_ships']

    def update_player_planet(self, planet: Dict, pending: Optional[Dict[Player, float]]):
        planet['n_ships'] += planet['growth_rate']
        if not pending:
            return
        own_incoming = pending.get(planet['owner'], 0.0)
        opp_incoming = pending.get(planet['owner'].opponent(), 0.0)
        planet['n_ships'] += own_incoming - opp_incoming
        if planet['n_ships'] < 0:
            planet['owner'] = planet['owner'].opponent()
            planet['n_ships'] = -planet['n_ships']

    def update_planets(self, pending: Dict[int, Dict[Player, float]]):
        for planet in self.state['planets']:
            p_pending = pending.get(planet.get('id'))
            if planet['owner'] == Player.Neutral:
                self.update_neutral_planet(planet, p_pending)
            else:
                self.update_player_planet(planet, p_pending)


if __name__ == "__main__":
    from core.game_state_factory import GameStateFactory
    from time import time
    

    params = GameParams()
    state = GameStateFactory(params).create_game()
    model = ForwardModel(state, params)
    start_time = time()
    for _ in range(1000):
        model.step({})  # simulate empty action dicts

    print(f"Steps: {ForwardModel.n_updates}")
    end_time = time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f'Steps per second: {ForwardModel.n_updates / (end_time - start_time)}')
