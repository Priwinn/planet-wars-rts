from typing import Dict, Any
import numpy as np
from core.game_state import Player, Action

class gym_policy:
    """Base class for gym policies"""
    
    def __init__(self, game_params: Dict[str, Any], player: Player = Player.Player1):
        self.game_params = game_params
        self.player = player

    def __call__(self, game_state: Dict[str, Any]) -> Action:
        """Override this method in subclasses to implement specific policies"""
        raise NotImplementedError("Subclasses must implement __call__ method")

class RandomPolicy(gym_policy):
    """Random opponent policy"""
    
    def __call__(self, game_state: Dict[str, Any]) -> Action:
        planets = game_state['planets']
        player_int = 1 if self.player == Player.Player1 else 2

        # Find planets owned by the opponent that can send ships
        owned_planets = [
            p for p in planets 
            if (p['owner'] == player_int and 
                p['numShips'] > 0 and 
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
            player_id=self.player,
            source_planet_id=source['id'],
            destination_planet_id=target['id'],
            num_ships=num_ships
        )


class GreedyPolicy(gym_policy):
    """Greedy opponent policy - attacks weakest nearby targets"""
    def __call__(self, game_state: Dict[str, Any]) -> Action:
        planets = game_state['planets']
        player_int = 1 if self.player == Player.Player1 else 2

        # Find planets owned by the player that can send ships
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
        
        # Heuristic: prefer weak, nearby, fast-growing targets 
        def target_score(target):
            distance = np.sqrt((source['x'] - target['x'])**2 + (source['y'] - target['y'])**2)
            ship_strength = target['numShips'] if target['owner'] == 0 else target['numShips'] * 1.5
            return ship_strength + distance - 2 * target['growthRate']
        
        target = min(target_candidates, key=target_score)
        
        # Estimate whether the attack would succeed 
        distance = np.sqrt((source['x'] - target['x'])**2 + (source['y'] - target['y'])**2)
        eta = distance / self.game_params.get('transporterSpeed', 3.0)
        estimated_defense = target['numShips'] + target['growthRate'] * eta
        
        # Check if attack would succeed 
        if source['numShips'] <= estimated_defense:
            return Action.do_nothing()
        
        # Send half the ships
        num_ships = source['numShips'] / 2
        
        return Action(
            player_id=self.player,
            source_planet_id=source['id'],
            destination_planet_id=target['id'],
            num_ships=num_ships
        )
    
class FocusPolicy(gym_policy):
    """Focuses on a single target. Find closest target to centroid of owned planets and attack it until it is conquered."""
    def __init__(self, game_params: Dict[str, Any], player: Player = Player.Player1):
        super().__init__(game_params, player)
        self.target_planet_id = None

    def __call__(self, game_state: Dict[str, Any]) -> Action:
        planets = game_state['planets']
        player_int = 1 if self.player == Player.Player1 else 2
        #If target planet is conquered find a new one
        if self.target_planet_id is not None:
            target_planet = next((p for p in planets if p['id'] == self.target_planet_id), None)
            if target_planet is None or target_planet['owner'] == player_int:
                self.target_planet_id = None

        if self.target_planet_id is None:
            # Find a target planet if not already set
            # Calculate centroid
            x_sum = sum(p['x'] for p in planets if p['owner'] == player_int)
            y_sum = sum(p['y'] for p in planets if p['owner'] == player_int)
            count = sum(1 for p in planets if p['owner'] == player_int)

            if count == 0:
                return Action.do_nothing()
            
            centroid = (x_sum / count, y_sum / count)

            # Find closest weakest target planet to centroid
            def target_score(target):
                distance = np.sqrt((centroid[0] - target['x'])**2 + (centroid[1] - target['y'])**2)
                strength = target['numShips'] if target['owner'] == 0 else target['numShips'] * 1.5
                return distance + strength * 0.1
            
            target_planets = [p for p in planets if p['owner'] != player_int]
            self.target_planet_id = min(target_planets, key=target_score)['id']

        # If we have a target planet, proceed with the action
        if self.target_planet_id is not None:
            # Find the source planet
            owned_planets = [
                p for p in game_state['planets'] 
                if p['owner'] == (1 if self.player == Player.Player1 else 2) and
                p.get('transporter') is None
            ]
            if not owned_planets:
                return Action.do_nothing()
            source = max(owned_planets, key=lambda p: p['numShips'])
            target = next(p for p in planets if p['id'] == self.target_planet_id)

            # Send appropriate number of ships
            distance = np.sqrt((source['x'] - target['x'])**2 + (source['y'] - target['y'])**2)
            eta = distance / self.game_params.get('transporterSpeed', 3.0)
            transporters = sum([p['transporter']['numShips'] for p in planets if p.get('transporter') is not None and p['transporter']["owner"] == player_int and p['transporter']['destinationIndex'] == target['id']])
            estimated_defense = target['numShips'] + target['growthRate'] * eta - transporters

            # Calculate the number of ships to send
            owned_ships = sum(p['numShips'] for p in owned_planets)
            num_ships = min(estimated_defense * 1.5 * source['numShips'] / owned_ships, source['numShips'] * 0.8) 


            return Action(
                player_id=self.player,
                source_planet_id=source['id'],
                destination_planet_id=target['id'],
                num_ships=num_ships
            )

        return Action.do_nothing()
    
class DefensivePolicy(gym_policy):
    """Defensive policy that focuses on defending owned planets"""
    
    def __call__(self, game_state: Dict[str, Any]) -> Action:
        planets = game_state['planets']
        player_int = 1 if self.player == Player.Player1 else 2

        # Find owned planets
        owned_planets = [
            p for p in planets 
            if p['owner'] == player_int and p['numShips'] > 10 and p.get('transporter') is None
        ]
        
        if not owned_planets:
            return Action.do_nothing()
        
        # Looks for incoming threats
        threats = {planet['id']: 0 for planet in owned_planets}
        for i, p in enumerate(planets):
            if p['transporter'] and p['owner'] != player_int:
                threats[p['transporter']['id']] += p['numShips']

        # Find the planet with the highest threat
        target_planet = max(threats, key=threats.get)
        if threats[target_planet] == 0:
            return Action.do_nothing()
        source_planet = max([p for p in owned_planets if p['id'] != target_planet], key=lambda p: p['numShips'])
        # Send a portion of ships to defend
        num_ships = source_planet['numShips'] * 0.5

        return Action(
            player_id=self.player,
            source_planet_id=source_planet['id'],
            destination_planet_id=target_planet,
            num_ships=num_ships
        )
