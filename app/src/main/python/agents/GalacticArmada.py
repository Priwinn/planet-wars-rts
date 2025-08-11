#https://github.com/nikivanstein/planet-wars-rts-python/tree/feature/codex-troubleshoot-submission-of-galacticarmada.py
import random
from typing import Optional

from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action, Player, GameParams
from core.game_state_factory import GameStateFactory


class GalacticArmada(PlanetWarsPlayer):
    def get_action(self, game_state: GameState) -> Action:
        my_planets = [p for p in game_state.planets
                      if p.owner == self.player and p.transporter is None and p.n_ships > 0]
        enemy_planets = [p for p in game_state.planets if p.owner == self.player.opponent()]
        neutral_planets = [p for p in game_state.planets if p.owner == Player.Neutral]
        all_planets = my_planets + enemy_planets + neutral_planets

        if not my_planets:
            return Action.do_nothing()

        # Prioritize attacking enemy planets if they exist, otherwise attack neutral planets
        targets = enemy_planets if enemy_planets else neutral_planets

        if not targets:
            return Action.do_nothing()

        # Sort my planets by ship count, descending
        my_planets = sorted(my_planets, key=lambda p: p.n_ships, reverse=True)

        # Find weakest friendly planet to defend
        weakest_planet = None
        if my_planets:
            weakest_planet = min(my_planets, key=lambda p: p.n_ships + p.growth_rate * 5) # 5 turns of growth

        # Consolidate forces on the strongest planet if close enough, but limit distance and ship count.
        strongest_planet = my_planets[0]
        for planet in my_planets[1:]:
            if planet.position.distance(strongest_planet.position) < 12 and planet.n_ships > 5 and strongest_planet.n_ships < 100:
                transfer_amount = int(min(planet.n_ships * 0.2, 40))  # Reduced transfer amount to conserve ships
                return Action(
                    player_id=self.player,
                    source_planet_id=planet.id,
                    destination_planet_id=strongest_planet.id,
                    num_ships=transfer_amount
                )

        # Dynamic target evaluation incorporating distance, ship strength and growth rate
        for source in my_planets:
            def target_score(target):
                distance = source.position.distance(target.position)
                ship_strength = target.n_ships
                if target.owner == Player.Neutral:
                    ship_strength *= 0.7  # Even less aggressive on neutrals
                else:
                    ship_strength *= 1.8 # Still aggressive on enemies, but slightly reduced
                growth_rate = target.growth_rate
                # Add a small random factor to break ties and explore diverse targets
                random_factor = random.uniform(-0.2, 0.2)
                return ship_strength + distance - 2.5 * growth_rate + random_factor# Prioritize nearby planets with high growth rate and low ship count.  Slightly reduced growth rate influence

            target = min(targets, key=target_score)

            # Enhanced ship allocation strategy: Send enough ships for a successful attack, plus some margin
            distance = source.position.distance(target.position)
            eta = distance / self.params.transporter_speed
            estimated_defense = target.n_ships + target.growth_rate * eta

            # Threat assessment: Increase the safety margin if the target is an enemy planet
            safety_margin = 1.2 if target.owner == self.player.opponent() else 1.05

            required_ships = int(estimated_defense * safety_margin)
            available_ships = source.n_ships
            #Conserve ships: Send a smaller percentage of ships, but more effectively.
            send_ships = min(available_ships, required_ships, int(available_ships * 0.7))

            if send_ships > 0:
                return Action(
                    player_id=self.player,
                    source_planet_id=source.id,
                    destination_planet_id=target.id,
                    num_ships=send_ships
                )

        # Improved defense action: Only defend if the weakest planet is under threat
        if weakest_planet is not None:
            # Check for nearby enemy planets that could attack the weakest planet
            nearby_enemies = [p for p in enemy_planets if p.position.distance(weakest_planet.position) < 20]

            # Consider enemies further away as well, scaled by distance
            far_enemies = [p for p in enemy_planets if 20 <= p.position.distance(weakest_planet.position) < 40]
            scaled_far_enemies = [(p, 0.3) for p in far_enemies] # Reduce the threat from far enemies more strongly

            all_threatening_enemies = nearby_enemies + [p[0] for p in scaled_far_enemies] # Only planets

            if all_threatening_enemies:
                # Assess the threat level: if the combined fleet size of nearby enemies is larger than the defense of the weakest planet, defend it.
                enemy_fleet_size = sum(p.n_ships for p in nearby_enemies) + sum(p.n_ships * 0.3 for p in far_enemies) #Weighted sum to take distance into account
                # Predict enemy arrival based on distance and transporter speed
                arrival_ticks = min(((weakest_planet.position.distance(enemy.position) / self.params.transporter_speed) for enemy in nearby_enemies), default=10)

                defense = weakest_planet.n_ships + weakest_planet.growth_rate * arrival_ticks #estimate growth until arrival

                if enemy_fleet_size > defense:
                    # Prioritize defense by checking if we can overwhelm the enemy attack from several planets at once
                    defending_planets = [p for p in my_planets if p.id != weakest_planet.id and p.n_ships > 10 and p.position.distance(weakest_planet.position) < 30]
                    total_available_ships = sum(min(p.n_ships * 0.4, 40) for p in defending_planets) # Reduced ship contribution, further reducing over-committing
                    if total_available_ships > (enemy_fleet_size - defense):
                        # Coordinate defense with multiple planets
                        for planet in defending_planets:
                            transfer_amount = int(min(planet.n_ships * 0.4, 40)) #Reduced ship contribution
                            if transfer_amount > 0:
                                action = Action(
                                    player_id=self.player,
                                    source_planet_id=planet.id,
                                    destination_planet_id=weakest_planet.id,
                                    num_ships=transfer_amount
                                )
                                return action
                    else:
                        # Transfer ships from the strongest planet to the weakest planet
                        if strongest_planet.id != weakest_planet.id and strongest_planet.n_ships > 20:
                            transfer_amount = int(min(strongest_planet.n_ships * 0.2, 50))  # Reduced transfer amount
                            action = Action(
                                player_id=self.player,
                                source_planet_id=strongest_planet.id,
                                destination_planet_id=weakest_planet.id,
                                num_ships=transfer_amount
                            )
                            return action

        return Action.do_nothing()

    def get_agent_type(self) -> str:
        return "Galactic Armada"