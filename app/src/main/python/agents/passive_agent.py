from agents.planet_wars_agent import PlanetWarsPlayer
from core.game_state import GameState, Action

class PassiveAgent(PlanetWarsPlayer):
    """
    An agent that does nothing.
    """
    def get_action(self, game_state: GameState) -> Action:
        return Action.do_nothing()

    def get_agent_type(self) -> str:
        return "Passive Agent"
