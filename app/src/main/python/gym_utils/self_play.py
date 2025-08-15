import random
from agents.random_agents import PureRandomAgent, CarefulRandomAgent
from agents.better_greedy_heuristic_agent import BetterGreedyHeuristicAgent
from agents.baseline_policies import GreedyPolicy


class SelfPlayBase:
    def get_opponent(self):
        """Get the opponent player for self-play."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    def add_opponent(self, opponent):
        """Add an opponent player for self-play."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class NaiveSelfPlay(SelfPlayBase):
    def __init__(self, player_id):
        self.player_id = player_id
        self.opponent = None


    def get_opponent(self):
        if self.opponent is None:
            raise ValueError("Opponent has not been set yet.")
        return self.opponent

    def add_opponent(self, opponent):
        self.opponent = opponent

class BufferSelfPlay(SelfPlayBase):
    def __init__(self, player_id, pool_size=5):
        self.player_id = player_id
        self.pool_size = pool_size
        self.opponents = []

    def get_opponent(self):
        if not self.opponents:
            raise ValueError("No opponents have been set yet.")
        return random.choice(self.opponents)

    def add_opponent(self, opponent):
        self.opponents.append(opponent)
        if len(self.opponents) > self.pool_size:
            self.opponents.pop(0)

class BaselineBufferSelfPlay(BufferSelfPlay):
    def __init__(self, player_id, pool_size=5, baseline_opponents=[BetterGreedyHeuristicAgent(), GreedyPolicy, CarefulRandomAgent()], baseline_ratio=0.5):
        super().__init__(player_id, pool_size)
        self.baseline_opponents = baseline_opponents if baseline_opponents is not None else []
        self.baseline_ratio = baseline_ratio

    def add_opponent(self, opponent):
        return super().add_opponent(opponent)
    
    def get_opponent(self):
        if not self.opponents and not self.baseline_opponents:
            raise ValueError("No opponents or baseline opponents have been set yet.")
        if not self.baseline_opponents:
            return super().get_opponent()
        else:
            return random.choice(self.baseline_opponents) if random.random() < self.baseline_ratio else super().get_opponent()