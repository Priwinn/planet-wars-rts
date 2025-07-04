import random


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

