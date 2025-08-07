from runner_utils.fast_agent_eval import fast_agent_eval
from runner_utils.evaluate_python_agent_in_league import evaluate_python_agent
from core.game_state import GameParams
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.better_greedy_heuristic_agent import BetterGreedyHeuristicAgent
from agents.torch_agent_gnn import TorchAgentGNN
from agents.gnn import PlanetWarsAgentGNN
from agents.ppo import Args
from agents.random_agents import PureRandomAgent, CarefulRandomAgent  

# test_agent = GreedyHeuristicAgent()  # replace with your actual agent
def _fast_agent_eval(test_agent, n_games, game_params, baseline_agents): 
    # time how long it takes to run the evaluation
    import time
    start_time = time.time()
    win_rate = fast_agent_eval(test_agent, n_games=n_games, game_params=game_params,baseline_agents=baseline_agents)
    print(f"\nFinal average win rate: {win_rate:.3f}")
    end_time = time.time()
    print(f"Total evaluation time: {end_time - start_time:.2f} seconds")

def _evaluate_agent_in_league(test_agent, port):
    return evaluate_python_agent(test_agent, port=port)

if __name__ == "__main__":
    import os
    test_agent = TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/PlanetWarsForwardModelGNN__ppo__random__1754542421_iter_700.pt") 
    # test_agent = BetterGreedyHeuristicAgent()  # replace with your actual agent
    _fast_agent_eval(test_agent, n_games=100, game_params=GameParams(num_planets=20), baseline_agents=[CarefulRandomAgent(), BetterGreedyHeuristicAgent()])
    # _evaluate_agent_in_league(test_agent, port=8080)