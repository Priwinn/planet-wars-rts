from runner_utils.fast_agent_eval import fast_agent_eval
from runner_utils.evaluate_python_agent_in_league import evaluate_python_agent
from core.game_state import GameParams
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.better_greedy_heuristic_agent import BetterGreedyHeuristicAgent
from agents.GalacticArmada import GalacticArmada
from agents.torch_agent_gnn import TorchAgentGNN
from agents.gnn import PlanetWarsAgentGNN
from agents.ppo import Args
from agents.random_agents import PureRandomAgent, CarefulRandomAgent  
import numpy as np

# test_agent = GreedyHeuristicAgent()  # replace with your actual agent
def _fast_agent_eval(test_agent, n_games, game_params, baseline_agents): 
    # time how long it takes to run the evaluation
    import time
    start_time = time.time()
    win_rate = fast_agent_eval(test_agent, n_games=n_games, game_params=game_params,baseline_agents=baseline_agents)
    print(f"\nFinal average win rate: {win_rate:.3f}")
    end_time = time.time()
    print(f"Total evaluation time: {end_time - start_time:.2f} seconds")
    return win_rate

def get_random_game_params():
    game_params={
                    'numPlanets': np.random.randint(10, 31),
                    'maxTicks': 2000,
                    'transporterSpeed': np.random.uniform(2.0, 5.0),
                    'width': 640,
                    'height': 480,
                    'minGrowthRate': 0.05,
                    'maxGrowthRate': 0.2,
                    'initialNeutralRatio': np.random.uniform(0.25, 0.35)
                }
    return GameParams(**game_params)

def _fast_agent_eval_random(test_agent, n_maps, games_per_map, baseline_agents):
    wr=[]
    for _ in range(n_maps):
        game_params = get_random_game_params()
        print(f"Evaluating on map with params: {game_params}")
        wr.append(_fast_agent_eval(test_agent, n_games=games_per_map, game_params=game_params, baseline_agents=baseline_agents))
    print(f"Average win rate over {n_maps} maps: {np.mean(wr):.3f} ± {np.std(wr):.3f}")

def _evaluate_agent_in_league(test_agent, port):
    return evaluate_python_agent(test_agent, port=port)

if __name__ == "__main__":
    import os
    test_agent = TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/PlanetWarsForwardModelGNN__ppo_config__random__1755944875_final.pt") 
    # test_agent = BetterGreedyHeuristicAgent()  # replace with your actual agent
    # _fast_agent_eval(test_agent, n_games=30, game_params=GameParams(num_planets=20), baseline_agents=[CarefulRandomAgent(), BetterGreedyHeuristicAgent()])
    _fast_agent_eval_random(test_agent, n_maps=5, games_per_map=10, baseline_agents=[GalacticArmada()])
    # _evaluate_agent_in_league(test_agent, port=8080)