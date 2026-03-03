from agents.gnn import PlanetWarsAgentGNN
from agents.torch_agent_gnn import TorchAgentGNN
import copy
import torch
from torch import nn
import cProfile
import pstats

from core.game_state import GameParams, Player
from core.game_state_factory import GameStateFactory

agent = TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_gamma_999_v0.pt", use_topk_q=False)
agent.prepare_to_play_as(Player.Player1, GameParams(num_planets=20))
game_state = GameStateFactory(GameParams(num_planets=20)).create_game()
pr = cProfile.Profile()
pr.enable()

for i in range(1000):
    action = agent.get_action(game_state)


pr.disable()
ps = pstats.Stats(pr).sort_stats('tottime')
ps.print_stats(20)

backend = "fbgemm"
m = agent.model
torch.quantization.fuse_modules(m, ['critic.0','critic.1'], inplace=True) # fuse first Conv-ReLU pair
torch.quantization.fuse_modules(m, ['source_actor.0','source_actor.1'], inplace=True) # fuse second Conv-ReLU pair
torch.quantization.fuse_modules(m, ['target_actor.0','target_actor.1'], inplace=True) # fuse third Conv-ReLU pair
torch.quantization.fuse_modules(m, ['ratio_actor_mean.0','ratio_actor_mean.1'], inplace=True) # fuse fourth Conv-ReLU pair
torch.quantization.fuse_modules(m, ['noop_actor.0','noop_actor.1'], inplace=True) # fuse fifth Conv-ReLU pair
m.critic = nn.Sequential(torch.quantization.QuantStub(), 
                        *m.critic,
                        torch.quantization.DeQuantStub())
m.source_actor = nn.Sequential(torch.quantization.QuantStub(),
                                    *m.source_actor,
                                    torch.quantization.DeQuantStub())
m.target_actor = nn.Sequential(torch.quantization.QuantStub(),
                                    *m.target_actor,
                                    torch.quantization.DeQuantStub())
m.ratio_actor_mean = nn.Sequential(torch.quantization.QuantStub(),
                                    *m.ratio_actor_mean,
                                    torch.quantization.DeQuantStub())
m.noop_actor = nn.Sequential(torch.quantization.QuantStub(),
                                    *m.noop_actor,
                                    torch.quantization.DeQuantStub())

m.qconfig = torch.quantization.get_default_qconfig(backend)
torch.quantization.prepare(m, inplace=True)


for i in range(1000):
    game_state = GameStateFactory(GameParams(num_planets=20)).create_game()
    action = agent.get_action(game_state)
torch.quantization.convert(m, inplace=True)

pr1 = cProfile.Profile()
pr1.enable()

for i in range(1000):
    action = agent.get_action(game_state)


pr1.disable()
ps1 = pstats.Stats(pr1).sort_stats('tottime')
ps1.print_stats(20)
