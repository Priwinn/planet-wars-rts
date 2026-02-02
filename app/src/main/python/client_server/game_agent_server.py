import asyncio
import argparse
import json
import uuid
from websockets import serve
from typing import Dict, Any, Callable

from agents.random_agents import CarefulRandomAgent
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.planet_wars_agent import PlanetWarsPlayer
# from agents.torch_agent import TorchAgent
# from agents.mlp import PlanetWarsAgentMLP
from agents.torch_agent_gnn import TorchAgentGNN
from agents.gnn import PlanetWarsAgentGNN
from config_files.ppo_config import Args 
from client_server.util import RemoteInvocationRequest, RemoteInvocationResponse, deserialize_args, serialize_result
from core.game_state import Player, camel_to_snake


class GameServerAgent:
    def __init__(self, host: str = "localhost", port: int = 8765, agent: PlanetWarsPlayer = CarefulRandomAgent):
        self.host = host
        self.port = port
        self.agent_map: Dict[str, PlanetWarsPlayer] = {}
        self.agent = agent

    async def handler(self, websocket):
        async for message in websocket:
            try:
                request = RemoteInvocationRequest.model_validate_json(message)
                start=asyncio.get_event_loop().time()
                # print(f"\nReceived request at time: {request}")

                if request.requestType == "init":
                    agent_id = str(uuid.uuid4())
                    agent = self.agent
                    self.agent_map[agent_id] = agent
                    result = {"objectId": agent_id}

                elif request.requestType == "invoke":
                    agent = self.agent_map.get(request.objectId)
                    if not agent:
                        raise ValueError(f"No such agent: {request.objectId}")

                    method_name = camel_to_snake(request.method)
                    method = getattr(agent, method_name, None)
                    if method is None:
                        raise ValueError(f"Unknown method: {request.method}")
                    if not hasattr(agent, method_name):
                        raise ValueError(f"Unknown method: {method_name}")

                    method: Callable = getattr(agent, method_name)
                    # print(f"Invoking method: {method_name} on agent {agent} with args: {request.args}")
                    args = deserialize_args(method_name, request.args)
                    # print(f"Deserialized args: {args}")
                    preprocess_time = asyncio.get_event_loop().time()
                    print(f"Preprocessing time for {method_name} took {preprocess_time - start:.4f} seconds")
                    result = method(*args)
                    postprocess_time = asyncio.get_event_loop().time()
                    print(f"Method execution time for {method_name} took {postprocess_time - preprocess_time:.4f} seconds")
                    result = serialize_result(result)

                elif request.requestType == "end":
                    removed = self.agent_map.pop(request.objectId, None)
                    msg = "Agent removed" if removed else "No such agent"
                    result = {"message": msg}

                else:
                    raise ValueError(f"Unknown request type: {request.requestType}")

                response = RemoteInvocationResponse(status="ok", result=result)
                end=asyncio.get_event_loop().time()
                try:
                    print(f"Postprocessing time took {end - postprocess_time:.4f} seconds")
                except Exception:
                    pass
                print(f"Sending response took {end - start:.4f} seconds")

            except Exception as e:
                print(f"Error handling message: {e}")
                response = RemoteInvocationResponse(status="error", error=str(e))
            
            # end_time = time.time()
            # print(f"Method took {end_time - start_time:.4f} seconds")

            await websocket.send(response.model_dump_json())

    async def start(self):
        async with serve(self.handler, self.host, self.port):
            print(f"GameServerAgent running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game Server Agent")
    parser.add_argument("--model_class", type=str, default="PlanetWarsAgentGNN", help="Model class name")
    parser.add_argument("--weights_path", type=str, default="models/cont_gamma_999__1765185011_final.pt", help="Path to model weights")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    args = parser.parse_args()

    if args.model_class == "PlanetWarsAgentGNN":
        agent = TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path=args.weights_path)  
    elif args.model_class == "galactic":
        from agents.GalacticArmada import GalacticArmada
        agent = GalacticArmada()
    asyncio.run(GameServerAgent(host="0.0.0.0", port=args.port, agent=agent).start())
    #Use 0.0.0.0 for container access