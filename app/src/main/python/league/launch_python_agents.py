import asyncio
import logging
import socket
from datetime import datetime
from typing import Dict, Type

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agents.planet_wars_agent import PlanetWarsPlayer
from agents.random_agents import CarefulRandomAgent, PureRandomAgent
from agents.greedy_heuristic_agent import GreedyHeuristicAgent
from agents.better_greedy_heuristic_agent import BetterGreedyHeuristicAgent
from agents.passive_agent import PassiveAgent
from agents.torch_agent_gnn import TorchAgentGNN
from agents.gnn import PlanetWarsAgentGNN
from agents.GalacticArmada import GalacticArmada
# Add other agents as needed

from client_server.game_agent_server import GameServerAgent
from league.init_db import get_default_db_path
from league.league_schema import Agent, AgentInstance, Base
from runner_utils.utils import find_free_port

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB Config
DB_PATH = get_default_db_path()
ENGINE = create_engine(DB_PATH)
Base.metadata.create_all(ENGINE)

#Limit torch threads
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# Define the agents we want to launch
# Name -> Agent Instance
LOCAL_AGENTS: Dict[str, PlanetWarsPlayer] = {
    "BetterGreedyHeuristic": BetterGreedyHeuristicAgent(),
    "GalacticArmada": GalacticArmada(),
    "cont_999_v0": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_gamma_999_v0.pt"),
    "cont_999_200M": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_gamma_999_200M__1770074759_final.pt"),
    "cont_999_v1": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_999_v1.pt"),
    "cont_999_v2": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_999_v2.pt"),
    "cont_1_v0": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_1_v0.pt"),
    "cont_1_v1": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_1_v1.pt"),
    "cont_1_v2": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_1_v2.pt"),
    "d11_999_v0": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/d11_999_v0.pt"),
    "d11_999_v1": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/d11_999_v1.pt"),
    "d11_999_v2": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/d11_999_v2.pt"),
    "d11_1_v0": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/d11_1_v0.pt"),
    "cont_999_ship_delta": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_999_ship_delta.pt"),
    "cont_999_no_hier": TorchAgentGNN(model_class=PlanetWarsAgentGNN, weights_path="models/cont_999_no_hier.pt"), 
}


async def launch_server(name: str, agent: PlanetWarsPlayer, port: int):
    """
    Launches the GameServerAgent on the specified port.
    """
    server = GameServerAgent(host="0.0.0.0", port=port, agent=agent)
    logger.info(f"🚀 Starting {name} on port {port}")
    try:
        await server.start()
    except Exception as e:
        logger.error(f"❌ Error running {name} on port {port}: {e}")

def ensure_agent_in_db(session: Session, name: str) -> Agent:
    """Ensures the agent exists in the DB. Returns the Agent object."""
    # We use repo_url="local" and commit="latest" to identify these local agents
    agent = session.query(Agent).filter_by(name=name, repo_url="local").first()
    if not agent:
        logger.info(f"Creating DB entry for {name}")
        agent = Agent(
            name=name,
            owner="local",
            repo_url="local",
            commit="latest"
        )
        session.add(agent)
        session.commit()
    return agent

def update_agent_instance(session: Session, agent_id: int, port: int):
    """Updates or creates the AgentInstance record."""
    inst = session.query(AgentInstance).filter_by(agent_id=agent_id).first()
    if inst:
        inst.port = port
        inst.container_id = "python-process"
        inst.last_seen = datetime.utcnow()
    else:
        inst = AgentInstance(
            agent_id=agent_id,
            port=port,
            container_id="python-process",
            last_seen=datetime.utcnow()
        )
        session.add(inst)
    session.commit()

async def main(agents: Dict[str, Type[PlanetWarsPlayer]] = LOCAL_AGENTS):
    tasks = []
    
    # We use a single session for setup, but be careful with threading if we were using threads.
    # Here we are single-threaded async, so it's fine.
    with Session(ENGINE) as session:
        for name, agent_instance in agents.items():
            # 1. Ensure Agent exists in DB
            db_agent = ensure_agent_in_db(session, name)
            
            # 2. Find a free port
            # Note: There's a small race condition here if ports are taken quickly,
            # but for a local script launching a few agents, it's usually fine.
            port = find_free_port()
            
            # 3. Update AgentInstance in DB
            update_agent_instance(session, db_agent.agent_id, port)
            
            # 4. Start Server in background task
            tasks.append(asyncio.create_task(launch_server(name, agent_instance, port)))

    if tasks:
        logger.info(f"Running {len(tasks)} agents...")
        # Wait for all servers (they run forever)
        await asyncio.gather(*tasks)
    else:
        logger.info("No agents to run.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
