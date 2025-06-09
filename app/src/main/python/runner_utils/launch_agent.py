import subprocess
from pathlib import Path
from typing import Optional
from agent_entry import AgentEntry  # Assuming AgentEntry is defined elsewhere
import os


def run_command(cmd: list[str], cwd: Optional[Path] = None):
    print(f"Running: {' '.join(cmd)} (in {cwd or Path.cwd()})")
    subprocess.run(cmd, check=True, cwd=cwd)


def launch_agent(agent: AgentEntry, base_dir: Path):
    """
    Launches a game agent container from a GitHub repo.
    Clones to base_dir/agent.id, optionally checks out a commit, builds and runs the image.
    """
    repo_dir = base_dir / agent.id
    gradlew_path = repo_dir / "gradlew"

    print(f"Launching agent: {agent.id}")

    # Ensure base directory exists
    base_dir.mkdir(parents=True, exist_ok=True)

    # Clone if needed
    if not repo_dir.exists():
        run_command(["git", "clone", agent.repo_url, str(repo_dir)])
    else:
        run_command(["git", "fetch"], cwd=repo_dir)

    # Always checkout specific commit if provided
    if agent.commit:
        run_command(["git", "checkout", agent.commit], cwd=repo_dir)

    if gradlew_path.exists():
        gradle_cmd = "gradlew.bat" if os.name == 'nt' else "./gradlew"
        run_command([gradle_cmd, "build"], cwd=repo_dir)
    else:
        raise RuntimeError(f"{gradlew_path} does not exist. Ensure the repo contains the Gradle wrapper.")

    run_command(["podman", "build", "-t", f"game-server-{agent.id}", "."], cwd=repo_dir)

    # Run container with exposed port
    run_command([
        "podman", "run", "-d",
        "-p", f"{agent.port}:8080",
        "--name", f"container-{agent.id}",
        f"game-server-{agent.id}"
    ])
