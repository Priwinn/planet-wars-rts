import subprocess
import sys
import time
import os
from league.run_agents_uniform import main as run_agents_uniform
from league.run_agents_trueskill import main as run_agents_trueskill

if __name__ == "__main__":
    # 1. Launch the Python agents server in a separate process
    # We do this because the agents server is async (websockets) and the league runner is sync/blocking.
    # If we run them in the same process, the blocking runner will freeze the agent servers.
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "NUMEXPR_NUM_THREADS": "16",
        "VECLIB_MAXIMUM_THREADS": "16",
    })
    print("🚀 Launching Python agents server...")
    agents_process = subprocess.Popen(
        [sys.executable, "-m", "league.launch_python_agents"],
        cwd=os.getcwd(),
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env
    )

    try:
        # Give the servers a moment to start up and register in the DB
        print("⏳ Waiting 60 seconds for agents to initialize...")
        time.sleep(60)

        # 2. Run the league (blocking)
        print("🏁 Starting league run...")
        run_agents_trueskill(3)
        print("✅ League run finished.")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # 3. Cleanup: Terminate the agents server
        print("💀 Killing agents server...")
        agents_process.terminate()
        try:
            agents_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agents_process.kill()
        print("👋 Done.")


