#!/bin/bash
#$ -cwd                        # Run from current directory
#$ -N apptainer_run            # Job name
#$ -pe smp 16                   # Request 8 CPU cores
#$ -l h_rt=01:00:00            # Runtime limit
#$ -l mem_free=1G              # 2GB per core (16GB total)
#$ -j y                        # Merge stdout and stderr
#$ -o run_output.log           # Combined output log
limit_threads() {
    local threads=$1
    shift
    export OMP_NUM_THREADS=$threads
    export MKL_NUM_THREADS=$threads
    export NUMEXPR_NUM_THREADS=$threads
    exec "$@"
}

NSLOTS_HALF=$((NSLOTS/2-1))
echo "=== Run job started on $(hostname) at $(date) ==="

# Set number of cores inside the container if needed


# Start server containers in background
limit_threads $NSLOTS_HALF apptainer run planetwars_python.sif --port 8080 --weights_path models/cont_gamma_999_no_galactic__1769651860_iter_3000.pt > server1.log 2>&1 &
SERVER1_PID=$!

limit_threads $NSLOTS_HALF apptainer run planetwars_python.sif --port 8081 --weights_path models/cont_gamma_999_v0.pt > server2.log 2>&1 &
SERVER2_PID=$!

# Wait for servers to be ready (check for "running" message in logs)
echo "Waiting for servers to start..."
# sleep 60  # Or use a more sophisticated check:
while ! grep -q "GameServerAgent running" server1.log; do sleep 1; done
while ! grep -q "GameServerAgent running" server2.log; do sleep 1; done

# Now run the client that connects to both servers
limit_threads 2 apptainer run planetwars.sif 8080,8081,10,50

# Cleanup: kill the background servers when client finishes
kill $SERVER1_PID $SERVER2_PID 2>/dev/null

echo "=== Run job finished at $(date) ==="