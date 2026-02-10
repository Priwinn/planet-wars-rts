#!/bin/bash
#$ -cwd                        # Run from current directory
#$ -N apptainer_run            # Job name
#$ -pe smp 16                   # Request 8 CPU cores
#$ -l h_rt=01:00:00            # Runtime limit
#$ -l mem_free=1G              # 2GB per core (16GB total)
#$ -j y                        # Merge stdout and stderr
#$ -o pair_logs/           # Combined output log


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

# Find two open ports for the servers (starting from 8080)
find_open_port() {
    local port=$1
    while netstat -tuln | grep -q ":$port "; do
        port=$((port + 1))
    done
    echo $port
}
PORT1=$(find_open_port 8080)
PORT2=$(find_open_port $((PORT1 + 1)))


# Start server containers in background
date_str=$(date +%s)
limit_threads $NSLOTS_HALF apptainer run apptainer/planetwars_python.sif --port 8080 --model_class galactic > "pair_logs/server1_${date_str}.log" 2>&1 &
# limit_threads $NSLOTS_HALF apptainer run apptainer/planetwars_python.sif --port $PORT1 --weights_path models/cont_gamma_999_v0.pt > "pair_logs/server1_${date_str}.log" 2>&1 &
SERVER1_PID=$!

limit_threads $NSLOTS_HALF apptainer run apptainer/planetwars_python.sif --port $PORT2 --weights_path models/cont_gamma_999_200M__1770074759_final.pt > "pair_logs/server2_${date_str}.log" 2>&1 &
SERVER2_PID=$!

# Wait for servers to be ready (check for "running" message in logs)
echo "Waiting for servers to start..."

while ! grep -q "GameServerAgent running" "pair_logs/server1_${date_str}.log"; do sleep 1; done
while ! grep -q "GameServerAgent running" "pair_logs/server2_${date_str}.log"; do sleep 1; done

# Now run the client that connects to both servers
limit_threads 2 apptainer run apptainer/planetwars.sif $PORT1,$PORT2,50,50

# Cleanup: kill the background servers when client finishes
kill $SERVER1_PID $SERVER2_PID 2>/dev/null

echo "=== Run job finished at $(date) ==="