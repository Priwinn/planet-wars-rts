#!/bin/bash
#$ -cwd                        # Run from current directory
#$ -N apptainer_run            # Job name
#$ -pe smp 16                   # Request 8 CPU cores
#$ -l h_rt=01:00:00            # Runtime limit
#$ -l mem_free=1G              # 2GB per core (16GB total)
#$ -j y                        # Merge stdout and stderr
#$ -o run_output.log           # Combined output log

echo "=== Run job started on $(hostname) at $(date) ==="

# Set number of cores inside the container if needed
export APPTAINERENV_NSLOTS=$((NSLOTS/2))

# Run the container
apptainer run planetwars_python.sif --port 8080
apptainer run planetwars_python.sif --port 8081

echo "=== Run job finished at $(date) ==="