#!/bin/bash
#$ -cwd
#$ -N apptainer_build
#$ -l h_rt=1:00:00
#$ -pe smp 16
#$ -l mem_free=1G
#$ -o apptainer/build_output.log
#$ -e apptainer/build_error.log

cd planet-wars-rts
echo "Build job started on $(hostname) at $(date)"
apptainer build --force apptainer/planetwars_python.sif apptainer/planetwars_python.def 
echo "Build job finished at $(date)"
