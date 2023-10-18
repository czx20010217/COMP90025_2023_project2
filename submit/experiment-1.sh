#!/bin/bash
#SBATCH --job-name=experiment-1
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:30:00
#SBATCH -o ./output_%x.out

module load "GCC/11.3.0"
module load "OpenMPI/4.1.4"

echo start compile
make clean
make solution
echo start run

echo 1000000 points with 50 iteration
time mpirun -np $SLURM_NNODES ./solution 1000000_points.txt 50 0.9

echo 100000 points
time mpirun -np $SLURM_NNODES ./solution 100000_points.txt 1000 0.9

echo 10000 points 
time mpirun -np $SLURM_NNODES ./solution 10000_points.txt 1000 0.9

echo 5000 points
time mpirun -np $SLURM_NNODES ./solution 5000_points.txt 1000 0.9

echo 1000 points
time mpirun -np $SLURM_NNODES ./solution 1000_points.txt 1000 0.9