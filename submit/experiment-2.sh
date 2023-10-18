#!/bin/bash
#SBATCH --job-name=experiment-2
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:30:00
#SBATCH -o ./output_%x.out

module load "GCC/11.3.0"
module load "OpenMPI/4.1.4"

echo start compile
make clean
make
echo start run

echo 10000 points
echo 16 nodes
time mpirun -np 16 ./solution 10000_points.txt 1000 0.7

echo 8 nodes
time mpirun -np 8 ./solution 10000_points.txt 1000 0.7

echo 4 nodes
time mpirun -np 4 ./solution 10000_points.txt 1000 0.7

echo 2 nodes
time mpirun -np 2 ./solution 10000_points.txt 1000 0.7

echo 1 nodes
time mpirun -np 1 ./solution 10000_points.txt 1000 0.7

echo 5000 points
echo 16 nodes
time mpirun -np 16 ./solution 5000_points.txt 1000 0.7

echo 8 nodes
time mpirun -np 8 ./solution 5000_points.txt 1000 0.7

echo 4 nodes
time mpirun -np 4 ./solution 5000_points.txt 1000 0.7

echo 2 nodes
time mpirun -np 2 ./solution 5000_points.txt 1000 0.7

echo 1 nodes
time mpirun -np 1 ./solution 5000_points.txt 1000 0.7

echo 1000 points
echo 16 nodes
time mpirun -np 16 ./solution 1000_points.txt 1000 0.7

echo 8 nodes
time mpirun -np 8 ./solution 1000_points.txt 1000 0.7

echo 4 nodes
time mpirun -np 4 ./solution 1000_points.txt 1000 0.7

echo 2 nodes
time mpirun -np 2 ./solution 1000_points.txt 1000 0.7

echo 1 nodes
time mpirun -np 1 ./solution 1000_points.txt 1000 0.7

echo 100 points
echo 16 nodes
time mpirun -np 8 ./solution 100_points.txt 1000 0.7

echo 8 nodes
time mpirun -np 8 ./solution 100_points.txt 1000 0.7

echo 4 nodes
time mpirun -np 4 ./solution 100_points.txt 1000 0.7

echo 2 nodes
time mpirun -np 2 ./solution 100_points.txt 1000 0.7

echo 1 nodes
time mpirun -np 1 ./solution 100_points.txt 1000 0.7