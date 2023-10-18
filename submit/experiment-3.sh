#!/bin/bash
#SBATCH --job-name=experiment-3
#SBATCH --nodes=8
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

echo theta 0 \n\n
time mpirun -np 8 ./solution 10000_points.txt 1000 0
echo theta 0.3 \n\n
time mpirun -np 8 ./solution 10000_points.txt 1000 0.3
echo theta 0.5 \n\n
time mpirun -np 8 ./solution 10000_points.txt 1000 0.5
echo theta 0.7 \n\n
time mpirun -np 8 ./solution 10000_points.txt 1000 0.7
echo theta 0.9 \n\n
time mpirun -np 8 ./solution 10000_points.txt 1000 0.9
echo theta 1.1 \n\n
time mpirun -np 8 ./solution 10000_points.txt 1000 1.1
echo theta 1.3 \n\n
time mpirun -np 8 ./solution 10000_points.txt 1000 1.3