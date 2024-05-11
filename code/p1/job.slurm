#!/bin/bash
#SBATCH --job-name=PeakPerformanceTest  # Name of the job
#SBATCH --nodes=1                      # Use one node
#SBATCH --ntasks-per-node=1            # Run one task on that node
#SBATCH --cpus-per-task=18             # Number of CPU cores per task (matches the cores of Xeon Platinum 8124M)
#SBATCH --time=01:00:00                # Time limit hrs:min:sec
#SBATCH --output=result-%j.out         # Standard output and error log
#SBATCH --error=result-%j.err

module load gcc/7.3.1                  # Load the required module for GCC 7.3.1
module load openmpi/3.1.4              # Load OpenMPI module if needed

# Ensure that OpenMP uses the right number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the main executable
./main

