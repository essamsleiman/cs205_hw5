// File       : main.cpp
// Description: Peak performance main driver.
// Copyright 2024 Harvard University. All Rights Reserved.
#include "helper.h"
#include <iomanip>
#include <iostream>
#include <vector>

// RULES:
//  1. Grading is based on one Skylake compute node.  Measurements reported
//     that do not correspond to the nominal hardware metrics of this node
//     cannot be graded and will not be published in the class ranking.  Use
//     $ salloc -N1 -c36 -t <time you need>
//     for an interactive compute node of the correct type.  If you use a SLURM
//     job script use at least the following:
//     #SBATCH --nodes=1
//     #SBATCH --ntasks-per-node=1
//     #SBATCH --cpus-per-task=36
//  2. You must use the GCC 7.3.1 compiler on the cluster
//  3. You must use the provided get_wtime() timer below to collect your time
//     measurements.

// TODO: Define here the total number of flops a single kernel invocation
//       of your implementation generates.  Do not just write down a number!
//       Use comments to explain your computation, for example:
//
//           // NLOOP * (ADD + MUL)
//           #define KERNEL_FLOPS (1000.0 * (2.0 + 2.0))
//
//       The above means that you use a loop with NLOOP iterations where you
//       perform 2 additions and 2 multiplications in each iteration.  Your
//       kernel implementation must match KERNEL_FLOPS.  The unit of this
//       number is flop (number of floating-point operations).

// NLOOP * (ADD + MUL)
#define KERNEL_FLOPS (1000000.0 * (8.0 + 8.0))

// Skylake nodes: 1 socket Intel Xeon Platinum 8124M.  The AVX2 boost frequency is 3.3GHz for
// this processor.  This frequency is sustainable for a small enough workload.
// If the CPU remains on that load for a long time, the frequency will scale
// back for thermal reasons. 
// TODO: Define the peak performance of the target architecture.  The unit of
// this number is Gflop/s.
// #define ARCHITECTURE_PEAK 1.0

// 18 cores * 3.3 GHz * 16 FLOPs/cycle = 950.4 GFLOP/s
#define ARCHITECTURE_PEAK 950.4

// Your peak performance kernel declaration.  See the kernel.cpp file
// TODO: update function signature if necessary
void kernel(void);

int main(void)
{
    // DISCLAIMER: Do not remove any lines.  You may add new lines of code or
    // extend existing lines where indicated.

    // You may add more code here
    std::vector<double> tthread;
    #pragma omp parallel num_threads(36) /* TODO: you may add clauses here */
    {
        // You may add more code here
        const double t0 = get_wtime();
        // TODO: update function arguments if necessary (you may allocate memory
        // for arrays in the sequential region above)
        kernel();
        const double t1 = get_wtime();
        // You may add more code here
    #pragma omp critical
        {
            // You may add more code here
            tthread.push_back(t1 - t0);
        }
    }

    // DISCLAIMER: You are not allowed to change the following lines but you may
    // add more lines to the report section below.
    const Stats t = statistics(tthread);
    // Performance based on max time in Gflop/s:
    const double perf = tthread.size() * KERNEL_FLOPS * 1.0e-9 / t.max;
    const double peak = ARCHITECTURE_PEAK;

    // Report
    std::cout << "number of threads: " << tthread.size() << '\n';
    std::cout << "tmean:             " << t.mean << " +/-" << t.std << '\n';
    std::cout << "tmin:              " << t.min << '\n';
    std::cout << "t50%:              " << t.p50 << '\n';
    std::cout << "tmax:              " << t.max << '\n';
    std::cout << "performance:       " << perf << " Gflop/s (" << std::fixed
              << std::setprecision(1) << perf / peak * 100.0 << "% of peak)\n";
    std::cout << "thread imbalance:  " << (t.max - t.mean) / t.max * 100.0
              << "%\n";

    return 0;
}
