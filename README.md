# Conway's Game of Life - MPI & CUDA Implementation
## Overview
This project implements Conway’s Game of Life using two parallel paradigms:

- <b>MPI Implementation:</b><br>Uses distributed memory parallelism via MPI with non-blocking communication. It distributes the grid among processes using a 1D row-wise decomposition, overlaps communication and computation, and gathers results for output.
- <b>CUDA Implementation:</b><br>Uses CUDA to accelerate convolution operations on video frames (as part of an image processing extension of the Game of Life). The CUDA version leverages GPU parallelism, applies various image kernels (such as blur, edge detection, sharpen, and identity), and optimizes performance by tuning block and grid sizes.

## Project Structure
### MPI Implementation
<ul>
    <li><b>life-nonblocking.cpp</b>
    <br>Contains the MPI version of the Game of Life. Key features include:
    <ul>
        <li> Data Distribution: The root process (rank 0) reads the initial live cell coordinates from an input file and distributes them among processes based on a 1D row-wise decomposition.
        <li> Non-blocking Communication: Uses MPI_Isend and MPI_Irecv to exchange boundary rows between neighboring processes without stalling computation.
        <li> Computation & Output: Each process computes the state for its assigned grid rows, and the root process aggregates the live cells into an output CSV file.
    </ul>
    (See <a href="https://github.com/Kryp6405/Conways-Game-of-Life/blob/main/MPI/life-nonblocking.cpp">MPI/life-nonblocking.cpp</a> for the source code details.)
    <li><b>Makefile</b><br>
    Automates compilation for the MPI implementation. (Makefile details are part of the project package.)
    <li><b>submit.sh</b>
    <br>A SLURM batch script for running the MPI executable on a cluster. This script:
    <ul>
        <li> Loads the OpenMPI module.
        <li> Compiles the project.
        <li> Runs the simulation with varying numbers of processes (4, 8, 16, 32, 64, 128) for performance analysis. (See <a href="https://github.com/Kryp6405/Conways-Game-of-Life/blob/main/MPI/submit.sh">MPI/submit.sh</a> for the full analysis.)
    </ul>
    <li><b>MPI_Analysis.pdf</b>
    <br>A report detailing the design, implementation strategy, and performance analysis of the MPI version. Key points include:
    <ul>
        <li> The 1D row-wise decomposition approach.
        <li> Performance scaling where optimal performance was observed with 64 processes.
        <li> The impact of communication overhead at high process counts (e.g., 128 processes).
    </ul>
    (See <a href="https://github.com/Kryp6405/Conways-Game-of-Life/blob/main/MPI_Analysis.pdf">MPI_Analysis.pdf</a> for the full analysis.)
</ul>

### CUDA Implementation
<ul>
    <li><b>game-of-life.cu</b><br>
    Contains the CUDA kernel(s) implementing the parallel processing. In the current form, these kernels perform convolution operations on video frames. They can be adapted for a Game of Life update rule (for example, by using a 3×3 kernel that sums eight neighbors with the center set to zero).<br>
    (Refer to <a href="https://github.com/Kryp6405/Conways-Game-of-Life/blob/main/CUDA/game-of-life.cu">CUDA/game-of-life.cu</a> for the driver implementation.)
    <li><b>Makefile (CUDA version)</b><br>
    Contains the rules to build the CUDA components of the project, compiling both the driver and the CUDA code.
    <li><b>Cuda_Analysis.pdf</b><br>
    This report details the CUDA implementation. Major points include:
    <ul>
        <li>Overview: The host transfers video frames and convolution kernels to the GPU and then uses a CUDA kernel (named convolveGPU) to process each frame. The processed frames are then written back to form an output video.
        <li>Performance Analysis: Experiments were carried out for various block sizes. It was observed that:
        <ul>
            <li>The execution time decreases significantly as the block size increases up to an optimal size (block size of 8).
            <li>Beyond the optimal point, performance degrades due to resource contention and reduced occupancy.
            <li>Careful tuning of grid and block dimensions is critical to balance parallelism and resource usage.
        </ul>
    </ul>
    (See <a href="https://github.com/Kryp6405/Conways-Game-of-Life/blob/main/CUDA_Analysis.pdf">CUDA_Analysis.pdf</a> for full details.)
</ul>

## Prerequisites
<ul>
    <li><b>MPI Implementation:</b>
    <ul>
        <li>MPI Library (e.g., OpenMPI)
        <li>C/C++ Compiler (e.g., GCC)
        <li>SLURM (for using the provided submit.sh on a cluster)
    </ul>
    <li><b>CUDA Implementation:</b>
    <ul>
        <li>CUDA Toolkit
        <li>A compatible NVIDIA GPU
        <li>C++ Compiler with CUDA support (e.g., nvcc)
    </ul>
</ul>

## Building the Project
The project is compiled using the provided Makefile. To build:
```bash
make clean
make all
```
This will compile the Game of Life implementations.

## Running the Implementations
### MPI Version
The executable expects the following command-line arguments:
```bash
mpirun -np <num_of_processes> ./life-nonblocking <input_file> <num_of_generations> <X_limit> <Y_limit>
```
- <b>input_file:</b> CSV file containing the initial live cell coordinates.
- <b>num_of_generations:</b> Number of generations for the simulation.
- <b>X_limit & Y_limit:</b> Dimensions (rows and columns) of the simulation grid.

For example, using the SLURM submission script:
```bash
mpirun -np 4 ./life-nonblocking life.1.512x512.data 500 512 512 > life-nonblocking4.out
mpirun -np 8 ./life-nonblocking life.2.512x512.data 500 512 512 > life-nonblocking8.out
...
```
Each command runs the simulation with a different number of processes to analyze scalability and performance.

### CUDA Version
The executable expects the following command-line arguments:
```bash
./game-of-life <input file name> <num_of_generations> <X_limit> <Y_limit> <gridSizeX> <gridSizeY> <output file name>
```
For example, using the SLURM submission script:
```bash
./game-of-life life.512x512.data 100 512 512 32 32 > result1.csv
./game-of-life life.22x22.data 100 22 22 8 8 > result2.csv
```
Each command runs the simulation with different block sizes to analyze scalability and performance.

## Implementation Details
### MPI Version
- <b>Input Data Distribution:</b><br>
The root process (rank 0) reads the input file, parses live cell coordinates, and distributes them to each process according to a 1D row-wise decomposition. Each process receives only the live cell data relevant to its assigned rows.

- <b>Non-blocking Communication:</b><br>
To update boundary cells efficiently, processes use non-blocking sends and receives (MPI_Isend, MPI_Irecv). This allows computation on inner cells to overlap with the communication of boundary data, optimizing performance.

- <b>Computation & Output Collection:</b><br>
Each process computes its portion of the new generation based on the Game of Life rules. Live cells are gathered across processes, and the root process writes the consolidated live cell coordinates to an output CSV file. The output file name is dynamically generated from the input file name and the generation count.

- <b>Performance Analysis:</b><br>
Detailed insights on performance, including execution time scaling and communication overhead, are discussed in the MPI_Analysis.pdf report. It is observed that while increasing process counts decreases computation time up to an optimal point (notably at 64 processes), further increases (e.g., 128 processes) introduce significant communication overhead.

### CUDA Version
- <b>Memory Management & Data Transfer:</b><br> The host allocates device memory for the simulation grid and initializes it with the initial state (live and dead cells). Data is transferred from host to GPU using CUDA memory management functions (e.g., cudaMemcpy) to ensure that the GPU has the necessary grid data prior to starting the simulation.

- <b>Parallel Computation:</b><br> The core of the simulation is performed by CUDA kernels launched from the host. Each kernel invocation computes the next generation for a subset of the grid in parallel. Every CUDA thread processes one or more cells, applying the standard Game of Life rules by evaluating the state of each cell’s eight neighbors. The kernel also handles boundary conditions to avoid out-of-bound memory access.
- <b>Double Buffering & Synchronization:</b><br> A double buffering strategy is employed where one buffer holds the current generation and another is used to store the next generation. After each generation, the buffers are swapped efficiently without excessive host-device synchronization, allowing the simulation to progress continuously across many generations.
- <b>Performance Optimization:</b><br> To maximize GPU performance, various optimization techniques are applied:
<ul>
    <ul>
        <li><b>Thread Block Tuning:</b> Careful selection of thread block sizes allows better occupancy and workload distribution.
        <li><b>Grid-Stride Loops:</b> Used to ensure all cells are processed even when the grid size exceeds the number of available threads.
        <li><b>Shared Memory Usage:</b> Where applicable, shared memory is utilized to reduce global memory access latency.
    </ul>
</ul>