# TDT4200 Problem set 6: Naive GEMM vs CUBLAS

## GEMM using CUDA and cuBLAS

In this assignment you will work on two implementation of general matrix multiplications. One using only CUDA, and one using CUDA and cuBLAS.

The serial solution can be found in `matmul_serial.c` and should be kept as a reference. Skeletons for your parallel implementations can be found in `matmul_cuda.cu` and `matmul_cublas.cu`. You should complete the parallel implementations as described by the problem set description.

## Run
### Setup
Set up the project structure using `make setup`.
Generate the matrices using `make generate`.

Creates folders `data`
- `data`: contains the binary files for the matrices

### Serial solution
**Compile**

`make serial`

**Run**

`./serial [matrix_size]`

**Example**

```
make serial
./serial 1024
```

### Parallel solution
**Compile**

`make parallel`

**Run**

`./parallel [matrix_size]`

**Example**

```
make parallel
./parallel 256
```

## Check
- `make check_cuda`
- `make check_cublas`

Compares the output of the CUDA or cuBLAS implementation to the serial implementation.

## Installing dependencies
**CUDA**

Linux/Ubuntu:

An in-depth overview of how to install CUDA on Linux can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

MacOSX:

NVIDIA no longer supports development on macOS. Please opt for using
the Snotra cluster or another operating system. The TDT4200 staff will not
be able to offer support for compiling and running CUDA applications on MacOSX.

