#include <stdio.h>
#include "mylib.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void kernel_hello()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	printf("This is thread %d\n", i);
}

extern "C"
void cudaHello()
{
	dim3 grid(2);
	dim3 block(32);
	kernel_hello<<<grid, block>>>();
	cudaDeviceSynchronize();
}
