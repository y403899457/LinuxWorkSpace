
#include <stdio.h>
#include "CCTest.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	printf("Hello! This is tid:%d\n", i);
}

extern "C"
void cudaHello()
{
	dim3 grid(2);
	dim3 block(32);
	kernel<<<grid, block>>> ();
	cudaDeviceSynchronize();
}
