#include "cudareduce.cuh"
#include <stdio.h>
#include <time.h>
#define BLOCK_SIZE 512


__global__ void kernel_reduce(int *a, int *result, int length)
{
	int tid = threadIdx.x;
	int IDx = tid + blockIdx.x * blockDim.x;
	if (IDx >= length) return;
	
	int *idata = a + blockIdx.x * blockDim.x;
	for (int step = 2; step <= blockDim.x; step *= 2)
	{
		if (tid % step == 0)
			idata[tid] += idata[tid + step/2];
		__syncthreads();
	}
	if (tid == 0) result[blockIdx.x] = idata[tid];
}

__global__ void kernel_reduce2(int *a, int *result, int length)
{
	int tid = threadIdx.x;
	int IDx = tid + blockIdx.x * blockDim.x;
	if (IDx >= length) return;
	
	int *idata = a + blockIdx.x * blockDim.x;
	for (int step = 2; step <= blockDim.x; step *= 2)
	{
		int index = step * tid;
		if (index < blockDim.x)
			idata[index] += idata[index + step/2];
		__syncthreads();
	}
	if (tid == 0) result[blockIdx.x] = idata[tid];
}

__global__ void kernel_reduce3(int *a, int *result, int length)
{
	int tid = threadIdx.x;
	int IDx = tid + blockIdx.x * blockDim.x;
	if (IDx >= length) return;
	
	int *idata = a + blockIdx.x * blockDim.x;
	for (int step = blockDim.x; step > 1; step /= 2)
	{
		if (tid < step/2)
			idata[tid] += idata[tid + step/2];
		__syncthreads();
	}
	if (tid == 0) result[blockIdx.x] = idata[tid];
}

__global__ void kernel_reduce4(int *a, int *result, int length)
{
	__shared__ int sdata[BLOCK_SIZE];
	int tid = threadIdx.x;
	int IDx = tid + blockIdx.x * blockDim.x;
	if (IDx >= length) return;
	
	sdata[tid] = a[IDx];
	__syncthreads();

	for (int step = blockDim.x; step > 1; step /= 2)
	{
		if (tid < step/2)
			sdata[tid] += sdata[tid + step/2];
		__syncthreads();
	}
	if (tid == 0) result[blockIdx.x] = sdata[tid];
}

__inline__ __device__ int warpreduce(int Mysum)
{
	Mysum += __shfl_xor_sync(0xffffffff, Mysum, 16);
	Mysum += __shfl_xor_sync(0xffffffff, Mysum, 8);
	Mysum += __shfl_xor_sync(0xffffffff, Mysum, 4);
	Mysum += __shfl_xor_sync(0xffffffff, Mysum, 2);
	Mysum += __shfl_xor_sync(0xffffffff, Mysum, 1);
	return Mysum;
}

__inline__ __device__ int warpreduceNor(int Mysum, int warpId)
{
	Mysum += __shfl_sync(0xffffffff, Mysum, warpId + 16);
	Mysum += __shfl_sync(0xffffffff, Mysum, warpId + 8);
	Mysum += __shfl_sync(0xffffffff, Mysum, warpId + 4);
	Mysum += __shfl_sync(0xffffffff, Mysum, warpId + 2);
	Mysum += __shfl_sync(0xffffffff, Mysum, warpId + 1);
	return Mysum;
}

__global__ void kernel_reduce5(int *a, int *result, int length)
{
	__shared__ int sdata[BLOCK_SIZE / 32];
	int tid = threadIdx.x;
	int IDx = tid + blockIdx.x * blockDim.x;
	if (IDx >= length) return;
	
	int Mysum = a[IDx];
	int LaneIdx = threadIdx.x / 32;
	int WarpIdx = threadIdx.x % 32;
	Mysum = warpreduce(Mysum);
	if (WarpIdx == 0) sdata[LaneIdx] = Mysum;
	__syncthreads();

	if (threadIdx.x < (BLOCK_SIZE / 32))
		Mysum = sdata[threadIdx.x];
	else
		Mysum = 0;
	Mysum = warpreduce(Mysum);

	if (tid == 0) result[blockIdx.x] = Mysum;
}

__global__ void kernel_reduce6(int *a, int *result, int length)
{
	__shared__ int sdata[BLOCK_SIZE / 32];
	int tid = threadIdx.x;
	int IDx = tid + blockIdx.x * blockDim.x;
	if (IDx >= length) return;
	
	int Mysum = a[IDx];
	int LaneIdx = threadIdx.x / 32;
	int WarpIdx = threadIdx.x % 32;

	Mysum = warpreduceNor(Mysum, WarpIdx);
	if (WarpIdx == 0) sdata[LaneIdx] = Mysum;
	__syncthreads();

	if (threadIdx.x < (BLOCK_SIZE / 32))
		Mysum = sdata[threadIdx.x];
	else
		Mysum = 0;
	Mysum = warpreduce(Mysum);

	if (tid == 0) result[blockIdx.x] = Mysum;
}

void cuda_reduce(int *a, int length)
{
	cudaFree(0);
	int *dev_a, *dev_result;
	cudaMalloc((void**)&dev_a, length * sizeof(int));
	cudaMallocManaged((void**)&dev_result, length/BLOCK_SIZE * sizeof(int));
	cudaMemcpy(dev_a, a, length * sizeof(int), cudaMemcpyHostToDevice);
	dim3 grid(length / BLOCK_SIZE);
	dim3 block(BLOCK_SIZE);
	clock_t start = clock();
	kernel_reduce6 <<<grid, block>>> (dev_a, dev_result, length);
	cudaDeviceSynchronize();
	clock_t end = clock();
	int sum = 0;
	for (int i = 0; i < length/BLOCK_SIZE; i++)
		sum += dev_result[i];
	printf("Sum = %d\n", sum);
	printf("Time = %fms\n", double(end - start) / (CLOCKS_PER_SEC / 1000));

	cudaFree(dev_a);
	cudaFree(dev_result);
}
