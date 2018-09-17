#include "stdio.h"
#include "CCTest.cuh"

extern "C" void cudaHello();

int main()
{
	printf("This is cc main calling cuda func:\n");
	cudaHello();
	return 0;
}
