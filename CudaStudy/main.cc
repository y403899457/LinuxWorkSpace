#include <stdio.h>
#include <malloc.h>

extern "C" void cuda_reduce(int *a, int length);

int main(int argc, char **argv)
{
	int N = 1 << 24;
	printf("Total array size: %d\n", N);
	int *a = (int *) malloc (N * sizeof(int));
	for (int i = 0; i < N; i++)
		a[i] = 1;
	cuda_reduce(a, N);
	free(a);
	return 0;
}
