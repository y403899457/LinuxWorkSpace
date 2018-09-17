#include <stdio.h>

extern"C"
void cudaHello();

int main()
{
	printf("This is main proc\n");
	cudaHello();
	return 0;
}
