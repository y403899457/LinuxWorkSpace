#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C"
void cuda_reduce(int *a, int length);
