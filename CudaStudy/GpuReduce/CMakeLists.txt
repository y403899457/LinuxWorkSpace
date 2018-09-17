project (cudareducelib)

find_package (CUDA REQUIRED)

cuda_add_library (cudareducelib cudareduce.cu)
