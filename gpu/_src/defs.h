#define FILTER_W 5
#define FILTER_H 5

#define RUNS 1
#define KERNEL_RUNS 1000

#define FIXED_OCL_DEVICE 0
#define FIXED_OCL_DEVICE_ID 0

#define KERNEL_FILE_NAME ".\\_src\\opencl_kernels.cl"
#define KERNEL_FUNCTION "kernel_median_filter"
#define LOCAL_SIZE_X 32								// workgroup X size: 16 for all kernels, except the last (32)
#define LOCAL_SIZE_Y 8                             // workgroup X size: 16 for all kernels, except the last (8), see kernel codes