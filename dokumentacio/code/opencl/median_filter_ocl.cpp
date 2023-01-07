// OCLTest1.cpp : Defines the entry point for the console application.
//

#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#include "time.h"

#include "CL\cl.h"

#include "defs.h"
#include "func.h"

////////////////////////////////////////////////////
// SEE defs.h for kernel selection!!!


const char *getErrorString(cl_int error)
{
	switch (error){
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}

#define MAX_PROG_SIZE 65536

void median_filter_ocl(int imgHeight, int imgWidth, int imgHeightF, int imgWidthF,
	int imgFOfssetH, int imgFOfssetW,
	unsigned char *imgSrc, unsigned char *imgDst)
{
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	clock_t s0, e0;
	double d0;

	int size_in;
	size_in = imgHeightF*imgWidthF * 3;
	int size_out;
	size_out = imgHeight*imgWidth * 3;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Init OpenCL

	/* Get Platform and Device Info */
	ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	cl_platform_id *platforms;
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)* ret_num_platforms);
	ret = clGetPlatformIDs(ret_num_platforms, platforms, &ret_num_platforms);
	
	int num_devices_all = 0;
	for (int platform_id = 0; platform_id < ret_num_platforms; platform_id++)
	{
		ret = clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices);
		num_devices_all = num_devices_all + ret_num_devices;
	}
	cl_device_id *devices;
	int device_offset = 0;
	devices = (cl_device_id*)malloc(sizeof(cl_device_id)* num_devices_all);
	for (int platform_id = 0; platform_id < ret_num_platforms; platform_id++)
	{
		ret = clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices);
		ret = clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, ret_num_devices, &devices[device_offset], &ret_num_devices);
		device_offset = device_offset + ret_num_devices;
	}

	char cBuffer[1024];
	for (int device_num = 0; device_num < num_devices_all; device_num++)
	{
		printf("Device id: %d,	", device_num);

		ret = clGetDeviceInfo(devices[device_num], CL_DEVICE_VENDOR, sizeof(cBuffer), &cBuffer, NULL);
		printf("%s ", cBuffer);

		ret = clGetDeviceInfo(devices[device_num], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
		printf("%s\r\n", cBuffer);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Select device to be used
#if FIXED_OCL_DEVICE == 0
	printf("\n\nSelect OpenCL device and press enter:");
	int device_sel = getchar()-0x30;
	device_id = devices[device_sel];
#else	
	device_id = devices[FIXED_OCL_DEVICE_ID];
	free(devices);
#endif

	// Load the source code containing the kernel
	const char *kernel_source="__kernel void kernel_median_filter(__global unsigned char* gInput,\n\
											 __global unsigned char* gOutput,\n\
											 int imgWidth,\n\
											 int imgWidthF)\n\
{\n\
	// calculate index in global memory for copying (1 byte)\n\
	int BI = (get_local_size(1)*get_group_id(1)*imgWidthF + get_local_size(0)*get_group_id(0)) * 3; // global base index\n\
	int L1DID = get_local_id(1)*get_local_size(0) + get_local_id(0); // local 1D index\n\
	\n\
	// calculate index in local memory for copying (1 byte)\n\
	int CYOIP = L1DID/(20*3); // copy y offset in pixels from global base address\n\
	int CXOIP = (L1DID%(20*3))/3; // copy x offset in pixels from global base address\n\
	int CCO = L1DID%3; // copy channel offset\n\
	int rowstep = (get_local_size(0)*get_local_size(1))/(20*3); // next component to copy is this many rows down\n\
	\n\
	// declare local memory, copy global -> shared (local) memory\n\
	// shared memory padded with one dummy channel per pixel plus two dummy channels (1 bank) per row at the ends\n\
	__local half shmem[20*4+2][20];\n\
	if(L1DID<20*3*rowstep)\n\
	{\n\
		for(int row=0; row<get_local_size(1)+4; row+=rowstep)\n\
		{\n\
			shmem[CXOIP*4+CCO][CYOIP+row] = (half)(gInput[BI + (CYOIP*imgWidthF + CXOIP + row*imgWidthF)*3 + CCO]);\n\
		}\n\
	}\n\
\n\
	// wait for other threads to finish copy\n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
	// choose median for the 3 channels of the given pixel\n\
	half tmp;\n\
	half r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24;\n\
	\n\
	// first result (byte) global index\n\
	BI = ((get_global_id(1))*imgWidth + get_global_id(0)) * 3;\n\
\n\
	for(int channel=0; channel<3; channel++)\n\
	{\n\
		// load the appropriate 25 values to be sorted\n\
		r00=shmem[(get_local_id(0)+0)*4+channel][get_local_id(1)+0];\n\
		r01=shmem[(get_local_id(0)+1)*4+channel][get_local_id(1)+0];\n\
		r02=shmem[(get_local_id(0)+2)*4+channel][get_local_id(1)+0];\n\
		r03=shmem[(get_local_id(0)+3)*4+channel][get_local_id(1)+0];\n\
		r04=shmem[(get_local_id(0)+4)*4+channel][get_local_id(1)+0];\n\
		r05=shmem[(get_local_id(0)+0)*4+channel][get_local_id(1)+1];\n\
		r06=shmem[(get_local_id(0)+1)*4+channel][get_local_id(1)+1];\n\
		r07=shmem[(get_local_id(0)+2)*4+channel][get_local_id(1)+1];\n\
		r08=shmem[(get_local_id(0)+3)*4+channel][get_local_id(1)+1];\n\
		r09=shmem[(get_local_id(0)+4)*4+channel][get_local_id(1)+1];\n\
		r10=shmem[(get_local_id(0)+0)*4+channel][get_local_id(1)+2];\n\
		r11=shmem[(get_local_id(0)+1)*4+channel][get_local_id(1)+2];\n\
		r12=shmem[(get_local_id(0)+2)*4+channel][get_local_id(1)+2];\n\
		r13=shmem[(get_local_id(0)+3)*4+channel][get_local_id(1)+2];\n\
		r14=shmem[(get_local_id(0)+4)*4+channel][get_local_id(1)+2];\n\
		r15=shmem[(get_local_id(0)+0)*4+channel][get_local_id(1)+3];\n\
		r16=shmem[(get_local_id(0)+1)*4+channel][get_local_id(1)+3];\n\
		r17=shmem[(get_local_id(0)+2)*4+channel][get_local_id(1)+3];\n\
		r18=shmem[(get_local_id(0)+3)*4+channel][get_local_id(1)+3];\n\
		r19=shmem[(get_local_id(0)+4)*4+channel][get_local_id(1)+3];\n\
		r20=shmem[(get_local_id(0)+0)*4+channel][get_local_id(1)+4];\n\
		r21=shmem[(get_local_id(0)+1)*4+channel][get_local_id(1)+4];\n\
		r22=shmem[(get_local_id(0)+2)*4+channel][get_local_id(1)+4];\n\
		r23=shmem[(get_local_id(0)+3)*4+channel][get_local_id(1)+4];\n\
		r24=shmem[(get_local_id(0)+4)*4+channel][get_local_id(1)+4];\n\
\n\
		// find the median, will be in r12\n\
		tmp=fmax(r00,r01); r00=fmin(r00,r01); r01=tmp;\n\
		tmp=fmax(r02,r03); r02=fmin(r02,r03); r03=tmp;\n\
		tmp=fmax(r04,r05); r04=fmin(r04,r05); r05=tmp;\n\
		tmp=fmax(r06,r07); r06=fmin(r06,r07); r07=tmp;\n\
		tmp=fmax(r08,r09); r08=fmin(r08,r09); r09=tmp;\n\
		tmp=fmax(r10,r11); r10=fmin(r10,r11); r11=tmp;\n\
		tmp=fmax(r12,r13); r12=fmin(r12,r13); r13=tmp;\n\
		tmp=fmax(r14,r15); r14=fmin(r14,r15); r15=tmp;\n\
		tmp=fmax(r16,r17); r16=fmin(r16,r17); r17=tmp;\n\
		tmp=fmax(r18,r19); r18=fmin(r18,r19); r19=tmp;\n\
		tmp=fmax(r20,r21); r20=fmin(r20,r21); r21=tmp;\n\
		tmp=fmax(r22,r23); r22=fmin(r22,r23); r23=tmp;\n\
		tmp=fmax(r00,r02); r00=fmin(r00,r02); r02=tmp;\n\
		tmp=fmax(r01,r03); r01=fmin(r01,r03); r03=tmp;\n\
		tmp=fmax(r04,r06); r04=fmin(r04,r06); r06=tmp;\n\
		tmp=fmax(r05,r07); r05=fmin(r05,r07); r07=tmp;\n\
		tmp=fmax(r08,r10); r08=fmin(r08,r10); r10=tmp;\n\
		tmp=fmax(r09,r11); r09=fmin(r09,r11); r11=tmp;\n\
		tmp=fmax(r12,r14); r12=fmin(r12,r14); r14=tmp;\n\
		tmp=fmax(r13,r15); r13=fmin(r13,r15); r15=tmp;\n\
		tmp=fmax(r16,r18); r16=fmin(r16,r18); r18=tmp;\n\
		tmp=fmax(r17,r19); r17=fmin(r17,r19); r19=tmp;\n\
		tmp=fmax(r20,r22); r20=fmin(r20,r22); r22=tmp;\n\
		tmp=fmax(r21,r23); r21=fmin(r21,r23); r23=tmp;\n\
		tmp=fmax(r01,r02); r01=fmin(r01,r02); r02=tmp;\n\
		tmp=fmax(r05,r06); r05=fmin(r05,r06); r06=tmp;\n\
		tmp=fmax(r09,r10); r09=fmin(r09,r10); r10=tmp;\n\
		tmp=fmax(r13,r14); r13=fmin(r13,r14); r14=tmp;\n\
		tmp=fmax(r17,r18); r17=fmin(r17,r18); r18=tmp;\n\
		tmp=fmax(r21,r22); r21=fmin(r21,r22); r22=tmp;\n\
		tmp=fmax(r00,r04); r00=fmin(r00,r04); r04=tmp;\n\
		tmp=fmax(r01,r05); r01=fmin(r01,r05); r05=tmp;\n\
		tmp=fmax(r02,r06); r02=fmin(r02,r06); r06=tmp;\n\
		tmp=fmax(r03,r07); r03=fmin(r03,r07); r07=tmp;\n\
		tmp=fmax(r08,r12); r08=fmin(r08,r12); r12=tmp;\n\
		tmp=fmax(r09,r13); r09=fmin(r09,r13); r13=tmp;\n\
		tmp=fmax(r10,r14); r10=fmin(r10,r14); r14=tmp;\n\
		tmp=fmax(r11,r15); r11=fmin(r11,r15); r15=tmp;\n\
		tmp=fmax(r16,r20); r16=fmin(r16,r20); r20=tmp;\n\
		tmp=fmax(r17,r21); r17=fmin(r17,r21); r21=tmp;\n\
		tmp=fmax(r18,r22); r18=fmin(r18,r22); r22=tmp;\n\
		tmp=fmax(r19,r23); r19=fmin(r19,r23); r23=tmp;\n\
		tmp=fmax(r02,r04); r02=fmin(r02,r04); r04=tmp;\n\
		tmp=fmax(r03,r05); r03=fmin(r03,r05); r05=tmp;\n\
		tmp=fmax(r10,r12); r10=fmin(r10,r12); r12=tmp;\n\
		tmp=fmax(r11,r13); r11=fmin(r11,r13); r13=tmp;\n\
		tmp=fmax(r18,r20); r18=fmin(r18,r20); r20=tmp;\n\
		tmp=fmax(r19,r21); r19=fmin(r19,r21); r21=tmp;\n\
		tmp=fmax(r01,r02); r01=fmin(r01,r02); r02=tmp;\n\
		tmp=fmax(r03,r04); r03=fmin(r03,r04); r04=tmp;\n\
		tmp=fmax(r05,r06); r05=fmin(r05,r06); r06=tmp;\n\
		tmp=fmax(r09,r10); r09=fmin(r09,r10); r10=tmp;\n\
		tmp=fmax(r11,r12); r11=fmin(r11,r12); r12=tmp;\n\
		tmp=fmax(r13,r14); r13=fmin(r13,r14); r14=tmp;\n\
		tmp=fmax(r17,r18); r17=fmin(r17,r18); r18=tmp;\n\
		tmp=fmax(r19,r20); r19=fmin(r19,r20); r20=tmp;\n\
		tmp=fmax(r21,r22); r21=fmin(r21,r22); r22=tmp;\n\
		tmp=fmax(r00,r08); r00=fmin(r00,r08); r08=tmp;\n\
		tmp=fmax(r01,r09); r01=fmin(r01,r09); r09=tmp;\n\
		tmp=fmax(r02,r10); r02=fmin(r02,r10); r10=tmp;\n\
		tmp=fmax(r03,r11); r03=fmin(r03,r11); r11=tmp;\n\
		tmp=fmax(r04,r12); r04=fmin(r04,r12); r12=tmp;\n\
		tmp=fmax(r05,r13); r05=fmin(r05,r13); r13=tmp;\n\
		tmp=fmax(r06,r14); r06=fmin(r06,r14); r14=tmp;\n\
		tmp=fmax(r07,r15); r07=fmin(r07,r15); r15=tmp;\n\
		tmp=fmax(r16,r24); r16=fmin(r16,r24); r24=tmp;\n\
		tmp=fmax(r04,r08); r04=fmin(r04,r08); r08=tmp;\n\
		tmp=fmax(r05,r09); r05=fmin(r05,r09); r09=tmp;\n\
		tmp=fmax(r06,r10); r06=fmin(r06,r10); r10=tmp;\n\
		tmp=fmax(r07,r11); r07=fmin(r07,r11); r11=tmp;\n\
		tmp=fmax(r20,r24); r20=fmin(r20,r24); r24=tmp;\n\
		tmp=fmax(r02,r04); r02=fmin(r02,r04); r04=tmp;\n\
		tmp=fmax(r03,r05); r03=fmin(r03,r05); r05=tmp;\n\
		tmp=fmax(r06,r08); r06=fmin(r06,r08); r08=tmp;\n\
		tmp=fmax(r07,r09); r07=fmin(r07,r09); r09=tmp;\n\
		tmp=fmax(r10,r12); r10=fmin(r10,r12); r12=tmp;\n\
		tmp=fmax(r11,r13); r11=fmin(r11,r13); r13=tmp;\n\
		tmp=fmax(r18,r20); r18=fmin(r18,r20); r20=tmp;\n\
		tmp=fmax(r19,r21); r19=fmin(r19,r21); r21=tmp;\n\
		tmp=fmax(r22,r24); r22=fmin(r22,r24); r24=tmp;\n\
		tmp=fmax(r01,r02); r01=fmin(r01,r02); r02=tmp;\n\
		tmp=fmax(r03,r04); r03=fmin(r03,r04); r04=tmp;\n\
		tmp=fmax(r05,r06); r05=fmin(r05,r06); r06=tmp;\n\
		tmp=fmax(r07,r08); r07=fmin(r07,r08); r08=tmp;\n\
		tmp=fmax(r09,r10); r09=fmin(r09,r10); r10=tmp;\n\
		tmp=fmax(r11,r12); r11=fmin(r11,r12); r12=tmp;\n\
		tmp=fmax(r13,r14); r13=fmin(r13,r14); r14=tmp;\n\
		tmp=fmax(r17,r18); r17=fmin(r17,r18); r18=tmp;\n\
		tmp=fmax(r19,r20); r19=fmin(r19,r20); r20=tmp;\n\
		tmp=fmax(r21,r22); r21=fmin(r21,r22); r22=tmp;\n\
		tmp=fmax(r23,r24); r23=fmin(r23,r24); r24=tmp;\n\
		tmp=fmax(r00,r16); r00=fmin(r00,r16); r16=tmp;\n\
		tmp=fmax(r01,r17); r01=fmin(r01,r17); r17=tmp;\n\
		tmp=fmax(r02,r18); r02=fmin(r02,r18); r18=tmp;\n\
		tmp=fmax(r03,r19); r03=fmin(r03,r19); r19=tmp;\n\
		tmp=fmax(r04,r20); r04=fmin(r04,r20); r20=tmp;\n\
		tmp=fmax(r05,r21); r05=fmin(r05,r21); r21=tmp;\n\
		tmp=fmax(r06,r22); r06=fmin(r06,r22); r22=tmp;\n\
		tmp=fmax(r07,r23); r07=fmin(r07,r23); r23=tmp;\n\
		tmp=fmax(r08,r24); r08=fmin(r08,r24); r24=tmp;\n\
		tmp=fmax(r08,r16); r08=fmin(r08,r16); r16=tmp;\n\
		tmp=fmax(r09,r17); r09=fmin(r09,r17); r17=tmp;\n\
		tmp=fmax(r10,r18); r10=fmin(r10,r18); r18=tmp;\n\
		tmp=fmax(r11,r19); r11=fmin(r11,r19); r19=tmp;\n\
		tmp=fmax(r12,r20); r12=fmin(r12,r20); r20=tmp;\n\
		tmp=fmax(r13,r21); r13=fmin(r13,r21); r21=tmp;\n\
		// tmp=fmax(r14,r22); r14=fmin(r14,r22); r22=tmp;\n\
		// tmp=fmax(r15,r23); r15=fmin(r15,r23); r23=tmp;\n\
		// tmp=fmax(r04,r08); r04=fmin(r04,r08); r08=tmp;\n\
		// tmp=fmax(r05,r09); r05=fmin(r05,r09); r09=tmp;\n\
		tmp=fmax(r06,r10); r06=fmin(r06,r10); r10=tmp;\n\
		tmp=fmax(r07,r11); r07=fmin(r07,r11); r11=tmp;\n\
		tmp=fmax(r12,r16); r12=fmin(r12,r16); r16=tmp;\n\
		tmp=fmax(r13,r17); r13=fmin(r13,r17); r17=tmp;\n\
		// tmp=fmax(r14,r18); r14=fmin(r14,r18); r18=tmp;\n\
		// tmp=fmax(r15,r19); r15=fmin(r15,r19); r19=tmp;\n\
		// tmp=fmax(r20,r24); r20=fmin(r20,r24); r24=tmp;\n\
		// tmp=fmax(r02,r04); r02=fmin(r02,r04); r04=tmp;\n\
		// tmp=fmax(r03,r05); r03=fmin(r03,r05); r05=tmp;\n\
		// tmp=fmax(r06,r08); r06=fmin(r06,r08); r08=tmp;\n\
		// tmp=fmax(r07,r09); r07=fmin(r07,r09); r09=tmp;\n\
		tmp=fmax(r10,r12); r10=fmin(r10,r12); r12=tmp;\n\
		tmp=fmax(r11,r13); r11=fmin(r11,r13); r13=tmp;\n\
		// tmp=fmax(r14,r16); r14=fmin(r14,r16); r16=tmp;\n\
		// tmp=fmax(r15,r17); r15=fmin(r15,r17); r17=tmp;\n\
		// tmp=fmax(r18,r20); r18=fmin(r18,r20); r20=tmp;\n\
		// tmp=fmax(r19,r21); r19=fmin(r19,r21); r21=tmp;\n\
		// tmp=fmax(r22,r24); r22=fmin(r22,r24); r24=tmp;\n\
		// tmp=fmax(r01,r02); r01=fmin(r01,r02); r02=tmp;\n\
		// tmp=fmax(r03,r04); r03=fmin(r03,r04); r04=tmp;\n\
		// tmp=fmax(r05,r06); r05=fmin(r05,r06); r06=tmp;\n\
		// tmp=fmax(r07,r08); r07=fmin(r07,r08); r08=tmp;\n\
		// tmp=fmax(r09,r10); r09=fmin(r09,r10); r10=tmp;\n\
		tmp=fmax(r11,r12); r11=fmin(r11,r12); r12=tmp;\n\
		// tmp=fmax(r13,r14); r13=fmin(r13,r14); r14=tmp;\n\
		// tmp=fmax(r15,r16); r15=fmin(r15,r16); r16=tmp;\n\
		// tmp=fmax(r17,r18); r17=fmin(r17,r18); r18=tmp;\n\
		// tmp=fmax(r19,r20); r19=fmin(r19,r20); r20=tmp;\n\
		// tmp=fmax(r21,r22); r21=fmin(r21,r22); r22=tmp;\n\
		// tmp=fmax(r23,r24); r23=fmin(r23,r24); r24=tmp;\n\
\n\
		// copy medians to global memory\n\
		gOutput[BI+channel] = (unsigned char)(r12);\n\
	}\n\
}\n\
";
	size_t kernel_size=strlen(kernel_source);
	//
	//FILE *kernel_file;
	//char fileName[] = KERNEL_FILE_NAME;

	//fopen_s(&kernel_file, fileName, "r");
	//if (kernel_file == NULL) {
	//	fprintf(stderr, "Failed to read kernel from file.\n");
	//	exit(1);
	//}
	//fseek(kernel_file, 0, SEEK_END);
	//kernel_size = ftell(kernel_file);
	//rewind(kernel_file);
	//kernel_source = (char *)malloc(kernel_size + 1);
	//kernel_source[kernel_size] = '\0';
	//int read = fread(kernel_source, sizeof(char), kernel_size, kernel_file);
	//if (read != kernel_size) {
	//	fprintf(stderr, "Error while reading the kernel.\n");
	//	exit(1);
	//}
	//fclose(kernel_file);

	/* Create OpenCL context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	/* Create Command Queue */
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

	/* Create Memory Buffer on device*/
	cl_mem device_imgSrc, device_imgDst, device_coeffs;
	device_imgSrc = clCreateBuffer(context, CL_MEM_READ_ONLY, size_in, NULL, &ret);
	device_imgDst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_out, NULL, &ret);


	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source,
		(const size_t *)&kernel_size, &ret);

	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	size_t param_value_size, param_value_size_ret;

	if (ret != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		cl_build_status bldstatus;
		printf("\nError %d: Failed to build program executable [ %s ]\n", ret, getErrorString(ret));
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(bldstatus), (void *)&bldstatus, &len);
		printf("Build Status %d: %s\n", ret, getErrorString(ret));
		printf("INFO: %s\n", getErrorString(bldstatus));
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer), buffer, &len);
		printf("Build Options %d: %s\n", ret, getErrorString(ret));
		printf("INFO: %s\n", buffer);
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("Build Log %d: %s\n", ret, getErrorString(ret));
		printf("%s\n", buffer);
		exit(1);
	}


	/* Create OpenCL Kernel */
	kernel = clCreateKernel(program, KERNEL_FUNCTION, &ret);

	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(device_imgSrc), (void *)&device_imgSrc);
	ret = clSetKernelArg(kernel, 1, sizeof(device_imgDst), (void *)&device_imgDst);
	ret = clSetKernelArg(kernel, 2, sizeof(int), &imgWidth);
	ret = clSetKernelArg(kernel, 3, sizeof(int), &imgWidthF);


	// Copy input data to device memory
	ret = clEnqueueWriteBuffer(command_queue, device_imgSrc, CL_TRUE, 0,
		size_in, imgSrc, 0, NULL, NULL);


	clFinish(command_queue);
	
	/* Execute OpenCL Kernel */
	size_t local_size[] = { LOCAL_SIZE_X, LOCAL_SIZE_Y };
	size_t global_size[] = { imgWidth, imgHeight };

	time_measure(1);

	cl_event event[1024];
	for (int runs = 0; runs < KERNEL_RUNS; runs++)
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &event[runs]);

	if (ret != CL_SUCCESS)
	{
		printf("\nError %d: Failed to build program executable [ %s ]\n", ret, getErrorString(ret));
		exit(1);
	}

	clWaitForEvents(1, &event[KERNEL_RUNS - 1]);

	double runtime = time_measure(2);

	cl_ulong time_start, time_end;
	double total_time;
	clGetEventProfilingInfo(event[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event[KERNEL_RUNS-1], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("Total kernel time = %6.4f ms, # of runs: %d\r\n", (total_time / (1000000.0)), KERNEL_RUNS);
	printf("Average single kernel time = %6.4f ms\r\n", (total_time / (1000000.0*KERNEL_RUNS)));
	double mpixel = (KERNEL_RUNS * 1000.0 * double(imgWidth*imgHeight) / (total_time / (1000000.0))) / 1000000;
	printf("Single run MPixel/s: %4.4f\r\n", mpixel);
	printf("Meas time: %6.4f ms\r\n", (total_time/1000000.0));



	/* Copy results from the memory buffer */
	ret = clEnqueueReadBuffer(command_queue, device_imgDst, CL_TRUE, 0,
		size_out, imgDst, 0, NULL, NULL);


	/* Finalization */
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	//free(kernel_source);

}
