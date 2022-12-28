/*

CUDA <-> OpenCL

//////////////////////////////////////////////////
    typedef struct cuda_id_struct
    {
        int x;
        int y;
    } cuda_id;

    cuda_id threadIdx, blockIdx, blockDim;
    threadIdx.x = get_local_id(0);
    threadIdx.y = get_local_id(1);
    blockIdx.x  = get_group_id(0);
    blockIdx.y  = get_group_id(1);
    blockDim.x  = get_local_size(0);
    blockDim.y  = get_local_size(1);


    __syncthreads() <->  barrier(CLK_LOCAL_MEM_FENCE);

*/

__kernel void kernel_copy(__global unsigned char* gInput,
                                 __global unsigned char* gOutput,
                                 __constant int *filter_coeffs,
								 int imgWidth,
								 int imgWidthF)
{

  



}


//----------------------------------------------------------------------------------------------------------------------------

__kernel void kernel_conv_global(__global unsigned char* gInput,
                                 __global unsigned char* gOutput,
                                 __constant int *filter_coeffs,
								 int imgWidth,
								 int imgWidthF)
{
 
  



}


//----------------------------------------------------------------------------------------------------------------------------


__kernel void kernel_conv_sh_uchar_int(__global unsigned char* gInput,
                                       __global unsigned char* gOutput,
                                       __constant int *filter_coeffs,
									   int imgWidth,
                                       int imgWidthF)
{
  


}



//----------------------------------------------------------------------------------------------------------------------------
__kernel void kernel_conv_sh_uchar_float(__global unsigned char* gInput,
                                         __global unsigned char* gOutput,
                                         __constant float *filter_coeffs,
										 int imgWidth,
                                         int imgWidthF)
{




}



//----------------------------------------------------------------------------------------------------------------------------
__kernel void kernel_conv_sh_float_float(__global unsigned char* gInput,
                                         __global unsigned char* gOutput,
                                         __constant float *filter_coeffs,
										 int imgWidth,
                                         int imgWidthF)
{



}



//----------------------------------------------------------------------------------------------------------------------------
__kernel void kernel_conv_sh_float_float_nbc(__global unsigned char* gInput,
                                             __global unsigned char* gOutput,
                                             __constant float *filter_coeffs,
											 int imgWidth,
                                             int imgWidthF)
{




}



//----------------------------------------------------------------------------------------------------------------------------
__kernel void kernel_median(__global unsigned char* gInput,
                                             __global unsigned char* gOutput,
                                             __constant float *filter_coeffs,
											 int imgWidth,
                                             int imgWidthF)
{
    // calculate index in global memory for copying (1 byte)
    int CGBI = (get_local_size(1)*get_group_id(1)*imgWidthF + get_local_size(0)*get_group_id(0)) * 3; // global base index
    int L1DID = get_local_id(1)*get_local_size(0) + get_local_id(0); // local 1D index
    
    // calculate index in local memory for copying (1 byte)
    int CYOIP = L1DID/60; // copy y offset in pixels from global base address
    int CXOIP = (L1DID%60)/3; // copy x offset in pixels from global base address
    int CCO = L1DID%3; // copy channel offset
    
    // declare local memory, copy global -> shared (local) memory
    __local unsigned char shmem[20][20][3];
    if(L1DID<240)
    {
        for(int i=0; i<5; i++)
        {
            shmem[CXOIP][CYOIP+i*240][CCO] = gInput[CGBI + CYOIP*imgWidthF*3 + CXOIP*3 + i*4*imgWidthF];
        }
    }

    // choose median for the 3 channels of the given pixel
    unsigned char values[5][5], tmp;
    unsigned char medians[3];
    for(int channel=0; channel<3; channel++)
    {
        // load the appropriate 25 bytes to be sorted
        for(int x=0; x<5; x++)
        {
            for(int y=0; y<5; y++)
            {
                values[x][y] = shmem[x][y][channel];
            }
        }

        // choose the median from the loaded values
            // template: a<b ? {} : {tmp=a; a=b; b=tmp;};
            // template result: a<b

        (values[ 0]<values[ 1]) ? {} : {tmp=values[ 0]; values[ 0]=values[ 1]; values[ 1]=tmp;};
        (values[ 2]<values[ 3]) ? {} : {tmp=values[ 2]; values[ 2]=values[ 3]; values[ 3]=tmp;};
        (values[ 4]<values[ 5]) ? {} : {tmp=values[ 4]; values[ 4]=values[ 5]; values[ 5]=tmp;};
        (values[ 6]<values[ 7]) ? {} : {tmp=values[ 6]; values[ 6]=values[ 7]; values[ 7]=tmp;};
        (values[ 8]<values[ 9]) ? {} : {tmp=values[ 8]; values[ 8]=values[ 9]; values[ 9]=tmp;};
        (values[10]<values[11]) ? {} : {tmp=values[10]; values[10]=values[11]; values[11]=tmp;};
        (values[12]<values[13]) ? {} : {tmp=values[12]; values[12]=values[13]; values[13]=tmp;};
        (values[14]<values[15]) ? {} : {tmp=values[14]; values[14]=values[15]; values[15]=tmp;};
        (values[16]<values[17]) ? {} : {tmp=values[16]; values[16]=values[17]; values[17]=tmp;};
        (values[18]<values[19]) ? {} : {tmp=values[18]; values[18]=values[19]; values[19]=tmp;};
        (values[20]<values[21]) ? {} : {tmp=values[20]; values[20]=values[21]; values[21]=tmp;};
        (values[22]<values[23]) ? {} : {tmp=values[22]; values[22]=values[23]; values[23]=tmp;};
        (values[ 0]<values[ 2]) ? {} : {tmp=values[ 0]; values[ 0]=values[ 2]; values[ 2]=tmp;};
        (values[ 1]<values[ 3]) ? {} : {tmp=values[ 1]; values[ 1]=values[ 3]; values[ 3]=tmp;};
        (values[ 4]<values[ 6]) ? {} : {tmp=values[ 4]; values[ 4]=values[ 6]; values[ 6]=tmp;};
        (values[ 5]<values[ 7]) ? {} : {tmp=values[ 5]; values[ 5]=values[ 7]; values[ 7]=tmp;};
        (values[ 8]<values[10]) ? {} : {tmp=values[ 8]; values[ 8]=values[10]; values[10]=tmp;};
        (values[ 9]<values[11]) ? {} : {tmp=values[ 9]; values[ 9]=values[11]; values[11]=tmp;};
        (values[12]<values[14]) ? {} : {tmp=values[12]; values[12]=values[14]; values[14]=tmp;};
        (values[13]<values[15]) ? {} : {tmp=values[13]; values[13]=values[15]; values[15]=tmp;};
        (values[16]<values[18]) ? {} : {tmp=values[16]; values[16]=values[18]; values[18]=tmp;};
        (values[17]<values[19]) ? {} : {tmp=values[17]; values[17]=values[19]; values[19]=tmp;};
        (values[20]<values[22]) ? {} : {tmp=values[20]; values[20]=values[22]; values[22]=tmp;};
        (values[21]<values[23]) ? {} : {tmp=values[21]; values[21]=values[23]; values[23]=tmp;};
        (values[ 1]<values[ 2]) ? {} : {tmp=values[ 1]; values[ 1]=values[ 2]; values[ 2]=tmp;};
        (values[ 5]<values[ 6]) ? {} : {tmp=values[ 5]; values[ 5]=values[ 6]; values[ 6]=tmp;};
        (values[ 9]<values[10]) ? {} : {tmp=values[ 9]; values[ 9]=values[10]; values[10]=tmp;};
        (values[13]<values[14]) ? {} : {tmp=values[13]; values[13]=values[14]; values[14]=tmp;};
        (values[17]<values[18]) ? {} : {tmp=values[17]; values[17]=values[18]; values[18]=tmp;};
        (values[21]<values[22]) ? {} : {tmp=values[21]; values[21]=values[22]; values[22]=tmp;};
        (values[ 0]<values[ 4]) ? {} : {tmp=values[ 0]; values[ 0]=values[ 4]; values[ 4]=tmp;};
        (values[ 1]<values[ 5]) ? {} : {tmp=values[ 1]; values[ 1]=values[ 5]; values[ 5]=tmp;};
        (values[ 2]<values[ 6]) ? {} : {tmp=values[ 2]; values[ 2]=values[ 6]; values[ 6]=tmp;};
        (values[ 3]<values[ 7]) ? {} : {tmp=values[ 3]; values[ 3]=values[ 7]; values[ 7]=tmp;};
        (values[ 8]<values[12]) ? {} : {tmp=values[ 8]; values[ 8]=values[12]; values[12]=tmp;};
        (values[ 9]<values[13]) ? {} : {tmp=values[ 9]; values[ 9]=values[13]; values[13]=tmp;};
        (values[10]<values[14]) ? {} : {tmp=values[10]; values[10]=values[14]; values[14]=tmp;};
        (values[11]<values[15]) ? {} : {tmp=values[11]; values[11]=values[15]; values[15]=tmp;};
        (values[16]<values[20]) ? {} : {tmp=values[16]; values[16]=values[20]; values[20]=tmp;};
        (values[17]<values[21]) ? {} : {tmp=values[17]; values[17]=values[21]; values[21]=tmp;};
        (values[18]<values[22]) ? {} : {tmp=values[18]; values[18]=values[22]; values[22]=tmp;};
        (values[19]<values[23]) ? {} : {tmp=values[19]; values[19]=values[23]; values[23]=tmp;};
        (values[ 2]<values[ 4]) ? {} : {tmp=values[ 2]; values[ 2]=values[ 4]; values[ 4]=tmp;};
        (values[ 3]<values[ 5]) ? {} : {tmp=values[ 3]; values[ 3]=values[ 5]; values[ 5]=tmp;};
        (values[10]<values[12]) ? {} : {tmp=values[10]; values[10]=values[12]; values[12]=tmp;};
        (values[11]<values[13]) ? {} : {tmp=values[11]; values[11]=values[13]; values[13]=tmp;};
        (values[18]<values[20]) ? {} : {tmp=values[18]; values[18]=values[20]; values[20]=tmp;};
        (values[19]<values[21]) ? {} : {tmp=values[19]; values[19]=values[21]; values[21]=tmp;};
        (values[ 1]<values[ 2]) ? {} : {tmp=values[ 1]; values[ 1]=values[ 2]; values[ 2]=tmp;};
        (values[ 3]<values[ 4]) ? {} : {tmp=values[ 3]; values[ 3]=values[ 4]; values[ 4]=tmp;};
        (values[ 5]<values[ 6]) ? {} : {tmp=values[ 5]; values[ 5]=values[ 6]; values[ 6]=tmp;};
        (values[ 9]<values[10]) ? {} : {tmp=values[ 9]; values[ 9]=values[10]; values[10]=tmp;};
        (values[11]<values[12]) ? {} : {tmp=values[11]; values[11]=values[12]; values[12]=tmp;};
        (values[13]<values[14]) ? {} : {tmp=values[13]; values[13]=values[14]; values[14]=tmp;};
        (values[17]<values[18]) ? {} : {tmp=values[17]; values[17]=values[18]; values[18]=tmp;};
        (values[19]<values[20]) ? {} : {tmp=values[19]; values[19]=values[20]; values[20]=tmp;};
        (values[21]<values[22]) ? {} : {tmp=values[21]; values[21]=values[22]; values[22]=tmp;};
        (values[ 0]<values[ 8]) ? {} : {tmp=values[ 0]; values[ 0]=values[ 8]; values[ 8]=tmp;};
        (values[ 1]<values[ 9]) ? {} : {tmp=values[ 1]; values[ 1]=values[ 9]; values[ 9]=tmp;};
        (values[ 2]<values[10]) ? {} : {tmp=values[ 2]; values[ 2]=values[10]; values[10]=tmp;};
        (values[ 3]<values[11]) ? {} : {tmp=values[ 3]; values[ 3]=values[11]; values[11]=tmp;};
        (values[ 4]<values[12]) ? {} : {tmp=values[ 4]; values[ 4]=values[12]; values[12]=tmp;};
        (values[ 5]<values[13]) ? {} : {tmp=values[ 5]; values[ 5]=values[13]; values[13]=tmp;};
        (values[ 6]<values[14]) ? {} : {tmp=values[ 6]; values[ 6]=values[14]; values[14]=tmp;};
        (values[ 7]<values[15]) ? {} : {tmp=values[ 7]; values[ 7]=values[15]; values[15]=tmp;};
        (values[16]<values[24]) ? {} : {tmp=values[16]; values[16]=values[24]; values[24]=tmp;};
        (values[ 4]<values[ 8]) ? {} : {tmp=values[ 4]; values[ 4]=values[ 8]; values[ 8]=tmp;};
        (values[ 5]<values[ 9]) ? {} : {tmp=values[ 5]; values[ 5]=values[ 9]; values[ 9]=tmp;};
        (values[ 6]<values[10]) ? {} : {tmp=values[ 6]; values[ 6]=values[10]; values[10]=tmp;};
        (values[ 7]<values[11]) ? {} : {tmp=values[ 7]; values[ 7]=values[11]; values[11]=tmp;};
        (values[20]<values[24]) ? {} : {tmp=values[20]; values[20]=values[24]; values[24]=tmp;};
        (values[ 2]<values[ 4]) ? {} : {tmp=values[ 2]; values[ 2]=values[ 4]; values[ 4]=tmp;};
        (values[ 3]<values[ 5]) ? {} : {tmp=values[ 3]; values[ 3]=values[ 5]; values[ 5]=tmp;};
        (values[ 6]<values[ 8]) ? {} : {tmp=values[ 6]; values[ 6]=values[ 8]; values[ 8]=tmp;};
        (values[ 7]<values[ 9]) ? {} : {tmp=values[ 7]; values[ 7]=values[ 9]; values[ 9]=tmp;};
        (values[10]<values[12]) ? {} : {tmp=values[10]; values[10]=values[12]; values[12]=tmp;};
        (values[11]<values[13]) ? {} : {tmp=values[11]; values[11]=values[13]; values[13]=tmp;};
        (values[18]<values[20]) ? {} : {tmp=values[18]; values[18]=values[20]; values[20]=tmp;};
        (values[19]<values[21]) ? {} : {tmp=values[19]; values[19]=values[21]; values[21]=tmp;};
        (values[22]<values[24]) ? {} : {tmp=values[22]; values[22]=values[24]; values[24]=tmp;};
        (values[ 1]<values[ 2]) ? {} : {tmp=values[ 1]; values[ 1]=values[ 2]; values[ 2]=tmp;};
        (values[ 3]<values[ 4]) ? {} : {tmp=values[ 3]; values[ 3]=values[ 4]; values[ 4]=tmp;};
        (values[ 5]<values[ 6]) ? {} : {tmp=values[ 5]; values[ 5]=values[ 6]; values[ 6]=tmp;};
        (values[ 7]<values[ 8]) ? {} : {tmp=values[ 7]; values[ 7]=values[ 8]; values[ 8]=tmp;};
        (values[ 9]<values[10]) ? {} : {tmp=values[ 9]; values[ 9]=values[10]; values[10]=tmp;};
        (values[11]<values[12]) ? {} : {tmp=values[11]; values[11]=values[12]; values[12]=tmp;};
        (values[13]<values[14]) ? {} : {tmp=values[13]; values[13]=values[14]; values[14]=tmp;};
        (values[17]<values[18]) ? {} : {tmp=values[17]; values[17]=values[18]; values[18]=tmp;};
        (values[19]<values[20]) ? {} : {tmp=values[19]; values[19]=values[20]; values[20]=tmp;};
        (values[21]<values[22]) ? {} : {tmp=values[21]; values[21]=values[22]; values[22]=tmp;};
        (values[23]<values[24]) ? {} : {tmp=values[23]; values[23]=values[24]; values[24]=tmp;};
        (values[ 0]<values[16]) ? {} : {tmp=values[ 0]; values[ 0]=values[16]; values[16]=tmp;};
        (values[ 1]<values[17]) ? {} : {tmp=values[ 1]; values[ 1]=values[17]; values[17]=tmp;};
        (values[ 2]<values[18]) ? {} : {tmp=values[ 2]; values[ 2]=values[18]; values[18]=tmp;};
        (values[ 3]<values[19]) ? {} : {tmp=values[ 3]; values[ 3]=values[19]; values[19]=tmp;};
        (values[ 4]<values[20]) ? {} : {tmp=values[ 4]; values[ 4]=values[20]; values[20]=tmp;};
        (values[ 5]<values[21]) ? {} : {tmp=values[ 5]; values[ 5]=values[21]; values[21]=tmp;};
        (values[ 6]<values[22]) ? {} : {tmp=values[ 6]; values[ 6]=values[22]; values[22]=tmp;};
        (values[ 7]<values[23]) ? {} : {tmp=values[ 7]; values[ 7]=values[23]; values[23]=tmp;};
        (values[ 8]<values[24]) ? {} : {tmp=values[ 8]; values[ 8]=values[24]; values[24]=tmp;};
        (values[ 8]<values[16]) ? {} : {tmp=values[ 8]; values[ 8]=values[16]; values[16]=tmp;};
        (values[ 9]<values[17]) ? {} : {tmp=values[ 9]; values[ 9]=values[17]; values[17]=tmp;};
        (values[10]<values[18]) ? {} : {tmp=values[10]; values[10]=values[18]; values[18]=tmp;};
        (values[11]<values[19]) ? {} : {tmp=values[11]; values[11]=values[19]; values[19]=tmp;};
        (values[12]<values[20]) ? {} : {tmp=values[12]; values[12]=values[20]; values[20]=tmp;};
        (values[13]<values[21]) ? {} : {tmp=values[13]; values[13]=values[21]; values[21]=tmp;};
//        (values[14]<values[22]) ? {} : {tmp=values[14]; values[14]=values[22]; values[22]=tmp;};
//        (values[15]<values[23]) ? {} : {tmp=values[15]; values[15]=values[23]; values[23]=tmp;};
//        (values[ 4]<values[ 8]) ? {} : {tmp=values[ 4]; values[ 4]=values[ 8]; values[ 8]=tmp;};
//        (values[ 5]<values[ 9]) ? {} : {tmp=values[ 5]; values[ 5]=values[ 9]; values[ 9]=tmp;};
        (values[ 6]<values[10]) ? {} : {tmp=values[ 6]; values[ 6]=values[10]; values[10]=tmp;};
        (values[ 7]<values[11]) ? {} : {tmp=values[ 7]; values[ 7]=values[11]; values[11]=tmp;};
        (values[12]<values[16]) ? {} : {tmp=values[12]; values[12]=values[16]; values[16]=tmp;};
        (values[13]<values[17]) ? {} : {tmp=values[13]; values[13]=values[17]; values[17]=tmp;};
//        (values[14]<values[18]) ? {} : {tmp=values[14]; values[14]=values[18]; values[18]=tmp;};
//        (values[15]<values[19]) ? {} : {tmp=values[15]; values[15]=values[19]; values[19]=tmp;};
//        (values[20]<values[24]) ? {} : {tmp=values[20]; values[20]=values[24]; values[24]=tmp;};
//        (values[ 2]<values[ 4]) ? {} : {tmp=values[ 2]; values[ 2]=values[ 4]; values[ 4]=tmp;};
//        (values[ 3]<values[ 5]) ? {} : {tmp=values[ 3]; values[ 3]=values[ 5]; values[ 5]=tmp;};
//        (values[ 6]<values[ 8]) ? {} : {tmp=values[ 6]; values[ 6]=values[ 8]; values[ 8]=tmp;};
//        (values[ 7]<values[ 9]) ? {} : {tmp=values[ 7]; values[ 7]=values[ 9]; values[ 9]=tmp;};
        (values[10]<values[12]) ? {} : {tmp=values[10]; values[10]=values[12]; values[12]=tmp;};
        (values[11]<values[13]) ? {} : {tmp=values[11]; values[11]=values[13]; values[13]=tmp;};
//        (values[14]<values[16]) ? {} : {tmp=values[14]; values[14]=values[16]; values[16]=tmp;};
//        (values[15]<values[17]) ? {} : {tmp=values[15]; values[15]=values[17]; values[17]=tmp;};
//        (values[18]<values[20]) ? {} : {tmp=values[18]; values[18]=values[20]; values[20]=tmp;};
//        (values[19]<values[21]) ? {} : {tmp=values[19]; values[19]=values[21]; values[21]=tmp;};
//        (values[22]<values[24]) ? {} : {tmp=values[22]; values[22]=values[24]; values[24]=tmp;};
//        (values[ 1]<values[ 2]) ? {} : {tmp=values[ 1]; values[ 1]=values[ 2]; values[ 2]=tmp;};
//        (values[ 3]<values[ 4]) ? {} : {tmp=values[ 3]; values[ 3]=values[ 4]; values[ 4]=tmp;};
//        (values[ 5]<values[ 6]) ? {} : {tmp=values[ 5]; values[ 5]=values[ 6]; values[ 6]=tmp;};
//        (values[ 7]<values[ 8]) ? {} : {tmp=values[ 7]; values[ 7]=values[ 8]; values[ 8]=tmp;};
//        (values[ 9]<values[10]) ? {} : {tmp=values[ 9]; values[ 9]=values[10]; values[10]=tmp;};
        (values[11]<values[12]) ? {} : {tmp=values[11]; values[11]=values[12]; values[12]=tmp;};
//        (values[13]<values[14]) ? {} : {tmp=values[13]; values[13]=values[14]; values[14]=tmp;};
//        (values[15]<values[16]) ? {} : {tmp=values[15]; values[15]=values[16]; values[16]=tmp;};
//        (values[17]<values[18]) ? {} : {tmp=values[17]; values[17]=values[18]; values[18]=tmp;};
//        (values[19]<values[20]) ? {} : {tmp=values[19]; values[19]=values[20]; values[20]=tmp;};
//        (values[21]<values[22]) ? {} : {tmp=values[21]; values[21]=values[22]; values[22]=tmp;};
//        (values[23]<values[24]) ? {} : {tmp=values[23]; values[23]=values[24]; values[24]=tmp;};

        medians[channel] = values[12];
    }

    // first result (byte) global index
    int FRGI = ((get_local_size(1)*get_group_id(1) + get_local_id(1))*imgWidth + get_local_size(0)*get_group_id(0) + get_local_id(0)) * 3;

    // copy medians to global memory
    gOutput[FRGI+0] = medians[0];
    gOutput[FRGI+1] = medians[1];
    gOutput[FRGI+2] = medians[2];
}
