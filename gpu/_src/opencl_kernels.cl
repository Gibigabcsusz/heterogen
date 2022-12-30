//----------------------------------------------------------------------------------------------------------------------------
__kernel void kernel_median_filter(__global unsigned char* gInput,
                                             __global unsigned char* gOutput,
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
            shmem[CXOIP][CYOIP+i*4][CCO] = gInput[CGBI + (CYOIP*imgWidthF + CXOIP + i*4*imgWidthF)*3 + CCO];
        }
    }

    // wait for other threads to finish copy
    barrier(CLK_LOCAL_MEM_FENCE);

    // choose median for the 3 channels of the given pixel
    unsigned char values[25], tmp;
    unsigned char medians[3];
    for(int channel=0; channel<3; channel++)
    {
        // load the appropriate 25 bytes to be sorted
        for(int x=0; x<5; x++)
        {
            for(int y=0; y<5; y++)
            {
                values[x+y*5] = shmem[get_local_id(0)+x][get_local_id(1)+y][channel];
            }
        }

        

        medians[channel] = values[12];
    }
    // first result (byte) global index
    int FRGI = ((get_global_id(1))*imgWidth + get_global_id(0)) * 3;

    // copy medians to global memory
    gOutput[FRGI+0] = medians[0];
    gOutput[FRGI+1] = medians[1];
    gOutput[FRGI+2] = medians[2];

}
