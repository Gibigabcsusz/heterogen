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
    int CYOIP = L1DID/(36*3); // copy y offset in pixels from global base address
    int CXOIP = (L1DID%(36*3))/3; // copy x offset in pixels from global base address
    int CCO = L1DID%3; // copy channel offset
    int rowstep = (get_local_size(0)*get_local_size(1))/(36*3); // next component to copy is this many rows down
    //int rowstep = 2; // next component to copy is this many rows down
    
    // declare local memory, copy global -> shared (local) memory
    __local float shmem[36][12][3];
    if(L1DID<36*3*rowstep)
    {
        for(int row=0; row<get_local_size(1)+4; row+=rowstep)
        {
            shmem[CXOIP][CYOIP+row][CCO] = (float)(gInput[CGBI + (CYOIP*imgWidthF + CXOIP + row*imgWidthF)*3 + CCO]);
        }
    }

    // wait for other threads to finish copy
    barrier(CLK_LOCAL_MEM_FENCE);

    // choose median for the 3 channels of the given pixel
    float values[25], tmp;
    float medians[3];

    // first result (byte) global index
    int FRGI = ((get_global_id(1))*imgWidth + get_global_id(0)) * 3;

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
        if(values[0]>values[1]) {tmp=values[0]; values[0]=values[1]; values[1]=tmp;}
        if(values[2]>values[3]) {tmp=values[2]; values[2]=values[3]; values[3]=tmp;}
        if(values[4]>values[5]) {tmp=values[4]; values[4]=values[5]; values[5]=tmp;}
        if(values[6]>values[7]) {tmp=values[6]; values[6]=values[7]; values[7]=tmp;}
        if(values[8]>values[9]) {tmp=values[8]; values[8]=values[9]; values[9]=tmp;}
        if(values[10]>values[11]) {tmp=values[10]; values[10]=values[11]; values[11]=tmp;}
        if(values[12]>values[13]) {tmp=values[12]; values[12]=values[13]; values[13]=tmp;}
        if(values[14]>values[15]) {tmp=values[14]; values[14]=values[15]; values[15]=tmp;}
        if(values[16]>values[17]) {tmp=values[16]; values[16]=values[17]; values[17]=tmp;}
        if(values[18]>values[19]) {tmp=values[18]; values[18]=values[19]; values[19]=tmp;}
        if(values[20]>values[21]) {tmp=values[20]; values[20]=values[21]; values[21]=tmp;}
        if(values[22]>values[23]) {tmp=values[22]; values[22]=values[23]; values[23]=tmp;}
        if(values[0]>values[2]) {tmp=values[0]; values[0]=values[2]; values[2]=tmp;}
        if(values[1]>values[3]) {tmp=values[1]; values[1]=values[3]; values[3]=tmp;}
        if(values[4]>values[6]) {tmp=values[4]; values[4]=values[6]; values[6]=tmp;}
        if(values[5]>values[7]) {tmp=values[5]; values[5]=values[7]; values[7]=tmp;}
        if(values[8]>values[10]) {tmp=values[8]; values[8]=values[10]; values[10]=tmp;}
        if(values[9]>values[11]) {tmp=values[9]; values[9]=values[11]; values[11]=tmp;}
        if(values[12]>values[14]) {tmp=values[12]; values[12]=values[14]; values[14]=tmp;}
        if(values[13]>values[15]) {tmp=values[13]; values[13]=values[15]; values[15]=tmp;}
        if(values[16]>values[18]) {tmp=values[16]; values[16]=values[18]; values[18]=tmp;}
        if(values[17]>values[19]) {tmp=values[17]; values[17]=values[19]; values[19]=tmp;}
        if(values[20]>values[22]) {tmp=values[20]; values[20]=values[22]; values[22]=tmp;}
        if(values[21]>values[23]) {tmp=values[21]; values[21]=values[23]; values[23]=tmp;}
        if(values[1]>values[2]) {tmp=values[1]; values[1]=values[2]; values[2]=tmp;}
        if(values[5]>values[6]) {tmp=values[5]; values[5]=values[6]; values[6]=tmp;}
        if(values[9]>values[10]) {tmp=values[9]; values[9]=values[10]; values[10]=tmp;}
        if(values[13]>values[14]) {tmp=values[13]; values[13]=values[14]; values[14]=tmp;}
        if(values[17]>values[18]) {tmp=values[17]; values[17]=values[18]; values[18]=tmp;}
        if(values[21]>values[22]) {tmp=values[21]; values[21]=values[22]; values[22]=tmp;}
        if(values[0]>values[4]) {tmp=values[0]; values[0]=values[4]; values[4]=tmp;}
        if(values[1]>values[5]) {tmp=values[1]; values[1]=values[5]; values[5]=tmp;}
        if(values[2]>values[6]) {tmp=values[2]; values[2]=values[6]; values[6]=tmp;}
        if(values[3]>values[7]) {tmp=values[3]; values[3]=values[7]; values[7]=tmp;}
        if(values[8]>values[12]) {tmp=values[8]; values[8]=values[12]; values[12]=tmp;}
        if(values[9]>values[13]) {tmp=values[9]; values[9]=values[13]; values[13]=tmp;}
        if(values[10]>values[14]) {tmp=values[10]; values[10]=values[14]; values[14]=tmp;}
        if(values[11]>values[15]) {tmp=values[11]; values[11]=values[15]; values[15]=tmp;}
        if(values[16]>values[20]) {tmp=values[16]; values[16]=values[20]; values[20]=tmp;}
        if(values[17]>values[21]) {tmp=values[17]; values[17]=values[21]; values[21]=tmp;}
        if(values[18]>values[22]) {tmp=values[18]; values[18]=values[22]; values[22]=tmp;}
        if(values[19]>values[23]) {tmp=values[19]; values[19]=values[23]; values[23]=tmp;}
        if(values[2]>values[4]) {tmp=values[2]; values[2]=values[4]; values[4]=tmp;}
        if(values[3]>values[5]) {tmp=values[3]; values[3]=values[5]; values[5]=tmp;}
        if(values[10]>values[12]) {tmp=values[10]; values[10]=values[12]; values[12]=tmp;}
        if(values[11]>values[13]) {tmp=values[11]; values[11]=values[13]; values[13]=tmp;}
        if(values[18]>values[20]) {tmp=values[18]; values[18]=values[20]; values[20]=tmp;}
        if(values[19]>values[21]) {tmp=values[19]; values[19]=values[21]; values[21]=tmp;}
        if(values[1]>values[2]) {tmp=values[1]; values[1]=values[2]; values[2]=tmp;}
        if(values[3]>values[4]) {tmp=values[3]; values[3]=values[4]; values[4]=tmp;}
        if(values[5]>values[6]) {tmp=values[5]; values[5]=values[6]; values[6]=tmp;}
        if(values[9]>values[10]) {tmp=values[9]; values[9]=values[10]; values[10]=tmp;}
        if(values[11]>values[12]) {tmp=values[11]; values[11]=values[12]; values[12]=tmp;}
        if(values[13]>values[14]) {tmp=values[13]; values[13]=values[14]; values[14]=tmp;}
        if(values[17]>values[18]) {tmp=values[17]; values[17]=values[18]; values[18]=tmp;}
        if(values[19]>values[20]) {tmp=values[19]; values[19]=values[20]; values[20]=tmp;}
        if(values[21]>values[22]) {tmp=values[21]; values[21]=values[22]; values[22]=tmp;}
        if(values[0]>values[8]) {tmp=values[0]; values[0]=values[8]; values[8]=tmp;}
        if(values[1]>values[9]) {tmp=values[1]; values[1]=values[9]; values[9]=tmp;}
        if(values[2]>values[10]) {tmp=values[2]; values[2]=values[10]; values[10]=tmp;}
        if(values[3]>values[11]) {tmp=values[3]; values[3]=values[11]; values[11]=tmp;}
        if(values[4]>values[12]) {tmp=values[4]; values[4]=values[12]; values[12]=tmp;}
        if(values[5]>values[13]) {tmp=values[5]; values[5]=values[13]; values[13]=tmp;}
        if(values[6]>values[14]) {tmp=values[6]; values[6]=values[14]; values[14]=tmp;}
        if(values[7]>values[15]) {tmp=values[7]; values[7]=values[15]; values[15]=tmp;}
        if(values[16]>values[24]) {tmp=values[16]; values[16]=values[24]; values[24]=tmp;}
        if(values[4]>values[8]) {tmp=values[4]; values[4]=values[8]; values[8]=tmp;}
        if(values[5]>values[9]) {tmp=values[5]; values[5]=values[9]; values[9]=tmp;}
        if(values[6]>values[10]) {tmp=values[6]; values[6]=values[10]; values[10]=tmp;}
        if(values[7]>values[11]) {tmp=values[7]; values[7]=values[11]; values[11]=tmp;}
        if(values[20]>values[24]) {tmp=values[20]; values[20]=values[24]; values[24]=tmp;}
        if(values[2]>values[4]) {tmp=values[2]; values[2]=values[4]; values[4]=tmp;}
        if(values[3]>values[5]) {tmp=values[3]; values[3]=values[5]; values[5]=tmp;}
        if(values[6]>values[8]) {tmp=values[6]; values[6]=values[8]; values[8]=tmp;}
        if(values[7]>values[9]) {tmp=values[7]; values[7]=values[9]; values[9]=tmp;}
        if(values[10]>values[12]) {tmp=values[10]; values[10]=values[12]; values[12]=tmp;}
        if(values[11]>values[13]) {tmp=values[11]; values[11]=values[13]; values[13]=tmp;}
        if(values[18]>values[20]) {tmp=values[18]; values[18]=values[20]; values[20]=tmp;}
        if(values[19]>values[21]) {tmp=values[19]; values[19]=values[21]; values[21]=tmp;}
        if(values[22]>values[24]) {tmp=values[22]; values[22]=values[24]; values[24]=tmp;}
        if(values[1]>values[2]) {tmp=values[1]; values[1]=values[2]; values[2]=tmp;}
        if(values[3]>values[4]) {tmp=values[3]; values[3]=values[4]; values[4]=tmp;}
        if(values[5]>values[6]) {tmp=values[5]; values[5]=values[6]; values[6]=tmp;}
        if(values[7]>values[8]) {tmp=values[7]; values[7]=values[8]; values[8]=tmp;}
        if(values[9]>values[10]) {tmp=values[9]; values[9]=values[10]; values[10]=tmp;}
        if(values[11]>values[12]) {tmp=values[11]; values[11]=values[12]; values[12]=tmp;}
        if(values[13]>values[14]) {tmp=values[13]; values[13]=values[14]; values[14]=tmp;}
        if(values[17]>values[18]) {tmp=values[17]; values[17]=values[18]; values[18]=tmp;}
        if(values[19]>values[20]) {tmp=values[19]; values[19]=values[20]; values[20]=tmp;}
        if(values[21]>values[22]) {tmp=values[21]; values[21]=values[22]; values[22]=tmp;}
        if(values[23]>values[24]) {tmp=values[23]; values[23]=values[24]; values[24]=tmp;}
        if(values[0]>values[16]) {tmp=values[0]; values[0]=values[16]; values[16]=tmp;}
        if(values[1]>values[17]) {tmp=values[1]; values[1]=values[17]; values[17]=tmp;}
        if(values[2]>values[18]) {tmp=values[2]; values[2]=values[18]; values[18]=tmp;}
        if(values[3]>values[19]) {tmp=values[3]; values[3]=values[19]; values[19]=tmp;}
        if(values[4]>values[20]) {tmp=values[4]; values[4]=values[20]; values[20]=tmp;}
        if(values[5]>values[21]) {tmp=values[5]; values[5]=values[21]; values[21]=tmp;}
        if(values[6]>values[22]) {tmp=values[6]; values[6]=values[22]; values[22]=tmp;}
        if(values[7]>values[23]) {tmp=values[7]; values[7]=values[23]; values[23]=tmp;}
        if(values[8]>values[24]) {tmp=values[8]; values[8]=values[24]; values[24]=tmp;}
        if(values[8]>values[16]) {tmp=values[8]; values[8]=values[16]; values[16]=tmp;}
        if(values[9]>values[17]) {tmp=values[9]; values[9]=values[17]; values[17]=tmp;}
        if(values[10]>values[18]) {tmp=values[10]; values[10]=values[18]; values[18]=tmp;}
        if(values[11]>values[19]) {tmp=values[11]; values[11]=values[19]; values[19]=tmp;}
        if(values[12]>values[20]) {tmp=values[12]; values[12]=values[20]; values[20]=tmp;}
        if(values[13]>values[21]) {tmp=values[13]; values[13]=values[21]; values[21]=tmp;}
        // if(values[14]>values[22]) {tmp=values[14]; values[14]=values[22]; values[22]=tmp;}
        // if(values[15]>values[23]) {tmp=values[15]; values[15]=values[23]; values[23]=tmp;}
        // if(values[4]>values[8]) {tmp=values[4]; values[4]=values[8]; values[8]=tmp;}
        // if(values[5]>values[9]) {tmp=values[5]; values[5]=values[9]; values[9]=tmp;}
        if(values[6]>values[10]) {tmp=values[6]; values[6]=values[10]; values[10]=tmp;}
        if(values[7]>values[11]) {tmp=values[7]; values[7]=values[11]; values[11]=tmp;}
        if(values[12]>values[16]) {tmp=values[12]; values[12]=values[16]; values[16]=tmp;}
        if(values[13]>values[17]) {tmp=values[13]; values[13]=values[17]; values[17]=tmp;}
        // if(values[14]>values[18]) {tmp=values[14]; values[14]=values[18]; values[18]=tmp;}
        // if(values[15]>values[19]) {tmp=values[15]; values[15]=values[19]; values[19]=tmp;}
        // if(values[20]>values[24]) {tmp=values[20]; values[20]=values[24]; values[24]=tmp;}
        // if(values[2]>values[4]) {tmp=values[2]; values[2]=values[4]; values[4]=tmp;}
        // if(values[3]>values[5]) {tmp=values[3]; values[3]=values[5]; values[5]=tmp;}
        // if(values[6]>values[8]) {tmp=values[6]; values[6]=values[8]; values[8]=tmp;}
        // if(values[7]>values[9]) {tmp=values[7]; values[7]=values[9]; values[9]=tmp;}
        if(values[10]>values[12]) {tmp=values[10]; values[10]=values[12]; values[12]=tmp;}
        if(values[11]>values[13]) {tmp=values[11]; values[11]=values[13]; values[13]=tmp;}
        // if(values[14]>values[16]) {tmp=values[14]; values[14]=values[16]; values[16]=tmp;}
        // if(values[15]>values[17]) {tmp=values[15]; values[15]=values[17]; values[17]=tmp;}
        // if(values[18]>values[20]) {tmp=values[18]; values[18]=values[20]; values[20]=tmp;}
        // if(values[19]>values[21]) {tmp=values[19]; values[19]=values[21]; values[21]=tmp;}
        // if(values[22]>values[24]) {tmp=values[22]; values[22]=values[24]; values[24]=tmp;}
        // if(values[1]>values[2]) {tmp=values[1]; values[1]=values[2]; values[2]=tmp;}
        // if(values[3]>values[4]) {tmp=values[3]; values[3]=values[4]; values[4]=tmp;}
        // if(values[5]>values[6]) {tmp=values[5]; values[5]=values[6]; values[6]=tmp;}
        // if(values[7]>values[8]) {tmp=values[7]; values[7]=values[8]; values[8]=tmp;}
        // if(values[9]>values[10]) {tmp=values[9]; values[9]=values[10]; values[10]=tmp;}
        if(values[11]>values[12]) {tmp=values[11]; values[11]=values[12]; values[12]=tmp;}
        // if(values[13]>values[14]) {tmp=values[13]; values[13]=values[14]; values[14]=tmp;}
        // if(values[15]>values[16]) {tmp=values[15]; values[15]=values[16]; values[16]=tmp;}
        // if(values[17]>values[18]) {tmp=values[17]; values[17]=values[18]; values[18]=tmp;}
        // if(values[19]>values[20]) {tmp=values[19]; values[19]=values[20]; values[20]=tmp;}
        // if(values[21]>values[22]) {tmp=values[21]; values[21]=values[22]; values[22]=tmp;}
        // if(values[23]>values[24]) {tmp=values[23]; values[23]=values[24]; values[24]=tmp;}

        barrier(CLK_LOCAL_MEM_FENCE);


        gOutput[FRGI+channel] = (unsigned char)(values[12]);
        //medians[channel] = values[12];
    }

    // copy medians to global memory
    //gOutput[FRGI+0] = (unsigned char)(medians[0]);
    //gOutput[FRGI+1] = (unsigned char)(medians[1]);
    //gOutput[FRGI+2] = (unsigned char)(medians[2]);

}
