//----------------------------------------------------------------------------------------------------------------------------
__kernel void kernel_median_filter(__global unsigned char* gInput,
                                             __global unsigned char* gOutput,
											 int imgWidth,
                                             int imgWidthF)
{
    // calculate index in global memory for copying (1 byte)
    int BI = (get_local_size(1)*get_group_id(1)*imgWidthF + get_local_size(0)*get_group_id(0)) * 3; // global base index
    int L1DID = get_local_id(1)*get_local_size(0) + get_local_id(0); // local 1D index
    
    // calculate index in local memory for copying (1 byte)
    int CYOIP = L1DID/(36*3); // copy y offset in pixels from global base address
    int CXOIP = (L1DID%(36*3))/3; // copy x offset in pixels from global base address
    int CCO = L1DID%3; // copy channel offset
    int rowstep = (get_local_size(0)*get_local_size(1))/(36*3); // next component to copy is this many rows down
    
    // declare local memory, copy global -> shared (local) memory
    __local half shmem[36][12][3];
    if(L1DID<36*3*rowstep)
    {
        for(int row=0; row<get_local_size(1)+4; row+=rowstep)
        {
            shmem[CXOIP][CYOIP+row][CCO] = (float)(gInput[BI + (CYOIP*imgWidthF + CXOIP + row*imgWidthF)*3 + CCO]);
        }
    }

    // wait for other threads to finish copy
    barrier(CLK_LOCAL_MEM_FENCE);

    // choose median for the 3 channels of the given pixel
    half tmp;
    half r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24;
    
    // first result (byte) global index
    BI = ((get_global_id(1))*imgWidth + get_global_id(0)) * 3;

    for(int channel=0; channel<3; channel++)
    {
        // load the appropriate 25 values to be sorted
        // for(int i=0; x<25; i++)
        // {
            // for(int y=0; y<5; y++)
            // {
                // values[x+y*5] = shmem[get_local_id(0)+x][get_local_id(1)+y][channel];
            // }
        // }
        r00=shmem[get_local_id(0)+0][get_local_id(1)+0][channel];
        r01=shmem[get_local_id(0)+1][get_local_id(1)+0][channel];
        r02=shmem[get_local_id(0)+2][get_local_id(1)+0][channel];
        r03=shmem[get_local_id(0)+3][get_local_id(1)+0][channel];
        r04=shmem[get_local_id(0)+4][get_local_id(1)+0][channel];
        r05=shmem[get_local_id(0)+0][get_local_id(1)+1][channel];
        r06=shmem[get_local_id(0)+1][get_local_id(1)+1][channel];
        r07=shmem[get_local_id(0)+2][get_local_id(1)+1][channel];
        r08=shmem[get_local_id(0)+3][get_local_id(1)+1][channel];
        r09=shmem[get_local_id(0)+4][get_local_id(1)+1][channel];
        r10=shmem[get_local_id(0)+0][get_local_id(1)+2][channel];
        r11=shmem[get_local_id(0)+1][get_local_id(1)+2][channel];
        r12=shmem[get_local_id(0)+2][get_local_id(1)+2][channel];
        r13=shmem[get_local_id(0)+3][get_local_id(1)+2][channel];
        r14=shmem[get_local_id(0)+4][get_local_id(1)+2][channel];
        r15=shmem[get_local_id(0)+0][get_local_id(1)+3][channel];
        r16=shmem[get_local_id(0)+1][get_local_id(1)+3][channel];
        r17=shmem[get_local_id(0)+2][get_local_id(1)+3][channel];
        r18=shmem[get_local_id(0)+3][get_local_id(1)+3][channel];
        r19=shmem[get_local_id(0)+4][get_local_id(1)+3][channel];
        r20=shmem[get_local_id(0)+0][get_local_id(1)+4][channel];
        r21=shmem[get_local_id(0)+1][get_local_id(1)+4][channel];
        r22=shmem[get_local_id(0)+2][get_local_id(1)+4][channel];
        r23=shmem[get_local_id(0)+3][get_local_id(1)+4][channel];
        r24=shmem[get_local_id(0)+4][get_local_id(1)+4][channel];

        // find the median, will be in r12
        tmp=fmax(r00,r01); r00=fmin(r00,r01); r01=tmp;
        tmp=fmax(r02,r03); r02=fmin(r02,r03); r03=tmp;
        tmp=fmax(r04,r05); r04=fmin(r04,r05); r05=tmp;
        tmp=fmax(r06,r07); r06=fmin(r06,r07); r07=tmp;
        tmp=fmax(r08,r09); r08=fmin(r08,r09); r09=tmp;
        tmp=fmax(r10,r11); r10=fmin(r10,r11); r11=tmp;
        tmp=fmax(r12,r13); r12=fmin(r12,r13); r13=tmp;
        tmp=fmax(r14,r15); r14=fmin(r14,r15); r15=tmp;
        tmp=fmax(r16,r17); r16=fmin(r16,r17); r17=tmp;
        tmp=fmax(r18,r19); r18=fmin(r18,r19); r19=tmp;
        tmp=fmax(r20,r21); r20=fmin(r20,r21); r21=tmp;
        tmp=fmax(r22,r23); r22=fmin(r22,r23); r23=tmp;
        tmp=fmax(r00,r02); r00=fmin(r00,r02); r02=tmp;
        tmp=fmax(r01,r03); r01=fmin(r01,r03); r03=tmp;
        tmp=fmax(r04,r06); r04=fmin(r04,r06); r06=tmp;
        tmp=fmax(r05,r07); r05=fmin(r05,r07); r07=tmp;
        tmp=fmax(r08,r10); r08=fmin(r08,r10); r10=tmp;
        tmp=fmax(r09,r11); r09=fmin(r09,r11); r11=tmp;
        tmp=fmax(r12,r14); r12=fmin(r12,r14); r14=tmp;
        tmp=fmax(r13,r15); r13=fmin(r13,r15); r15=tmp;
        tmp=fmax(r16,r18); r16=fmin(r16,r18); r18=tmp;
        tmp=fmax(r17,r19); r17=fmin(r17,r19); r19=tmp;
        tmp=fmax(r20,r22); r20=fmin(r20,r22); r22=tmp;
        tmp=fmax(r21,r23); r21=fmin(r21,r23); r23=tmp;
        tmp=fmax(r01,r02); r01=fmin(r01,r02); r02=tmp;
        tmp=fmax(r05,r06); r05=fmin(r05,r06); r06=tmp;
        tmp=fmax(r09,r10); r09=fmin(r09,r10); r10=tmp;
        tmp=fmax(r13,r14); r13=fmin(r13,r14); r14=tmp;
        tmp=fmax(r17,r18); r17=fmin(r17,r18); r18=tmp;
        tmp=fmax(r21,r22); r21=fmin(r21,r22); r22=tmp;
        tmp=fmax(r00,r04); r00=fmin(r00,r04); r04=tmp;
        tmp=fmax(r01,r05); r01=fmin(r01,r05); r05=tmp;
        tmp=fmax(r02,r06); r02=fmin(r02,r06); r06=tmp;
        tmp=fmax(r03,r07); r03=fmin(r03,r07); r07=tmp;
        tmp=fmax(r08,r12); r08=fmin(r08,r12); r12=tmp;
        tmp=fmax(r09,r13); r09=fmin(r09,r13); r13=tmp;
        tmp=fmax(r10,r14); r10=fmin(r10,r14); r14=tmp;
        tmp=fmax(r11,r15); r11=fmin(r11,r15); r15=tmp;
        tmp=fmax(r16,r20); r16=fmin(r16,r20); r20=tmp;
        tmp=fmax(r17,r21); r17=fmin(r17,r21); r21=tmp;
        tmp=fmax(r18,r22); r18=fmin(r18,r22); r22=tmp;
        tmp=fmax(r19,r23); r19=fmin(r19,r23); r23=tmp;
        tmp=fmax(r02,r04); r02=fmin(r02,r04); r04=tmp;
        tmp=fmax(r03,r05); r03=fmin(r03,r05); r05=tmp;
        tmp=fmax(r10,r12); r10=fmin(r10,r12); r12=tmp;
        tmp=fmax(r11,r13); r11=fmin(r11,r13); r13=tmp;
        tmp=fmax(r18,r20); r18=fmin(r18,r20); r20=tmp;
        tmp=fmax(r19,r21); r19=fmin(r19,r21); r21=tmp;
        tmp=fmax(r01,r02); r01=fmin(r01,r02); r02=tmp;
        tmp=fmax(r03,r04); r03=fmin(r03,r04); r04=tmp;
        tmp=fmax(r05,r06); r05=fmin(r05,r06); r06=tmp;
        tmp=fmax(r09,r10); r09=fmin(r09,r10); r10=tmp;
        tmp=fmax(r11,r12); r11=fmin(r11,r12); r12=tmp;
        tmp=fmax(r13,r14); r13=fmin(r13,r14); r14=tmp;
        tmp=fmax(r17,r18); r17=fmin(r17,r18); r18=tmp;
        tmp=fmax(r19,r20); r19=fmin(r19,r20); r20=tmp;
        tmp=fmax(r21,r22); r21=fmin(r21,r22); r22=tmp;
        tmp=fmax(r00,r08); r00=fmin(r00,r08); r08=tmp;
        tmp=fmax(r01,r09); r01=fmin(r01,r09); r09=tmp;
        tmp=fmax(r02,r10); r02=fmin(r02,r10); r10=tmp;
        tmp=fmax(r03,r11); r03=fmin(r03,r11); r11=tmp;
        tmp=fmax(r04,r12); r04=fmin(r04,r12); r12=tmp;
        tmp=fmax(r05,r13); r05=fmin(r05,r13); r13=tmp;
        tmp=fmax(r06,r14); r06=fmin(r06,r14); r14=tmp;
        tmp=fmax(r07,r15); r07=fmin(r07,r15); r15=tmp;
        tmp=fmax(r16,r24); r16=fmin(r16,r24); r24=tmp;
        tmp=fmax(r04,r08); r04=fmin(r04,r08); r08=tmp;
        tmp=fmax(r05,r09); r05=fmin(r05,r09); r09=tmp;
        tmp=fmax(r06,r10); r06=fmin(r06,r10); r10=tmp;
        tmp=fmax(r07,r11); r07=fmin(r07,r11); r11=tmp;
        tmp=fmax(r20,r24); r20=fmin(r20,r24); r24=tmp;
        tmp=fmax(r02,r04); r02=fmin(r02,r04); r04=tmp;
        tmp=fmax(r03,r05); r03=fmin(r03,r05); r05=tmp;
        tmp=fmax(r06,r08); r06=fmin(r06,r08); r08=tmp;
        tmp=fmax(r07,r09); r07=fmin(r07,r09); r09=tmp;
        tmp=fmax(r10,r12); r10=fmin(r10,r12); r12=tmp;
        tmp=fmax(r11,r13); r11=fmin(r11,r13); r13=tmp;
        tmp=fmax(r18,r20); r18=fmin(r18,r20); r20=tmp;
        tmp=fmax(r19,r21); r19=fmin(r19,r21); r21=tmp;
        tmp=fmax(r22,r24); r22=fmin(r22,r24); r24=tmp;
        tmp=fmax(r01,r02); r01=fmin(r01,r02); r02=tmp;
        tmp=fmax(r03,r04); r03=fmin(r03,r04); r04=tmp;
        tmp=fmax(r05,r06); r05=fmin(r05,r06); r06=tmp;
        tmp=fmax(r07,r08); r07=fmin(r07,r08); r08=tmp;
        tmp=fmax(r09,r10); r09=fmin(r09,r10); r10=tmp;
        tmp=fmax(r11,r12); r11=fmin(r11,r12); r12=tmp;
        tmp=fmax(r13,r14); r13=fmin(r13,r14); r14=tmp;
        tmp=fmax(r17,r18); r17=fmin(r17,r18); r18=tmp;
        tmp=fmax(r19,r20); r19=fmin(r19,r20); r20=tmp;
        tmp=fmax(r21,r22); r21=fmin(r21,r22); r22=tmp;
        tmp=fmax(r23,r24); r23=fmin(r23,r24); r24=tmp;
        tmp=fmax(r00,r16); r00=fmin(r00,r16); r16=tmp;
        tmp=fmax(r01,r17); r01=fmin(r01,r17); r17=tmp;
        tmp=fmax(r02,r18); r02=fmin(r02,r18); r18=tmp;
        tmp=fmax(r03,r19); r03=fmin(r03,r19); r19=tmp;
        tmp=fmax(r04,r20); r04=fmin(r04,r20); r20=tmp;
        tmp=fmax(r05,r21); r05=fmin(r05,r21); r21=tmp;
        tmp=fmax(r06,r22); r06=fmin(r06,r22); r22=tmp;
        tmp=fmax(r07,r23); r07=fmin(r07,r23); r23=tmp;
        tmp=fmax(r08,r24); r08=fmin(r08,r24); r24=tmp;
        tmp=fmax(r08,r16); r08=fmin(r08,r16); r16=tmp;
        tmp=fmax(r09,r17); r09=fmin(r09,r17); r17=tmp;
        tmp=fmax(r10,r18); r10=fmin(r10,r18); r18=tmp;
        tmp=fmax(r11,r19); r11=fmin(r11,r19); r19=tmp;
        tmp=fmax(r12,r20); r12=fmin(r12,r20); r20=tmp;
        tmp=fmax(r13,r21); r13=fmin(r13,r21); r21=tmp;
        // tmp=fmax(r14,r22); r14=fmin(r14,r22); r22=tmp;
        // tmp=fmax(r15,r23); r15=fmin(r15,r23); r23=tmp;
        // tmp=fmax(r04,r08); r04=fmin(r04,r08); r08=tmp;
        // tmp=fmax(r05,r09); r05=fmin(r05,r09); r09=tmp;
        tmp=fmax(r06,r10); r06=fmin(r06,r10); r10=tmp;
        tmp=fmax(r07,r11); r07=fmin(r07,r11); r11=tmp;
        tmp=fmax(r12,r16); r12=fmin(r12,r16); r16=tmp;
        tmp=fmax(r13,r17); r13=fmin(r13,r17); r17=tmp;
        // tmp=fmax(r14,r18); r14=fmin(r14,r18); r18=tmp;
        // tmp=fmax(r15,r19); r15=fmin(r15,r19); r19=tmp;
        // tmp=fmax(r20,r24); r20=fmin(r20,r24); r24=tmp;
        // tmp=fmax(r02,r04); r02=fmin(r02,r04); r04=tmp;
        // tmp=fmax(r03,r05); r03=fmin(r03,r05); r05=tmp;
        // tmp=fmax(r06,r08); r06=fmin(r06,r08); r08=tmp;
        // tmp=fmax(r07,r09); r07=fmin(r07,r09); r09=tmp;
        tmp=fmax(r10,r12); r10=fmin(r10,r12); r12=tmp;
        tmp=fmax(r11,r13); r11=fmin(r11,r13); r13=tmp;
        // tmp=fmax(r14,r16); r14=fmin(r14,r16); r16=tmp;
        // tmp=fmax(r15,r17); r15=fmin(r15,r17); r17=tmp;
        // tmp=fmax(r18,r20); r18=fmin(r18,r20); r20=tmp;
        // tmp=fmax(r19,r21); r19=fmin(r19,r21); r21=tmp;
        // tmp=fmax(r22,r24); r22=fmin(r22,r24); r24=tmp;
        // tmp=fmax(r01,r02); r01=fmin(r01,r02); r02=tmp;
        // tmp=fmax(r03,r04); r03=fmin(r03,r04); r04=tmp;
        // tmp=fmax(r05,r06); r05=fmin(r05,r06); r06=tmp;
        // tmp=fmax(r07,r08); r07=fmin(r07,r08); r08=tmp;
        // tmp=fmax(r09,r10); r09=fmin(r09,r10); r10=tmp;
        tmp=fmax(r11,r12); r11=fmin(r11,r12); r12=tmp;
        // tmp=fmax(r13,r14); r13=fmin(r13,r14); r14=tmp;
        // tmp=fmax(r15,r16); r15=fmin(r15,r16); r16=tmp;
        // tmp=fmax(r17,r18); r17=fmin(r17,r18); r18=tmp;
        // tmp=fmax(r19,r20); r19=fmin(r19,r20); r20=tmp;
        // tmp=fmax(r21,r22); r21=fmin(r21,r22); r22=tmp;
        // tmp=fmax(r23,r24); r23=fmin(r23,r24); r24=tmp;

        // copy medians to global memory
        gOutput[BI+channel] = (unsigned char)(r12);
    }

    //barrier(CLK_LOCAL_MEM_FENCE);
    //gOutput[BI+0] = (unsigned char)(medians[0]);
    //gOutput[BI+1] = (unsigned char)(medians[1]);
    //gOutput[BI+2] = (unsigned char)(medians[2]);

}
