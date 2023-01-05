#include <stdio.h>

int main(void)
{
    int n=32;
    for(int p=1; p<n; p*=2)
    {
        for(int k=p; k>=1; k/=2)
        {
            for(int j=k%p; j<=n-k-1; j+=2*k)
            {
                for(int i=0; i<k; i++)
                {
                    if((i+j)/(p*2)==(i+j+k)/(p*2) && i+j+k<25)
                        //a<b ? {} : {tmp=a; a=b; b=tmp;};
                        //printf("if(values[%d]>values[%d]) {tmp=values[%d]; values[%d]=values[%d]; values[%d]=tmp;}\n", i+j, i+j+k, i+j, i+j, i+j+k, i+j+k);
                        printf("tmp=fmax(values[%d],values[%d]); values[%d]=fmin(values[%d],values[%d]); values[%d]=tmp;\n", i+j, i+j+k, i+j, i+j, i+j+k, i+j+k);
                }
            }
        }
    }
    return 0;
}
