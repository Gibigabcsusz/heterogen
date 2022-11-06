#include "defs.h"
#include "stdio.h"

// after sort2(a,b,temp): a>=b
#define sort2(a, b, t) \
	(t) = ((a)<(b))?(a):(b); \
	(a)=((a)>(b))?(a):(b); \
	(b)=(t);

void median_filter(int imgHeight, int imgWidth, int imgWidthF, unsigned char *imgSrcExt, unsigned char *imgDst)
{
	unsigned char read[75], temp;

	for (int yout = 0; yout < imgHeight; yout++)
	{
		for (int xout = 0; xout < imgWidth; xout++)
		{
			// filling up "read" with rgb bytes from 5*5 pixels
			for (int rgb=0; rgb < 3; rgb++)
			{
				for (int dy = 0; dy < 5; dy++)
				{
					for (int dx = 0; dx < 5; dx++)
					{
						read[(dy * 5 + dx) * 3 + rgb] = imgSrcExt[((yout+dy)*imgWidthF+xout+dx)*3+rgb];
					}
				}
			}
			// finding medians for r, g and b
			for (int rgb=0; rgb < 3; rgb++)
			{
				for (int p = 1; p < 32; p=p*2)
				{
					for (int k = p; k >=1 ; k/=2)
					{
						for (int j = k % p; j <= 32 - 1 - k; j += 2 * k)
						{
							for (int i = 0; i < k; i++)
							{
								if ((i + j) / (p * 2) == (i + j + k) / (p * 2))
								{
									// not doing comparisons outside of 25 elements
									if (i + j + k < 25)
									{
										sort2(read[(i + j + k) * 3 + rgb], read[(i + j) * 3 + rgb], temp)
									}
								}
							}
						}
					}
				}
			}
			// writing 3 medians to output image
			for (int rgb = 0; rgb < 3; rgb++)
			{
				imgDst[(yout * imgWidth + xout) * 3 + rgb] = read[12 * 3 + rgb];
			}

		}
	}


}
