#include "defs.h"
#include "stdio.h"
#include "omp.h"

#include <IL/ilut.h>
#include <IL/ilu.h>

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"

// after sort2(a,b,temp): a>=b
#define sort2(a, b, temp) \
	(temp) = _mm256_min_epu8 ((a), (b)); \
	(a) = _mm256_max_epu8 ((a), (b)); \
	(b) = (temp);

// after sort8(...): d0>=d1>=d2>=...>=d7
#define sort8(d0,d1,d2,d3,d4,d5,d6,d7,temp) \
	/*sort 4 pairs*/\
	sort2(d0, d1, temp) \
	sort2(d2, d3, temp) \
	sort2(d4, d5, temp) \
	sort2(d6, d7, temp) \
	/*merge 4 pairs into 2 groups of 4 sorted elements*/\
	sort2(d0, d2, temp) \
	sort2(d1, d3, temp) \
	sort2(d4, d6, temp) \
	sort2(d5, d7, temp) \
	sort2(d1, d2, temp) \
	sort2(d5, d6, temp) \
	/*merge groups of 4 to sorted group of 8*/\
	sort2(d0, d4, temp) \
	sort2(d1, d5, temp) \
	sort2(d2, d6, temp) \
	sort2(d3, d7, temp) \
	sort2(d2, d4, temp) \
	sort2(d3, d5, temp) \
	sort2(d1, d2, temp) \
	sort2(d3, d4, temp) \
	sort2(d5, d6, temp)


void median_filter_avx_omp(int imgHeight, int imgWidth, int imgWidthF, unsigned char *imgSrcExt, unsigned char *imgDst)
{
	// r00...r14 are used to store image data, tmp is only used as temporary register during sorting
	register __m256i r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, tmp;
	int y_out, x_rgb_out, thr_num;



	#pragma omp parallel private( y_out, x_rgb_out, thr_num, \
		r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, tmp) \
		shared(imgHeight, imgWidth, imgWidthF, imgSrcExt, imgDst)
	{
		// array to temporarily store register contents when registers need to be freed
		__m256i* arr = (__m256i*)_mm_malloc(25 * 32, 32);



		#pragma omp for nowait
		for (y_out = 0; y_out < imgHeight; y_out++)
		{
			for (x_rgb_out = 0; x_rgb_out < imgWidth * 3; x_rgb_out += 32)
			{
				/* load and sort the first and second 8 long group of values*/

				// load the first 8 values (A00...A07 into regs r00...r07)
				r00 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 0) * 3 + x_rgb_out + 0 * 3));
				r01 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 0) * 3 + x_rgb_out + 1 * 3));
				r02 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 0) * 3 + x_rgb_out + 2 * 3));
				r03 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 0) * 3 + x_rgb_out + 3 * 3));
				r04 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 0) * 3 + x_rgb_out + 4 * 3));
				r05 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 1) * 3 + x_rgb_out + 0 * 3));
				r06 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 1) * 3 + x_rgb_out + 1 * 3));
				r07 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 1) * 3 + x_rgb_out + 2 * 3));
				// values:		A00, A01, A02, A03, A04, A05, A06, A07, xxx, xxx, xxx, xxx, xxx, xxx, xxx
				// registers:	r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14

				// sort first 8 values
				sort8(r00, r01, r02, r03, r04, r05, r06, r07, tmp)

					// store A07 in arr and load next 8 values
					// (load A08...A15 into regs r07...r14)
					_mm256_store_si256(arr + 7, r07);
				r07 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 1) * 3 + x_rgb_out + 3 * 3));
				r08 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 1) * 3 + x_rgb_out + 4 * 3));
				r09 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 2) * 3 + x_rgb_out + 0 * 3));
				r10 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 2) * 3 + x_rgb_out + 1 * 3));
				r11 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 2) * 3 + x_rgb_out + 2 * 3));
				r12 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 2) * 3 + x_rgb_out + 3 * 3));
				r13 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 2) * 3 + x_rgb_out + 4 * 3));
				r14 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 3) * 3 + x_rgb_out + 0 * 3));
				// values:		A00, A01, A02, A03, A04, A05, A06, A08, A09, A10, A11, A12, A13, A14, A15
				// registers:	r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14

				// sort the next 8 values
				sort8(r07, r08, r09, r10, r11, r12, r13, r14, tmp)



                /* merge the first and secont sorted 8 long groups */

                // sort A00 and A08
                sort2(r00, r07, tmp)

                // store A00 from r00 and load A07 to r00
                _mm256_store_si256(arr + 0, r00);
				r00 = _mm256_load_si256(arr + 7);
				// values:		A07, A01, A02, A03, A04, A05, A06, A08, A09, A10, A11, A12, A13, A14, A15
				// registers:	r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14

				// sort value pairs		(A01,A09), ... , (A06,A14), (A07,A15)
				// sort register pairs	(r01,r08), ... , (r06,r13), (r00,r14)
				sort2(r01, r08, tmp)
                sort2(r02, r09, tmp)
                sort2(r03, r10, tmp)
                sort2(r04, r11, tmp)
                sort2(r05, r12, tmp)
                sort2(r06, r13, tmp)
                sort2(r00, r14, tmp)

                // sort value pairs		(A04,A08), (A05,A09), (A06,A10), (A07,A11)
                // sort register pairs	(r04,r07), (A05,A08), (r06,r09), (r00,r10)
                sort2(r04, r07, tmp)
                sort2(r05, r08, tmp)
                sort2(r06, r09, tmp)
                sort2(r00, r10, tmp)

                // sort value pairs		(A02,A04), (A03,A05), (A06,A08), (A07,A09), (A10,A12), (A11,A13)
                // sort register pairs	(r02,r04), (r03,r05), (r06,r07), (r00,r08), (r09,r11), (r10,r12)
                sort2(r02, r04, tmp)
                sort2(r03, r05, tmp)
                sort2(r06, r07, tmp)
                sort2(r00, r08, tmp)
                sort2(r09, r11, tmp)
                sort2(r10, r12, tmp)

                // sort value pairs		(A01,A02), (A03,A04), (A05,A06), (A07,A08), (A09,A10), (A11,A12), (A13,A14)
                // sort register pairs	(r01,r02), (r03,r04), (r05,r06), (r00,r07), (r08,r09), (r10,r11), (r12,r13)
                sort2(r01, r02, tmp)
                sort2(r03, r04, tmp)
                sort2(r05, r06, tmp)
                sort2(r00, r07, tmp)
                sort2(r08, r09, tmp)
                sort2(r10, r11, tmp)
                sort2(r12, r13, tmp)



                /* load and sort the last 9 values */

                // store values A07...A13 (in regs r0,r7,...,r12)
                // values in r13 and r14 don't need to be stored, these regs can be reused
                _mm256_store_si256(arr + 7, r00);
				_mm256_store_si256(arr + 8, r07);
				_mm256_store_si256(arr + 9, r08);
				_mm256_store_si256(arr + 10, r09);
				_mm256_store_si256(arr + 11, r10);
				_mm256_store_si256(arr + 12, r11);
				_mm256_store_si256(arr + 13, r12);
				// values:		xxx, A01, A02, A03, A04, A05, A06, xxx, xxx, xxx, xxx, xxx, xxx, xxx, xxx
				// registers:	r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14

				// load values A16...A24 to regs r07...r14,r00
				r07 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 3) * 3 + x_rgb_out + 1 * 3));
				r08 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 3) * 3 + x_rgb_out + 2 * 3));
				r09 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 3) * 3 + x_rgb_out + 3 * 3));
				r10 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 3) * 3 + x_rgb_out + 4 * 3));
				r11 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 4) * 3 + x_rgb_out + 0 * 3));
				r12 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 4) * 3 + x_rgb_out + 1 * 3));
				r13 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 4) * 3 + x_rgb_out + 2 * 3));
				r14 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 4) * 3 + x_rgb_out + 3 * 3));
				r00 = _mm256_lddqu_si256((__m256i*)(imgSrcExt + imgWidthF * (y_out + 4) * 3 + x_rgb_out + 4 * 3));
				// values:		A24, A01, A02, A03, A04, A05, A06, A16, A17, A18, A19, A20, A21, A22, A23
				// registers:	r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14

				// sort the first 8 of the new 9 values
				sort8(r07, r08, r09, r10, r11, r12, r13, r14, tmp)

                // sort value pairs		(A16,A24), (A20,A24), (A18,A20), (A19,A21), (A22,A24)
                // sort register pairs	(r07,r00), (r11,r00), (r09,r11), (r10,r12), (r13,r00)
                sort2(r07, r00, tmp)
                sort2(r11, r00, tmp)
                sort2(r09, r11, tmp)
                sort2(r10, r12, tmp)
                sort2(r13, r00, tmp)

                // sort value pairs		(A17,A18), (A19,A20), (A21,A22), (A23,A24)
                // sort register pairs	(r08,r09), (r10,r11), (r12,r13), (r14,r00)
                sort2(r08, r09, tmp)
                sort2(r10, r11, tmp)
                sort2(r12, r13, tmp)
                sort2(r14, r00, tmp)



                /* merge first 16 and last 9 sorted elements, find median */

                // store A24 from r00 and load A00 to r00
                _mm256_store_si256(arr + 24, r00);
				r00 = _mm256_load_si256(arr + 0);
				// values:		A00, A01, A02, A03, A04, A05, A06, A16, A17, A18, A19, A20, A21, A22, A23
				// registers:	r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14

				// sort value pairs		(A00,A16), ... , (A06,A22)
				// sort register pairs	(r00,r07), ... , (r06,r13)
				sort2(r00, r07, tmp)
                sort2(r01, r08, tmp)
                sort2(r02, r09, tmp)
                sort2(r03, r10, tmp)
                sort2(r04, r11, tmp)
                sort2(r05, r12, tmp)
                sort2(r06, r13, tmp)

                // store A06 from r06; load A07...A13 to regs r00...r06; load A24 to r13
                // values in r00...r05,r13 don't need to be stored, these regs can be reused
                _mm256_store_si256(arr + 6, r06);
				r00 = _mm256_load_si256(arr + 7);
				r01 = _mm256_load_si256(arr + 8);
				r02 = _mm256_load_si256(arr + 9);
				r03 = _mm256_load_si256(arr + 10);
				r04 = _mm256_load_si256(arr + 11);
				r05 = _mm256_load_si256(arr + 12);
				r06 = _mm256_load_si256(arr + 13);
				r13 = _mm256_load_si256(arr + 24);
				// values:		A07, A08, A09, A10, A11, A12, A13, A16, A17, A18, A19, A20, A21, A24, A23
				// registers:	r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14

				// sort value pairs		(A07,A23), (A08,A24)
				// sort register pairs	(r00,r14), (r01,r13)
				sort2(r00, r14, tmp)
                sort2(r01, r13, tmp)

                // load A06 into r14
                // value in r14 don't need to be stored, this reg can be reused
                r14 = _mm256_load_si256(arr + 6);
				// values:		A07, A08, A09, A10, A11, A12, A13, A16, A17, A18, A19, A20, A21, A24, A06
				// registers:	r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14

				// sort value pairs		(A08,A16), ..., (A13,A21)
				// sort register pairs	(r01,r07), ..., (r06,r12)
				sort2(r01, r07, tmp)
                sort2(r02, r08, tmp)
                sort2(r03, r09, tmp)
                sort2(r04, r10, tmp)
                sort2(r05, r11, tmp)
                sort2(r06, r12, tmp)

                // sort value pairs		(A12,A16), (A13,A17), (A06,A10), (A07,A11)
                // sort register pairs	(r05,r07), (r06,r08), (r14,r03), (r00,r04)
                sort2(r05, r07, tmp)
                sort2(r06, r08, tmp)
                sort2(r14, r03, tmp)
                sort2(r00, r04, tmp)

                // sort value pairs		(A10,A12), (A11,A13), (A11,A12)
                // sort register pairs	(r03,r05), (r04,r06), (r04,r05)
                sort2(r03, r05, tmp)
                sort2(r04, r06, tmp)
                sort2(r04, r05, tmp)



                /* median is in r05 */
                _mm256_storeu_si256((__m256i*)(imgDst + imgWidth * y_out * 3 + x_rgb_out), r05);

			}
		}
		_mm_free(arr);
	}
	
}


/* intrinsic functions used */
// max:				__m256i _mm256_max_epu8 (__m256i a, __m256i b) // Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst.
// min:				__m256i _mm256_min_epu8 (__m256i a, __m256i b) // Compare packed unsigned 8-bit integers in a and b, and store packed minimum values in dst.
// unaligned load:	__m256i _mm256_lddqu_si256 (__m256i const * mem_addr) // Load 256-bits of integer data from unaligned memory into dst. This intrinsic may perform better than _mm256_loadu_si256 when the data crosses a cache line boundary.
// unaligned store:	void _mm256_storeu_si256 (__m256i * mem_addr, __m256i a) // Store 256-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
// aligned load:	__m256i _mm256_load_si256 (__m256i const * mem_addr) // Load 256-bits of integer data from memory into dst. mem_addr must be aligned on a 32-byte boundary or a general-protection exception may be generated.
// aligned store:	void _mm256_store_si256 (__m256i * mem_addr, __m256i a) // Store 256-bits of integer data from a into memory. mem_addr must be aligned on a 32-byte boundary or a general-protection exception may be generated.
// aligned malloc:	void* _mm_malloc (size_t size, size_t align) // Allocate size bytes of memory, aligned to the alignment specified in align, and return a pointer to the allocated memory. _mm_free should be used to free memory that is allocated with _mm_malloc.
// aligned free:	void _mm_free (void * mem_addr) // Free aligned memory that was allocated with _mm_malloc.